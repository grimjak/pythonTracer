#include <stdlib.h>
#include <stdio.h>
#include "AMQPcpp.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <tuple>

//#include <boost/uuid/uuid_io.hpp>
//#include <boost/uuid.hpp>
//#include <boost/uuid/uuid_generators.hpp>

//#include "rapidjson/document.h"
//#include "rapidjson/writer.h"
//#include "rapidjson/stringbuffer.h"

#include <msgpack.hpp>


#include <tbb/concurrent_queue.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathColor.h>


#include <OpenImageIO/fmath.h>

#include <influxdb.hpp>
#include <netdb.h>
#include <unistd.h>
#include <limits.h>

#define IMATH
#include "messaging.h"

#include "sobol.h"

using namespace std;
//using namespace rapidjson;
using namespace tbb;

static int depth_max = 10;

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

int numSamples = 0;
float totalSampleTime = 0;
int numPacketsIn = 0;
int numPacketsOut = 0;
float totalPacketsInTime = 0;
float totalPacketsOutTime = 0;
unsigned int msgBatchSize = 144;



typedef struct ShadeJob {
  PixelSample ps; 
  Vec3f P;
  Vec3f N;
  TRay tray;
  unsigned int materialid;
  ShadeJob(){};
  ShadeJob(PixelSample p,Vec3f pp,Vec3f nn,TRay t,unsigned int mid):ps(p),P(pp),N(nn),tray(t),materialid(mid)
  {};
} ShadeJob;

struct TangentFrame {
    // build frame from unit normal
    TangentFrame(const Vec3f& n) : w(n) {
        u = (fabsf(w.x) >.01f ? Vec3f(w.z, 0, -w.x) :
                                Vec3f(0, -w.z, w.y)).normalize();
        v = w.cross(u);
    }

    // build frame from unit normal and unit tangent
    TangentFrame(const Vec3f& n, const Vec3f& t) : w(n) {
        v = w.cross(t);
        u = v.cross(w);
    }

    // transform vector
    Vec3f get(float x, float y, float z) const {
        return x * u + y * v + z * w;
    }

    // untransform vector
    float getx(const Vec3f& a) const { return a.dot(u); }
    float gety(const Vec3f& a) const { return a.dot(v); }
    float getz(const Vec3f& a) const { return a.dot(w); }

    Vec3f tolocal(const Vec3f &a) const {
      return Vec3f(a.dot(u), a.dot(v), a.dot(w));
    }
    Vec3f toworld(const Vec3f &a) const {
      return get(a.x, a.y, a.z);
    }

private:
    Vec3f u, v, w;
};


struct Sampling {
    /// Warp the unit disk onto the unit sphere
    /// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    static void to_unit_disk(float& x, float& y) {
        const float PI_OVER_4 = float(M_PI_4);
        const float PI_OVER_2 = float(M_PI_2);
        float phi, r;
        float a = 2 * x - 1;
        float b = 2 * y - 1;
        if (a * a > b * b) { // use squares instead of absolute values
            r = a;
            phi = PI_OVER_4 * (b / a);
        } else if (b != 0) { // b is largest
            r = b;
            phi = PI_OVER_2 - PI_OVER_4 * (a / b);
        } else { // a == b == 0
            r = 0;
            phi = 0;
        }
        OIIO::fast_sincos(phi, &x, &y);
        x *= r;
        y *= r;
    }

    static void sample_cosine_hemisphere(const Vec3f& N, float rndx, float rndy, Vec3f& out, float& pdf) {
        to_unit_disk(rndx, rndy);
        float cos_theta = sqrtf(std::max(1 - rndx * rndx - rndy * rndy, 0.0f));
        TangentFrame f(N);
        out = f.get(rndx, rndy, cos_theta);
        pdf = cos_theta * float(M_1_PI);
    }
};

concurrent_bounded_queue<ShadeJob> shadeworkqueue;


void shadeworker(int tid)
{
    PixelSample ps;
    Vec3f P;
    Vec3f N;
    TRay tray;
    ShadeJob rj;

    std::string host ="rabbitmq";
    std::string occlusionqueue = "occlusionqueue";
    std::string rayqueue = "rayqueue";
    std::string radiancequeue = "radiancequeue";


    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    AMQPQueue * queue = amqp.createQueue(occlusionqueue);
    queue->Declare();
	  queue->Bind( "ptex", occlusionqueue);

    AMQPQueue * queue2 = amqp.createQueue(rayqueue);
    queue2->Declare();
	  queue2->Bind( "ptex", rayqueue);

    AMQPQueue * queue3 = amqp.createQueue(radiancequeue);
		queue3->Declare();
		queue3->Bind( "ptex", radiancequeue);

    cout << "shade worker: " << tid << " started" << endl;
    int index = 0;

    msgpack::sbuffer shadss;
    msgpack::packer<msgpack::sbuffer> shadpk(&shadss);

    msgpack::sbuffer rayss;
    msgpack::packer<msgpack::sbuffer> raypk(&rayss);  

    msgpack::sbuffer radss;
    msgpack::packer<msgpack::sbuffer> radpk(&radss);

    hostent * record = gethostbyname("influxdb");
    in_addr * address = (in_addr * )record->h_addr;
	  string ip_address = inet_ntoa(* address);
    influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");
    Timer tmr;

    float thisSampleTime = 0;
    float thisPacketTime = 0;
    unsigned int shadBatch = 0;
    unsigned int rayBatch = 0;
    unsigned int radBatch = 0;

    float totalDirectSampleTime = 0;
    float totalIndirectSampleTime = 0;

    //precompute sobol sequence?


    while(true)
    {
      //if (shadeworkqueue.try_pop(rj))
      shadeworkqueue.pop(rj);
      {
        ps = rj.ps;
        P = rj.P;
        N = rj.N.normalize();
        tray = rj.tray;

        //if (tray.depth >= depth_max) continue; //if we're terminating the ray we need to write the accumulated radiance

        numSamples++;
        tmr.reset();

        unsigned int materialid = rj.materialid;
        Vec3f wi = Vec3f(tray.d.x,tray.d.y,tray.d.z);
        Vec3f Cs = Vec3f(1,1,1);
        Vec3f Ca = Vec3f(0,0,0); //emmission

        switch(materialid)
        {
          case 0: Cs = Vec3f(0,0.15,1); Ca = Vec3f(0,0,0); break;
          case 1: Cs = Vec3f(0,1,0); Ca = Vec3f(0,0,0); break;
          case 2: Cs = Vec3f(1,1,1); Ca = Vec3f(50,50,50); break;
          case 3: Cs = Vec3f(1,0,0); Ca = Vec3f(0,0,0); break;
          case 4: Cs = Vec3f(1,1,1); break;
        }
        
        //what to do if we've hit an emmissive surface?  We could just pretend we're a light and return radiance.
        //or accumulate radiance and add it

        //if we hit a light and this is a camera ray

        //do shading
        //direct lighting
       //random point on square
        index++;
        float lx = sobol::sample(index+ps.o,0);
        float ly = sobol::sample(index+ps.o,1);
        Vec3f L = Vec3f(213+(lx*((343-213)/2)),548,227+(ly*((332-227)/2)));
        Vec3f color_light(1.5,1.5,1.5);

        Vec3f lightVec = (L-P);
        float len = lightVec.length();
        TRay rayToLight(tray.pdf,tray.depth+1,P+N*0.0001,lightVec.normalize());

        //calculate radiance here and pass it to be added if ray hits?
        Vec3f rad = ps.t * ((color_light * Cs * N.dot(rayToLight.d))) * M_1_PI; //separate PDF

        thisSampleTime = tmr.elapsed();
        totalSampleTime+=thisSampleTime;
        totalDirectSampleTime+=thisSampleTime;
        tmr.reset();

        numPacketsOut++;
        shadpk.pack(ps);
        shadpk.pack(rayToLight);
        shadpk.pack(len);
        shadpk.pack(rad);

        /* turn off direct lighting
        shadBatch++;
        if(shadBatch == msgBatchSize)
        {
          ex->Publish(shadss.data(),shadss.size(),occlusionqueue);
          shadss.clear();
          shadBatch=0;
        }
        */
        thisPacketTime = tmr.elapsed();
        totalPacketsOutTime+=thisPacketTime;
        tmr.reset();
        
        //indirect ray
  
        float pdf = 1;
        Vec3f out_dir;
        index++;
        float rx = sobol::sample(index+ps.o,0);
        float ry = sobol::sample(index+ps.o,1);
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);

        Vec3f t = Vec3f(ps.t.x,ps.t.y,ps.t.z); //accumulated trnsmittance for the sample, not including this bounce
        Vec3f r = Vec3f(ps.r.x,ps.r.y,ps.r.z); //current accumulated radiance for this sample
        t *= Cs * tray.pdf; //update transmittance with effect of this bounce
        r += Ca * t; //add radiance from current emmission if any
        ps.r = r;

        //russian roulette
        Vec3f tr = t+r;
        //float p = std::max(tr.x,std::max(tr.y,tr.z));
        float p = 1; //turn off 
        index++;
        if(p < sobol::sample(index,0))
        {
          cerr<<"russian roulette"<<endl;
          thisSampleTime += tmr.elapsed();
          totalSampleTime+=thisSampleTime;          
          tmr.reset();
          //need to write accumulated radiance to the radiance buffer
          radpk.pack(ps);
          radpk.pack(ps.r);
          radpk.pack(tray.depth);

          radBatch++;
          if(radBatch==msgBatchSize)
          {
            cerr<<"publishing from rr: "<< radss.size()<< endl;
            ex->Publish(radss.data(),radss.size(),radiancequeue);
            radss.clear();
            radBatch=0;
          }
          
          continue; //need some kind of signal to say we've stopped.  Could just write 0 rad to radiance buffer?
        }
        t *= 1/p;
        r *= 1/p;

        TRay ray(P+N*0.0001,out_dir.normalize());
        ray.depth = tray.depth+1;
        ray.pdf = pdf;
       // ray.pdf = std::max(N.normalize().dot(out_dir.normalize()), 0.000000f) * M_1_PI;

        ps.t = t;
       // ps.r = r;

        thisSampleTime += tmr.elapsed();
        totalSampleTime+=thisSampleTime;
        totalIndirectSampleTime+=tmr.elapsed();
        tmr.reset();
        
        raypk.pack(ps);
        raypk.pack(ray);

        rayBatch++;
        if(rayBatch==msgBatchSize)
        //if((rayBatch==msgBatchSize) && (ray.depth < 0))
        {
          ex->Publish(rayss.data(),rayss.size(),rayqueue);
          rayss.clear();
          rayBatch=0;
        }
        tmr.reset();
        thisPacketTime += tmr.elapsed();
        totalPacketsOutTime+=thisPacketTime;

        if ((numSamples%1000 == 0) && (numSamples > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("shadeserver")
          .tag("name", "totalSamples")
          .tag("hostname", hostname)
          .field("numSamples", numSamples)
          .field("totalSampleTime", totalSampleTime)
          .field("totalDirectSampleTime", totalDirectSampleTime)
          .field("totalIndirectSampleTime", totalIndirectSampleTime)          
          .field("samplesPerSecond", 1.0 / thisSampleTime)
          .post_http(si);
        }
        if ((numPacketsOut%1000 == 0) && (numPacketsOut > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("shadeserver")
          .tag("name", "totalPacketsOut")
          .tag("hostname", hostname)
          .field("num", numPacketsOut)
          .field("totalTime", totalPacketsOutTime)
          .field("packetsPerSecond", 1.0 / thisPacketTime)
          .post_http(si);
        }
      }      
    }

}

int onCancel(AMQPMessage * message ) {
	cout << "cancel tag="<< message->getDeliveryTag() << endl;
	return 0;
}

int  shadeMessageHandler( AMQPMessage * message  ) 
{
  uint32_t j = 0;
	char * data = message->getMessage(&j);
	if (data)
  {
    msgpack::unpacker pac;
    pac.reserve_buffer(j);
    memcpy(pac.buffer(), data, j);
    pac.buffer_consumed(j);
    msgpack::object_handle oh;
    
    PixelSample ps;
    TRay tray;
    Vec3f P,N,Cs,Rad;
    unsigned int materialid;

    for (int i = 0; i < msgBatchSize; i++)
    {
      pac.next(oh);
      //std::cout << oh.get() << std::endl;
      oh.get().convert(ps);
      pac.next(oh);
      oh.get().convert(tray);
      pac.next(oh);
      oh.get().convert(P);
      pac.next(oh);
      oh.get().convert(N); 
      pac.next(oh);
      oh.get().convert(Cs); 
      pac.next(oh);
      oh.get().convert(materialid);

      shadeworkqueue.push( ShadeJob(ps,P,N,tray,materialid));
    }
  }
  return 0;
}

int main(int argc, char *argv[])
{
  cout << "Subscriber started" << endl;

  std::string host ="rabbitmq";
  std::string queuename = "shadequeue";

  //added delay for rabbit mq start to avoid failing with socket error
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
  //thread for tracing rays
  const int numthreads = 1;
  std::thread shadeworkthreads[numthreads];
  //launch threads
  shadeworkqueue.set_capacity(msgBatchSize);

  for(int i=0;i<numthreads;++i) shadeworkthreads[i] = std::thread(shadeworker,i);
  
  try {
		AMQP amqp(host);
    //add occlusion queue 
		AMQPQueue * queue = amqp.createQueue(queuename);

		queue->Declare();
		//qu2->Bind( "", "");

		queue->setConsumerTag("tag_123");
    queue->addEvent(AMQP_MESSAGE, shadeMessageHandler);
		queue->addEvent(AMQP_CANCEL, onCancel );

		queue->Consume(AMQP_NOACK);//

	} catch (AMQPException e) {
		std::cout << e.getMessage() << std::endl;
	}
  cout << "OK"<<endl;    
}