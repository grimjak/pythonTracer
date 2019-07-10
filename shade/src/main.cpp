<<<<<<< HEAD
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

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include <tbb/concurrent_queue.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathColor.h>


#include <OpenImageIO/fmath.h>

#include "sobol.h"

using namespace std;
using namespace rapidjson;
using namespace tbb;

typedef float Float;
typedef Imath_2_2::Vec3<Float>     Vec3;
typedef Imath_2_2::Matrix33<Float> Matrix33;
typedef Imath_2_2::Matrix44<Float> Matrix44;
typedef Imath_2_2::Color3<Float>   Color3;
typedef Imath_2_2::Vec2<Float>     Vec2;

static int depth_max = 0;

typedef struct PixelSample {
  int o, w, i, j; 
  Vec3 t;
  //void Serialize(Writer &writer)
  //{

  //}
  } PixelSample;
typedef struct TRay {float pdf; 
                    int depth; 
                    Vec3 o, d;
                    TRay(){};
                    TRay(Vec3 origin, Vec3 direction):o(origin),d(direction){};
} TRay;

typedef struct ShadeJob {
  PixelSample ps; 
  Vec3 P;
  Vec3 N;
  TRay tray;
  ShadeJob(){};
  ShadeJob(PixelSample p,Vec3 pp,Vec3 nn,TRay t):ps(p),P(pp),N(nn),tray(t)
  {};
} ShadeJob;

struct TangentFrame {
    // build frame from unit normal
    TangentFrame(const Vec3& n) : w(n) {
        u = (fabsf(w.x) >.01f ? Vec3(w.z, 0, -w.x) :
                                Vec3(0, -w.z, w.y)).normalize();
        v = w.cross(u);
    }

    // build frame from unit normal and unit tangent
    TangentFrame(const Vec3& n, const Vec3& t) : w(n) {
        v = w.cross(t);
        u = v.cross(w);
    }

    // transform vector
    Vec3 get(float x, float y, float z) const {
        return x * u + y * v + z * w;
    }

    // untransform vector
    float getx(const Vec3& a) const { return a.dot(u); }
    float gety(const Vec3& a) const { return a.dot(v); }
    float getz(const Vec3& a) const { return a.dot(w); }

    Vec3 tolocal(const Vec3 &a) const {
      return Vec3(a.dot(u), a.dot(v), a.dot(w));
    }
    Vec3 toworld(const Vec3 &a) const {
      return get(a.x, a.y, a.z);
    }

private:
    Vec3 u, v, w;
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

    static void sample_cosine_hemisphere(const Vec3& N, float rndx, float rndy, Vec3& out, float& pdf) {
        to_unit_disk(rndx, rndy);
        float cos_theta = sqrtf(std::max(1 - rndx * rndx - rndy * rndy, 0.0f));
        TangentFrame f(N);
        out = f.get(rndx, rndy, cos_theta);
        pdf = cos_theta * float(M_1_PI);
    }
};

concurrent_queue<ShadeJob> shadeworkqueue;


void shadeworker(int tid)
{
    PixelSample ps;
    Vec3 P;
    Vec3 N;
    TRay tray;
    ShadeJob rj;

    std::string host ="rabbitmq";
    std::string occlusionqueue = "occlusionqueue";
    std::string rayqueue = "rayqueue";

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    cout << "shade worker: " << tid << " started" << endl;
    int index = 0;
    while(true)
    {
      if (shadeworkqueue.try_pop(rj))
      {
        ps = rj.ps;
        P = rj.P;
        N = rj.N.normalize();
        tray = rj.tray;
        Vec3 wi = Vec3(tray.d.x,tray.d.y,tray.d.z);
        Vec3 Cs = Vec3(1,1,1);
        
        //do shading
        //direct lighting
        //Vec3 L = Vec3(5,5,-10);
        Vec3 L = Vec3(100,540,400);
       // Vec3 color_light(20,20,20);
                Vec3 color_light(1,1,1);

        //TRay rayToLight(P+N*0.0001,(L-P).normalize());
        Vec3 lightVec = (L-P);
        float len = lightVec.length();
        //TRay rayToLight(P+N*0.0001,lightVec.normalize());
        TRay rayToLight(P,lightVec.normalize());

        //TRay rayToLight(P+N*0.0001,Vec3(0,1,0));

        //calculate radiance here and pass it to be added if ray hits?
        Vec3 rad = ps.t * color_light * Cs * N.dot(rayToLight.d) * M_1_PI; //separate PDF
       // Vec3 rad = ps.t * color_light * Cs; //separate PDF

        rad = Vec3(0.5,1,0.5);

        StringBuffer s;
        Writer<StringBuffer> writer(s);
        writer.StartObject();
        writer.Key("ps");
        writer.StartObject();
        writer.Key("i");
        writer.Int(ps.i);
        writer.Key("j");
        writer.Int(ps.j);
        writer.Key("o");
        writer.Int(ps.o);
        writer.Key("w");
        writer.Int(ps.w);
        writer.Key("t");
        writer.StartArray();
        writer.Double(ps.t.x);writer.Double(ps.t.y);writer.Double(ps.t.z);
        writer.EndArray();
        writer.EndObject();

        writer.Key("ray");
        writer.StartObject();
        writer.Key("pdf");
        writer.Double(tray.pdf);
        writer.Key("depth");
        writer.Int(tray.depth);
        writer.Key("o");
        writer.StartArray();
        writer.Double(rayToLight.o.x);writer.Double(rayToLight.o.y);writer.Double(rayToLight.o.z);
        writer.EndArray();
        writer.Key("d");
        writer.StartArray();
        writer.Double(rayToLight.d.x);writer.Double(rayToLight.d.y);writer.Double(rayToLight.d.z);
        writer.EndArray();
        writer.EndObject();

        writer.Key("len");
        writer.Double(len);
        writer.Key("rad");
        writer.StartArray();
        writer.Double(rad.x);writer.Double(rad.y);writer.Double(rad.z);
        writer.EndArray();
        writer.EndObject();

        ex->Publish(s.GetString(),occlusionqueue);

        //indirect ray
       // Vec3 wi = tray.d;
        float pdf = 1;
        Vec3 out_dir;
        index++;
        float rx = sobol::sample(index+ps.o,0);
        float ry = sobol::sample(index+ps.o,1);
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        //pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI); //incoming pdf

        Vec3 t = Vec3(ps.t.x,ps.t.y,ps.t.z);
        t *= Cs * tray.pdf;
        TRay ray(P+N*0.0001,out_dir.normalize());
        ray.depth = tray.depth+1;
        ray.pdf = pdf;
       // ray.pdf = std::max(N.normalize().dot(out_dir.normalize()), 0.000000f) * M_1_PI;

        if (ray.depth > depth_max) continue;

        StringBuffer s2;
        writer.Reset(s2);
        writer.StartObject();
        writer.Key("ps");
        writer.StartObject();
        writer.Key("i");
        writer.Int(ps.i);
        writer.Key("j");
        writer.Int(ps.j);
        writer.Key("o");
        writer.Int(ps.o);
        writer.Key("w");
        writer.Int(ps.w);
        writer.Key("t");
        writer.StartArray();
        writer.Double(t.x);writer.Double(t.y);writer.Double(t.z);
        writer.EndArray();
        writer.EndObject();

        writer.Key("ray");
        writer.StartObject();
        writer.Key("pdf");
        writer.Double(ray.pdf);
        writer.Key("depth");
        writer.Int(ray.depth);
        writer.Key("o");
        writer.StartArray();
        writer.Double(ray.o.x);writer.Double(ray.o.y);writer.Double(ray.o.z);
        writer.EndArray();
        writer.Key("d");
        writer.StartArray();
        writer.Double(ray.d.x);writer.Double(ray.d.y);writer.Double(ray.d.z);
        writer.EndArray();
        writer.EndObject();
        writer.EndObject();

        //cerr<<"occlusion ray: " << s.GetString() << endl;
        //cerr<<"publishing indirect ray: " << s2.GetString() << " to " <<rayqueue<<endl;
        ex->Publish(s2.GetString(),rayqueue);

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
    Document document;
    document.Parse(data);

    PixelSample ps;
    ps.i = document["ps"]["i"].GetInt();
    ps.j = document["ps"]["j"].GetInt();
    ps.o = document["ps"]["o"].GetInt();
    ps.w = document["ps"]["w"].GetInt();
    Vec3 t;
    t.x = document["ps"]["t"][0].GetFloat();t.y = document["ps"]["t"][1].GetFloat();t.z = document["ps"]["t"][2].GetFloat();
    ps.t = t;

    //Need to clean this whole section up so we don't have multiple copies of ray data
    TRay tray;
    tray.pdf = document["ray"]["pdf"].GetFloat();
    tray.depth = document["ray"]["depth"].GetFloat();
    Vec3 o;
    o.x = document["ray"]["o"][0].GetFloat();o.y = document["ray"]["o"][1].GetFloat();o.z = document["ray"]["o"][2].GetFloat();
    tray.o = o;
    Vec3 d;
    d.x = document["ray"]["d"][0].GetFloat();d.y = document["ray"]["d"][1].GetFloat();d.z = document["ray"]["d"][2].GetFloat();
    tray.d = d;

    Vec3 N;
    N.x = document["N"][0].GetFloat();N.y = document["N"][1].GetFloat();N.z = document["N"][2].GetFloat();
    Vec3 P;
    P.x = document["P"][0].GetFloat();P.y = document["P"][1].GetFloat();P.z = document["P"][2].GetFloat();
    
    

    shadeworkqueue.push( ShadeJob(ps,P,N,tray));
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
  const int numthreads = 2;
  std::thread shadeworkthreads[numthreads];
  //launch threads

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
=======
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

concurrent_queue<ShadeJob> shadeworkqueue;


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

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    AMQPQueue * queue = amqp.createQueue(occlusionqueue);
    queue->Declare();
	  queue->Bind( "ptex", occlusionqueue);

    AMQPQueue * queue2 = amqp.createQueue(rayqueue);
    queue2->Declare();
	  queue2->Bind( "ptex", rayqueue);

    cout << "shade worker: " << tid << " started" << endl;
    int index = 0;

    msgpack::sbuffer ss;
    msgpack::packer<msgpack::sbuffer> pk(&ss);

    hostent * record = gethostbyname("influxdb");
    in_addr * address = (in_addr * )record->h_addr;
	  string ip_address = inet_ntoa(* address);
    influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");
    Timer tmr;

    float thisSampleTime = 0;
    float thisPacketTime = 0;
    while(true)
    {
      if (shadeworkqueue.try_pop(rj))
      {
        ps = rj.ps;
        P = rj.P;
        N = rj.N.normalize();
        tray = rj.tray;

        if (tray.depth >= depth_max) continue;

        numSamples++;
        tmr.reset();

        unsigned int materialid = rj.materialid;
        Vec3f wi = Vec3f(tray.d.x,tray.d.y,tray.d.z);
        Vec3f Cs = Vec3f(1,1,1);

        switch(materialid)
        {
          case 1: Cs = Vec3f(1,0,0); break;
          case 2: Cs = Vec3f(0,1,0); break;
          case 3: Cs = Vec3f(0,0,1); break;
          case 4: Cs = Vec3f(1,1,1); break;

        }
        
        //do shading
        //direct lighting
        //Vec3f L = Vec3f(5,5,-10);
       // Vec3f L = Vec3f(100,540,400);
       //random point on square
        index++;
        float lx = sobol::sample(index+ps.o,0);
        float ly = sobol::sample(index+ps.o,1);
        Vec3f L = Vec3f(213+(lx*((343-213)/2)),548,227+(ly*((332-227)/2)));
       // Vec3f L = Vec3f(235,540,235);
        Vec3f color_light(1.5,1.5,1.5);
        //        Vec3f color_light(1,1,1);

        //TRay rayToLight(P+N*0.0001,(L-P).normalize());
        Vec3f lightVec = (L-P);
        float len = lightVec.length();
        //TRay rayToLight(P+N*0.0001,lightVec.normalize());
        TRay rayToLight(tray.pdf,tray.depth,P+N*0.0001,lightVec.normalize());

        //TRay rayToLight(P+N*0.0001,Vec3f(0,1,0));

        //calculate radiance here and pass it to be added if ray hits?
        Vec3f rad = ps.t * color_light * Cs * N.dot(rayToLight.d) * M_1_PI; //separate PDF
       // Vec3f rad = ps.t * color_light * Cs; //separate PDF

        //rad = N;
        thisSampleTime = tmr.elapsed();
        totalSampleTime+=thisSampleTime;
        tmr.reset();

        numPacketsOut++;
        ss.clear();
        pk.pack(ps);
        pk.pack(rayToLight);
        pk.pack(len);
        pk.pack(rad);
        ex->Publish(ss.data(),ss.size(),occlusionqueue);

        thisPacketTime = tmr.elapsed();
        totalPacketsOutTime+=thisPacketTime;
        tmr.reset();
        //indirect ray
       // Vec3f wi = tray.d;
        float pdf = 1;
        Vec3f out_dir;
        index++;
        float rx = sobol::sample(index+ps.o,0);
        float ry = sobol::sample(index+ps.o,1);
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        //pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI); //incoming pdf

        Vec3f t = Vec3f(ps.t.x,ps.t.y,ps.t.z);
        t *= Cs * tray.pdf;

        //russian roulette
       /* float p = std::max(t.x,std::max(t.y,t.z));
        index++;
        if(p < sobol::sample(index,0))
        {
          thisSampleTime += tmr.elapsed();
          totalSampleTime+=thisSampleTime;          
          tmr.reset();
          continue; //need some kind of signal to say we've stopped.  Could just write 0 rad to radiance buffer?
        }
        t *= 1/p;
*/
        TRay ray(P+N*0.0001,out_dir.normalize());
        ray.depth = tray.depth+1;
        ray.pdf = pdf;
       // ray.pdf = std::max(N.normalize().dot(out_dir.normalize()), 0.000000f) * M_1_PI;

        ps.t = t;

        thisSampleTime += tmr.elapsed();
        totalSampleTime+=thisSampleTime;
        tmr.reset();
        
        ss.clear();
        pk.pack(ps);
        pk.pack(ray);
        ex->Publish(ss.data(),ss.size(),rayqueue);

        thisPacketTime += tmr.elapsed();
        totalPacketsOutTime+=thisPacketTime;

        if ((numSamples%1000 == 0) && (numSamples > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("shadeserver")
          .tag("name", "totalSamples")
          .field("numSamples", numSamples)
          .field("totalSampleTime", totalSampleTime)
          .field("samplesPerSecond", 1.0 / thisSampleTime)
          .post_http(si);
        }
        if ((numPacketsOut%1000 == 0) && (numPacketsOut > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("shadeserver")
          .tag("name", "totalPacketsOut")
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
    Vec3f P,N,Cs;
    unsigned int materialid;

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
  const int numthreads = 2;
  std::thread shadeworkthreads[numthreads];
  //launch threads

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
>>>>>>> 6d56b8e31cc975bfe21cc0fcd57a09595e374fcd
}