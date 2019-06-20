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

static int depth_max = 4;

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
        N = rj.N;
        tray = rj.tray;
        Vec3 Cs = Vec3(1,1,1);
        
        //do shading
        //direct lighting
        Vec3 L = Vec3(5,5,-10);
        Vec3 color_light(1,1,1);
        TRay rayToLight(P+N*0.0001,(L-P).normalize());
        //calculate radiance here and pass it to be added if ray hits?
        Vec3 rad = ps.t * color_light * Cs * N.dot(rayToLight.d) * M_1_PI; //separate PDF

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

        writer.Key("rad");
        writer.StartArray();
        writer.Double(rad.x);writer.Double(rad.y);writer.Double(rad.z);
        writer.EndArray();
        writer.EndObject();

        ex->Publish(s.GetString(),occlusionqueue);

        //indirect ray
       // Vec3 wi = tray.d;
        float pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI);
        Vec3 out_dir;
        index++;
        float rx = sobol::sample(index+ps.o,0);
        float ry = sobol::sample(index+ps.o,1);
        cerr << rx << ", " << ry << endl;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        ps.t *= Cs / 1.0;
        TRay ray(P+N*0.0001,out_dir.normalize());
        ray.depth = tray.depth+1;
        ray.pdf = 1;

        //if (ray.depth > depth_max) continue;


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
        writer.Double(ps.t.x);writer.Double(ps.t.y);writer.Double(ps.t.z);
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

        cerr<<"occlusion ray: " << s.GetString() << endl;
        cerr<<"publishing indirect ray: " << s2.GetString() << " to " <<rayqueue<<endl;
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
}