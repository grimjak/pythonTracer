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

#include <msgpack.hpp>


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

static int depth_max = 10;

typedef struct PixelSample {
  int o, w, i, j; 
  Vec3 t;
  PixelSample(){};
  PixelSample(int offset, int weight, int ii,int jj, Vec3 throughput):o(offset),w(weight),i(ii),j(jj),t(throughput){}
} PixelSample;

typedef struct TRay {float pdf; 
                    int depth; 
                    Vec3 o, d;
                    TRay(){};
                    TRay(Vec3 origin, Vec3 direction):pdf(1.0),depth(0),o(origin),d(direction){};
                    TRay(float p, int d, Vec3 origin, Vec3 direction):pdf(p),depth(d),o(origin),d(direction){};

} TRay;

// Routines to convert to and from messagepacks
// User defined class template specialization
namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template<>
struct convert<PixelSample> {
    msgpack::object const& operator()(msgpack::object const& o, PixelSample& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 7) throw msgpack::type_error();
        v = PixelSample(
            o.via.array.ptr[0].as<int>(),
            o.via.array.ptr[1].as<int>(),
            o.via.array.ptr[2].as<int>(),
            o.via.array.ptr[3].as<int>(),
            Vec3(
              o.via.array.ptr[4].as<float>(),
              o.via.array.ptr[5].as<float>(),
              o.via.array.ptr[6].as<float>()           
            )
            );

        return o;
    }
};

template<>
struct pack<PixelSample> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, PixelSample const& v) const {
        // packing member variables as an array.
        o.pack_array(7);
        o.pack(v.o);o.pack(v.w);o.pack(v.i);o.pack(v.j);
        o.pack(v.t.x);o.pack(v.t.y);o.pack(v.t.z);
        return o;
    }
};

template<>
struct convert<TRay> {
    msgpack::object const& operator()(msgpack::object const& o, TRay& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 8) throw msgpack::type_error();
        v = TRay(
            o.via.array.ptr[0].as<float>(),
            o.via.array.ptr[1].as<int>(),
            Vec3(
              o.via.array.ptr[2].as<float>(),
              o.via.array.ptr[3].as<float>(),
              o.via.array.ptr[4].as<float>()           
            ),
            Vec3(
              o.via.array.ptr[5].as<float>(),
              o.via.array.ptr[6].as<float>(),
              o.via.array.ptr[7].as<float>()           
            )
            );

        return o;
    }
};

template<>
struct pack<TRay> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, TRay const& v) const {
        // packing member variables as an array.
        o.pack_array(8);
        o.pack(v.pdf);o.pack(v.depth);
        o.pack(v.o.x);o.pack(v.o.y);o.pack(v.o.z);
        o.pack(v.d.x);o.pack(v.d.y);o.pack(v.d.z);

        return o;
    }
};

template<>
struct convert<Vec3> {
    msgpack::object const& operator()(msgpack::object const& o, Vec3& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 3) throw msgpack::type_error();
        v = Vec3(
              o.via.array.ptr[0].as<float>(),
              o.via.array.ptr[1].as<float>(),
              o.via.array.ptr[2].as<float>()           
            );

        return o;
    }
};

template<>
struct pack<Vec3> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, Vec3 const& v) const {
        // packing member variables as an array.
        o.pack_array(3);
        o.pack(v.x);o.pack(v.y);o.pack(v.z);

        return o;
    }
};

}
}
}


typedef struct ShadeJob {
  PixelSample ps; 
  Vec3 P;
  Vec3 N;
  TRay tray;
  unsigned int materialid;
  ShadeJob(){};
  ShadeJob(PixelSample p,Vec3 pp,Vec3 nn,TRay t,unsigned int mid):ps(p),P(pp),N(nn),tray(t),materialid(mid)
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

        if (tray.depth >= depth_max) continue;

        unsigned int materialid = rj.materialid;
        Vec3 wi = Vec3(tray.d.x,tray.d.y,tray.d.z);
        Vec3 Cs = Vec3(1,1,1);
        switch(materialid)
        {
          case 1: Cs = Vec3(1,0,0); break;
          case 2: Cs = Vec3(0,1,0); break;
          case 3: Cs = Vec3(0,0,1); break;
          case 4: Cs = Vec3(1,1,1); break;

        }
        
        //do shading
        //direct lighting
        //Vec3 L = Vec3(5,5,-10);
       // Vec3 L = Vec3(100,540,400);
       //random point on square
        index++;
        float lx = sobol::sample(index+ps.o,0);
        float ly = sobol::sample(index+ps.o,1);
        Vec3 L = Vec3(213+(lx*((343-213)/2)),548,227+(ly*((332-227)/2)));
        std::cerr << L << endl;
       // Vec3 L = Vec3(235,540,235);
        Vec3 color_light(1.5,1.5,1.5);
        //        Vec3 color_light(1,1,1);

        //TRay rayToLight(P+N*0.0001,(L-P).normalize());
        Vec3 lightVec = (L-P);
        float len = lightVec.length();
        //TRay rayToLight(P+N*0.0001,lightVec.normalize());
        TRay rayToLight(P+N*0.0001,lightVec.normalize());

        //TRay rayToLight(P+N*0.0001,Vec3(0,1,0));

        //calculate radiance here and pass it to be added if ray hits?
        Vec3 rad = ps.t * color_light * Cs * N.dot(rayToLight.d) * M_1_PI; //separate PDF
       // Vec3 rad = ps.t * color_light * Cs; //separate PDF

        //rad = N;

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

        //russian roulette
        float p = std::max(t.x,std::max(t.y,t.z));
        index++;
        if(p < sobol::sample(index,0)) continue;

        t *= 1/p;

        TRay ray(P+N*0.0001,out_dir.normalize());
        ray.depth = tray.depth+1;
        ray.pdf = pdf;
       // ray.pdf = std::max(N.normalize().dot(out_dir.normalize()), 0.000000f) * M_1_PI;

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
    msgpack::unpacker pac;
    pac.reserve_buffer(j);
    memcpy(pac.buffer(), data, j);
    pac.buffer_consumed(j);
    msgpack::object_handle oh;
    
    PixelSample ps;
    TRay tray;
    Vec3 P,N,Cs;
    unsigned int materialid;

    pac.next(oh);
    std::cout << oh.get() << std::endl;
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
/*
    while(pac.next(oh)) {
      std::cout << oh.get() << std::endl;
    }

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
    
    unsigned int materialid = document["materialid"].GetInt();
*/
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
}