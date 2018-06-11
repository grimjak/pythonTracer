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

#include <random>
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

static Vec3 O = Vec3(0,0.35,-1);
static int w = 640;
static int h = 480;
static float r = (float)w/(float)h;
static float filterwidth = 6.0;


typedef struct PixelSample {
  int o, w, i, j; 
  Vec3 t;
  PixelSample(){};
  PixelSample(int offset, int weight, int ii,int jj, Vec3 throughput):o(offset),w(weight),i(ii),j(jj),t(throughput){}
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


concurrent_queue<ShadeJob> shadeworkqueue;

//generate a single sample for a given set of coordinates
//x and y in NDC, 0 to 1
//need to define O
TRay generateSample(float x, float y)
{
  Vec3 Q = Vec3(x,y,0);
  Vec3 D = (Q-O).normalize();
  return TRay(O,D);
}

//generate one iterations worth of samples
//need to define w and h and filterwidth
// iterate over pixels, need to keep track of index
void iterate(int &index, int iteration,float *offsets)
{
  std::string host ="rabbitmq";
  std::string rayqueue = "rayqueue";

  AMQP amqp(host);
  AMQPExchange *ex = amqp.createExchange("ptex");
  ex->Declare("ptex","direct");

  //need an offset per pixel

  for (int i = 0; i<w; i++)
  {

    for (int j = 0; j<h; j++)
    {
      cerr << i << "," << j << ":" << offsets[j*w+i] << endl;
      cerr << j*w+i << endl;
      float rx = sobol::sample(iteration,0);
      float ry = sobol::sample(iteration,1);
      float offset = offsets[j*w+i];
      rx = (rx+offset);if (rx > 1) rx-=1;
      ry = (ry+offset);if (ry > 1) ry-=1;


      cerr << rx <<","<<ry << endl;
      float offsetu = filterwidth * rx - (filterwidth*0.5);
      float offsetv = filterwidth * ry - (filterwidth*0.5);


      float x = 2*((float)i / (float)w - 0.5);
      float y =2*((float)j / (float)h - 0.5); 
      y/=r;

      x += offsetu/w;
      y += offsetv/w;
      TRay r = generateSample(x,y);
      int o = rand()%(256);

      PixelSample ps = PixelSample(o,iteration+1,i,j,Vec3(1,1,1));

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
      writer.Double(1.0);
      writer.Key("depth");
      writer.Int(0);
      writer.Key("o");
      writer.StartArray();
      writer.Double(r.o.x);writer.Double(r.o.y);writer.Double(r.o.z);
      writer.EndArray();
      writer.Key("d");
      writer.StartArray();
      writer.Double(r.d.x);writer.Double(r.d.y);writer.Double(r.d.z);
      writer.EndArray();
      writer.EndObject();
      writer.EndObject();

      ex->Publish(s.GetString(),rayqueue);

    }
  }
}

int onCancel(AMQPMessage * message ) {
	cout << "cancel tag="<< message->getDeliveryTag() << endl;
	return 0;
}

int  raygenMessageHandler( AMQPMessage * message  ) 
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

//wait for a command to send some rays then send a batch of samples
//each sample is importance sampled according to the filter being used.

int main(int argc, char *argv[])
{
  cout << "Subscriber started" << endl;

  std::string host ="rabbitmq";
  std::string queuename = "raygenqueue";

  //added delay for rabbit mq start to avoid failing with socket error
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));

  /*  
  //thread for tracing rays
  const int numthreads = 1;
  std::thread shadeworkthreads[numthreads];
  //launch threads

  for(int i=0;i<numthreads;++i) shadeworkthreads[i] = std::thread(shadeworker,i);
  */

  //for now just start generating rays
  int index = 0;
  int samples = 256;
  float offsets[w*h];

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0,1);

  for(int i = 0; i < w*h; i++)
    offsets[i] = dist(e2);
    
  for(int s = 0; s < samples; s++)
  {
    iterate(index,s,&offsets[0]);
  }
  cerr << "index = " << index << endl;
  try {
		AMQP amqp(host);
    //add occlusion queue 
		AMQPQueue * queue = amqp.createQueue(queuename);

		queue->Declare();
		//qu2->Bind( "", "");

		queue->setConsumerTag("tag_123");
    queue->addEvent(AMQP_MESSAGE, raygenMessageHandler);
		queue->addEvent(AMQP_CANCEL, onCancel );

		queue->Consume(AMQP_NOACK);//

	} catch (AMQPException e) {
		std::cout << e.getMessage() << std::endl;
	}
  cout << "OK"<<endl;    
}