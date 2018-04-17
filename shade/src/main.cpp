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

using namespace AmqpClient;
using namespace std;
using namespace rapidjson;
using namespace tbb;
 
typedef struct PixelSample {
  int o, w, i, j; 
  Vec3fa t;
  Serialize(Writer &writer)
  {

  }
  } PixelSample;
typedef struct TRay {float pdf; int depth; Vec3fa o, d;} TRay;

typedef struct ShadeJob {
  PixelSample ps; 
  Vec3f P;
  Vec3f N;
  TRay tray;
  RayJob(){};
  RayJob(PixelSample p,Vec3f pp,vec3f nn,TRay t):ps(p),P(pp),N(nn),tray(t)
  {};
} ShadeJob;

concurrent_queue<ShadeJob> shadeworkqueue;

void rayworker(int tid)
{
    PixelSample ps;
    Vec3f P;
    Vec3f N;
    TRay tray;
    RayJob rj;

    std::string host ="rabbitmq";
    std::string occlusionqueue = "occlusionqueue";
    std::string rayqueue = "rayqueue";

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    cout << "shade worker: " << tid << " started" << endl;
    while(true)
    {
      if (shadeworkqueue.try_pop(rj))
      {
        ps = rj.ps;
        P = rj.P;
        N = rj.N;
        tray = rj.tray;
        
        //do shading

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
        writer.Double(tray.o.x);writer.Double(tray.o.y);writer.Double(tray.o.z);
        writer.EndArray();
        writer.Key("d");
        writer.StartArray();
        writer.Double(tray.d.x);writer.Double(tray.d.y);writer.Double(tray.d.z);
        writer.EndArray();
        writer.EndObject();

        writer.Key("Rad");
        writer.StartArray();
        writer.Double(1.0);writer.Double(1.0);writer.Double(1.0);
        writer.EndArray();
        writer.EndObject();

        ex->Publish(s.GetString(),occlusionqueue);

      }
    }
}

int onCancel(AMQPMessage * message ) {
	cout << "cancel tag="<< message->getDeliveryTag() << endl;
	return 0;
}

int  rayMessageHandler( AMQPMessage * message  ) 
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
    Vec3f t;
    t.x = document["ps"]["t"][0].GetFloat();t.y = document["ps"]["t"][1].GetFloat();t.z = document["ps"]["t"][2].GetFloat();
    ps.t = t;

    //Need to clean this whole section up so we don't have multiple copies of ray data
    TRay tray;
    tray.pdf = document["ray"]["pdf"].GetFloat();
    tray.depth = document["ray"]["depth"].GetFloat();
    Vec3f o;
    o.x = document["ray"]["o"][0].GetFloat();o.y = document["ray"]["o"][1].GetFloat();o.z = document["ray"]["o"][2].GetFloat();
    tray.o = o;
    Vec3f d;
    d.x = document["ray"]["d"][0].GetFloat();d.y = document["ray"]["d"][1].GetFloat();d.z = document["ray"]["d"][2].GetFloat();
    tray.d = d;

    Vec3f N;
    N.x = document["N"][0].GetFloat();N.y = document["N"][1].GetFloat();N.z = document["N"][2].GetFloat();
    Vec3f P;
    P.x = document["P"][0].GetFloat();P.y = document["P"][1].GetFloat();P.z = document["P"][2].GetFloat();
    
    
    Ray ray(Vec3fa(tray.o),Vec3fa(tray.d),0.0,inf);

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