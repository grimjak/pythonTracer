#include <stdlib.h>
#include <stdio.h>
#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include "AMQPcpp.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <tuple>

//#include <boost/uuid/uuid_io.hpp>
//#include <boost/uuid.hpp>
//#include <boost/uuid/uuid_generators.hpp>
#include <embree3/rtcore.h> 
#include<tutorials/common/math/math.h>
#include<tutorials/common/math/vec.h>
#include<tutorials/common/core/ray.h>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include <tbb/concurrent_queue.h>

using namespace AmqpClient;
using namespace std;
using namespace rapidjson;
using namespace embree;
using namespace tbb;
 
/* scene data */
RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;

const int numPhi = 120;
const int numTheta = 2*numPhi;


/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

//struct Vec3f {float x,y,z; };
typedef struct PixelSample { int o, w, i, j; Vec3fa t;} PixelSample;
typedef struct TRay {float pdf; int depth; Vec3fa o, d;} TRay;

typedef struct RayJob {
  Ray ray; 
  PixelSample ps; 
  TRay tray;
  RayJob(){};
  RayJob(Ray r,PixelSample p,TRay t):ray(r),ps(p),tray(t)
  {};
} RayJob;

typedef struct OcclusionJob {
  Ray ray; 
  PixelSample ps; 
  TRay tray;
  Vec3fa rad;
  OcclusionJob(){};
  OcclusionJob(Ray r,PixelSample p,TRay t,Vec3fa rd):ray(r),ps(p),tray(t),rad(rd)
  {};
} OcclusionJob;

concurrent_queue<RayJob> rayworkqueue;
concurrent_queue<OcclusionJob> occlusionworkqueue;

/* adds a sphere to the scene */
unsigned int createSphere (RTCBuildQuality quality, std::vector<float> pos, const float r)
{
  /* create a triangulated sphere */
  RTCGeometry geom = rtcNewGeometry (g_device, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geom, quality);

  /* map triangle and vertex buffer */
  Vertex*   vertices  = (Vertex*  ) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),numTheta*(numPhi+1));
  Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),2*numTheta*(numPhi-1));

  /* create sphere geometry */
  int tri = 0;
  //const float rcpNumTheta = rcp((float)numTheta);
  //const float rcpNumPhi   = rcp((float)numPhi);
  const float rcpNumTheta = 1.0f/(float)numTheta;
  const float rcpNumPhi   = 1.0f/(float)numPhi;  
  for (int phi=0; phi<=numPhi; phi++)
  {
    for (int theta=0; theta<numTheta; theta++)
    {
      const float phif   = phi*float(M_PI)*rcpNumPhi;
      const float thetaf = theta*2.0f*float(M_PI)*rcpNumTheta;
      Vertex& v = vertices[phi*numTheta+theta];
      v.x = pos[0] + r*embree::sin(phif)*embree::sin(thetaf);
      v.y = pos[1] + r*embree::cos(phif);
      v.z = pos[2] + r*embree::sin(phif)*embree::cos(thetaf);
    }
    if (phi == 0) continue;

    for (int theta=1; theta<=numTheta; theta++)
    {
      int p00 = (phi-1)*numTheta+theta-1;
      int p01 = (phi-1)*numTheta+theta%numTheta;
      int p10 = phi*numTheta+theta-1;
      int p11 = phi*numTheta+theta%numTheta;

      if (phi > 1) {
        triangles[tri].v0 = p10;
        triangles[tri].v1 = p01;
        triangles[tri].v2 = p00;
        tri++;
      }

      if (phi < numPhi) {
        triangles[tri].v0 = p11;
        triangles[tri].v1 = p01;
        triangles[tri].v2 = p10;
        tri++;
      }
    }
  }

  rtcCommitGeometry(geom);
  unsigned int geomID = rtcAttachGeometry(g_scene,geom);
  rtcReleaseGeometry(geom);
  return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane (RTCScene scene_i)
{
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry geom = rtcNewGeometry (g_device, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geom, quality);

  /* set vertices */
  Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),4);
  vertices[0].x = -10; vertices[0].y = -0.5; vertices[0].z = -10;
  vertices[1].x = -10; vertices[1].y = -0.5; vertices[1].z = +10;
  vertices[2].x = +10; vertices[2].y = -0.5; vertices[2].z = -10;
  vertices[3].x = +10; vertices[3].y = -0.5; vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),2);
  triangles[0].v0 = 0; triangles[0].v1 = 1; triangles[0].v2 = 2;
  triangles[1].v0 = 1; triangles[1].v1 = 3; triangles[1].v2 = 2;

  rtcCommitGeometry(geom);
  unsigned int geomID = rtcAttachGeometry(scene_i,geom);
  rtcReleaseGeometry(geom);
  return geomID;
}

void rayworker(int tid)
{
    Ray ray;
    PixelSample ps;
    TRay tray;
    RayJob rj;

    std::string host ="rabbitmq";

    Channel::ptr_t channel;

    channel = Channel::Create(host);//host rabbit shoould work here from compose file, but had to define i
    std::string shadequeue = "shadequeue";
    channel->DeclareExchange(shadequeue,"fanout",false,false,false);

    cout << "ray worker: " << tid << " started" << endl;
    while(true)
    {
      if (rayworkqueue.try_pop(rj))
      {
        ray = rj.ray;
        ps = rj.ps;
        tray = rj.tray;
        /*intersect ray with scene*/
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcIntersect1(g_scene,&context,RTCRayHit_(ray));

        //if we've got an intersection add to shade queue
        //ps, P, N, ray, Cs
        if (ray.geomID != RTC_INVALID_GEOMETRY_ID)
        {
            Vec3f P = ray.org + ray.tfar*ray.dir;
            Vec3f N = normalize(ray.Ng);
           // cout << P << endl;
           // cout << ray.org <<endl;

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

            writer.Key("P");
            writer.StartArray();
            writer.Double(P.x);writer.Double(P.y);writer.Double(P.z);
            writer.EndArray();

            writer.Key("N");
            writer.StartArray();
            writer.Double(N.x);writer.Double(N.y);writer.Double(N.z);
            writer.EndArray();

            writer.Key("Cs");
            writer.StartArray();
            writer.Double(1.0);writer.Double(1.0);writer.Double(1.0);
            writer.EndArray();
            writer.EndObject();
  
            BasicMessage::ptr_t outgoing_message = BasicMessage::Create();
            outgoing_message->Body(s.GetString());
            channel->BasicPublish("",shadequeue,outgoing_message);

           //    cout << s.GetString() << endl;
        }
      }
    }
}

void occlusionworker(int tid)
{
    Ray ray;
    PixelSample ps;
    TRay tray;
    Vec3fa rad;
    OcclusionJob rj;

    std::string host ="rabbitmq";

    Channel::ptr_t channel;

    channel = Channel::Create(host);//host rabbit shoould work here from compose file, but had to define i
    std::string radiancequeue = "radiancequeue";
    channel->DeclareExchange(radiancequeue,"fanout",false,false,false);

    cout << "occlusion worker: " << tid << " started" << endl;
    while(true)
    {
      if (occlusionworkqueue.try_pop(rj))
      {
        ray = rj.ray;
        ps = rj.ps;
        tray = rj.tray;
        rad = rj.rad;
        /*intersect ray with scene*/
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcOccluded1(g_scene,&context,RTCRay_(ray));

        if (ray.tfar >= 0.0f)
        {
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

            writer.Key("rgb");
            writer.StartArray();
            writer.Double(rad.x);writer.Double(rad.y);writer.Double(rad.z);
            writer.EndArray();
            writer.EndObject();
  
            BasicMessage::ptr_t outgoing_message = BasicMessage::Create();
            outgoing_message->Body(s.GetString());
            channel->BasicPublish("",radiancequeue,outgoing_message);
        }
      }
    }
}

void setup_scene()
{
    //create device
    g_device = rtcNewDevice("");
    g_scene = rtcNewScene(g_device);
    rtcSetSceneFlags(g_scene,RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(g_scene,RTC_BUILD_QUALITY_LOW);

    RTCBuildQuality quality = RTC_BUILD_QUALITY_LOW;

    int id = createSphere(quality,{.75, .1, 1.}, .6);
    id = createSphere(quality,{-.75, .1, 2.25}, .6);
    id = createSphere(quality,{-2.75, .1, 3.5}, .6);
    id = addGroundPlane(g_scene);

    rtcCommitScene(g_scene);
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

    Ray ray(Vec3fa(tray.o),Vec3fa(tray.d),0.0,inf);

    rayworkqueue.push( RayJob(ray,ps,tray));
  }
  return 0;
}

int  occlusionMessageHandler( AMQPMessage * message  ) 
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

    Vec3fa rad;
    rad.x = document["rad"][0].GetFloat(); rad.y = document["rad"][1].GetFloat(); rad.z = document["rad"][2].GetFloat();

    Ray ray(Vec3fa(tray.o),Vec3fa(tray.d),0.001,inf);

    occlusionworkqueue.push( OcclusionJob(ray,ps,tray,rad));
  }
  return 0;
}

int main(int argc, char *argv[])
{
    cout << "Subscriber started" << endl;

  std::string host ="rabbitmq";
  std::string queuename = "rayqueue";
  bool occlusion = false;
  if(std::string(argv[1]) == "occlusion")
  {
    queuename = "occlusionqueue";
    occlusion = true;
    cout << "Occlusion server started" << endl;
  } else {
    
    
    // cout << "Ray server started" << endl;
  }

  //added delay for rabbit mq start to avoid failing with socket error
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
  setup_scene();

  //thread for tracing rays
  const int numthreads = 2;
  std::thread rayworkthreads[numthreads];
  //launch threads
  if (occlusion)
  {
    for(int i=0;i<numthreads;++i) rayworkthreads[i] = std::thread(occlusionworker,i);
  } else
  {
    for(int i=0;i<numthreads;++i) rayworkthreads[i] = std::thread(rayworker,i);
  }
  
  try {
		AMQP amqp(host);
    //add occlusion queue 
		AMQPQueue * queue = amqp.createQueue(queuename);

		queue->Declare();
		//qu2->Bind( "", "");

		queue->setConsumerTag("tag_123");
    if(occlusion)	queue->addEvent(AMQP_MESSAGE, occlusionMessageHandler );
    else queue->addEvent(AMQP_MESSAGE, rayMessageHandler);
		queue->addEvent(AMQP_CANCEL, onCancel );

		queue->Consume(AMQP_NOACK);//

	} catch (AMQPException e) {
		std::cout << e.getMessage() << std::endl;
	}

    rtcReleaseScene (g_scene); g_scene = nullptr;
    rtcReleaseDevice(g_device); g_device = nullptr;
    cout << "OK"<<endl;    
}