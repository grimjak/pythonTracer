#include <stdlib.h>
#include <stdio.h>
#include "AMQPcpp.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <tuple>

#include <embree3/rtcore.h> 
#include <tutorials/common/math/math.h>
#include <tutorials/common/math/vec.h>
#include <tutorials/common/core/ray.h>

#include <msgpack.hpp>

#include <tbb/concurrent_queue.h>

#include "obj_loader.h"
#define EMBREE
#include "messaging.h"

//#include <influxdb_raw_db_utf8.h>
//#include <influxdb_simple_api.h>
#include <influxdb.hpp>

#include <netdb.h>


//using namespace AmqpClient;
using namespace std;
using namespace embree;
using namespace tbb;
 
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

int numRays = 0;
float totalRayTime = 0;
int numPacketsIn = 0;
int numPacketsOut = 0;
float totalPacketsInTime = 0;
float totalPacketsOutTime = 0;
int rayHist[11];
unsigned int rayDepth;

/* scene data */
RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;
std::vector<std::vector<unsigned int>> materialids;


const int numPhi = 64;
const int numTheta = 2*numPhi;


/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

//struct Vec3f {float x,y,z; };


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
  Vec3f rad;
  OcclusionJob(){};
  OcclusionJob(Ray r,PixelSample p,TRay t,Vec3f rd):ray(r),ps(p),tray(t),rad(rd)
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
      //  triangles[tri].materialid = 0;
        triangles[tri].v0 = p10;
        triangles[tri].v1 = p01;
        triangles[tri].v2 = p00;
        tri++;
      }

      if (phi < numPhi) {
      //  triangles[tri].materialid = 0;
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
unsigned int addGroundPlane (RTCBuildQuality quality, RTCScene scene_i)
{
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry geom = rtcNewGeometry (g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* set vertices */
  Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),4);
  vertices[0].x = -10; vertices[0].y = -0.5; vertices[0].z = -10;
  vertices[1].x = -10; vertices[1].y = -0.5; vertices[1].z = +10;
  vertices[2].x = +10; vertices[2].y = -0.5; vertices[2].z = -10;
  vertices[3].x = +10; vertices[3].y = -0.5; vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),2);
  
  //triangles[0].materialid = 0; triangles[0].v0 = 0; triangles[0].v1 = 1; triangles[0].v2 = 2;
  //triangles[1].materialid = 0; triangles[1].v0 = 1; triangles[1].v1 = 3; triangles[1].v2 = 2;

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
    std::string shadequeue = "shadequeue";

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    AMQPQueue * queue = amqp.createQueue(shadequeue);
		queue->Declare();
		queue->Bind( "ptex", shadequeue);

    cout << "ray worker: " << tid << " started" << endl;
    msgpack::sbuffer ss;
    msgpack::packer<msgpack::sbuffer> pk(&ss);

    hostent * record = gethostbyname("influxdb");
    in_addr * address = (in_addr * )record->h_addr;
	  string ip_address = inet_ntoa(* address);
    influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");
    Timer tmr;
    float thisRayTime = 0;
    float thisPacketTime = 0;
    while(true)
    {
      if (rayworkqueue.try_pop(rj))
      {
        numRays++;
        tmr.reset();
        ray = rj.ray;
        ps = rj.ps;
        tray = rj.tray;
        rayHist[tray.depth]++;

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
            unsigned int materialid = materialids[ray.geomID][ray.primID];
            Vec3f Cs(1,1,1);

            thisRayTime = tmr.elapsed();
            totalRayTime += thisRayTime;

            tmr.reset();
            numPacketsOut++;

            pk.pack(ps);
            pk.pack(tray);
            pk.pack(P);
            pk.pack(N);
            pk.pack(Cs);
            pk.pack(materialid);

            ex->Publish(ss.data(),ss.size(),shadequeue);
            ss.clear();
            thisPacketTime = tmr.elapsed();
            totalPacketsOutTime += thisPacketTime;
        } else {
            thisRayTime = tmr.elapsed();
            totalRayTime += thisRayTime;
        }

        if ((numRays%1000 == 0) && (numRays > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("rayserver")
          .tag("name", "totalRays")
          .field("numrays", numRays)
          .field("totalRayTime", totalRayTime)
          .field("raysPerSecond", 1.0 / thisRayTime)

          .meas("raydepth")
          .tag("depth", "0")
          .field("count",rayHist[0])

          .meas("raydepth")
          .tag("depth", "1")
          .field("count",rayHist[1])

          .meas("raydepth")
          .tag("depth", "2")
          .field("count",rayHist[2])

          .meas("raydepth")
          .tag("depth", "3")
          .field("count",rayHist[3])

          .meas("raydepth")
          .tag("depth", "4")
          .field("count",rayHist[4])

          .meas("raydepth")
          .tag("depth", "5")
          .field("count",rayHist[5])

          .meas("raydepth")
          .tag("depth", "6")
          .field("count",rayHist[6])

          .meas("raydepth")
          .tag("depth", "7")
          .field("count",rayHist[7])

          .meas("raydepth")
          .tag("depth", "8")
          .field("count",rayHist[8])

          .meas("raydepth")
          .tag("depth", "9")
          .field("count",rayHist[9])

          .meas("raydepth")
          .tag("depth", "10")
          .field("count",rayHist[10])                    
          .post_http(si);
        }
        if ((numPacketsOut%1000 == 0) && (numPacketsOut > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("rayserver")
          .tag("name", "totalPacketsOut")
          .field("num", numPacketsOut)
          .field("totalTime", totalPacketsOutTime)
          .field("packetsPerSecond", 1.0 / thisPacketTime)
          .post_http(si);
        }
      }
    }
}

void occlusionworker(int tid)
{
    Ray ray;
    PixelSample ps;
    TRay tray;
    Vec3f rad;
    OcclusionJob rj;

    std::string host ="rabbitmq";
    std::string radiancequeue = "radiancequeue";

    AMQP amqp(host);
    AMQPExchange *ex = amqp.createExchange("ptex");
    ex->Declare("ptex","direct");

    AMQPQueue * queue = amqp.createQueue(radiancequeue);
		queue->Declare();
		queue->Bind( "ptex", radiancequeue);

    cout << "occlusion worker: " << tid << " started" << endl;
    hostent * record = gethostbyname("influxdb");
    in_addr * address = (in_addr * )record->h_addr;
	  string ip_address = inet_ntoa(* address);
    influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");

    Timer tmr;

    msgpack::sbuffer ss;
    msgpack::packer<msgpack::sbuffer> pk(&ss);
    float thisRayTime = 0;
    float thisPacketTime = 0;
    while(true)
    {
      if (occlusionworkqueue.try_pop(rj))
      {
        numRays++;
        tmr.reset();
        ray = rj.ray;
        ps = rj.ps;
        tray = rj.tray;
        rad = rj.rad;
   
        /*intersect ray with scene*/
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcOccluded1(g_scene,&context,RTCRay_(ray));
        //rtcIntersect1(g_scene,&context,RTCRayHit_(ray));
        thisRayTime = tmr.elapsed();
        totalRayTime += thisRayTime;
        
        if (ray.tfar >= 0.0001f)
        //if (ray.geomID == RTC_INVALID_GEOMETRY_ID)
        {
          numPacketsOut++;
          tmr.reset();

          pk.pack(ps);
          pk.pack(rad);
          pk.pack(tray.depth);

          ex->Publish(ss.data(),ss.size(),radiancequeue);
          ss.clear();
          thisPacketTime = tmr.elapsed();
          totalPacketsOutTime += thisPacketTime;
        } else {
          numPacketsOut++;
          tmr.reset();
          
          pk.pack(ps);
          pk.pack(Vec3f(0,0,0));
          pk.pack(tray.depth);

          ex->Publish(ss.data(),ss.size(),radiancequeue);
          ss.clear();
          thisPacketTime = tmr.elapsed();
          totalPacketsOutTime += thisPacketTime;
        }
      }
        if ((numRays%1000 == 0) && (numRays > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("occlusionserver")
          .tag("name", "totalRays")
          .field("numrays", numRays)
          .field("totalRayTime", totalRayTime)
          .field("raysPerSecond", 1.0 / thisRayTime)
          .post_http(si);
        }
        if ((numPacketsOut%1000 == 0) && (numPacketsOut > 0))
        {
        int success = influxdb_cpp::builder()
          .meas("occlusionserver")
          .tag("name", "totalPacketsOut")
          .field("num", numPacketsOut)
          .field("totalTime", totalPacketsOutTime)
          .field("packetsPerSecond", 1.0 / thisPacketTime)
          .post_http(si);
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
    id = addGroundPlane(quality,g_scene);

    rtcCommitScene(g_scene);
}

void setup_obj_scene()
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, "/usr/src/app/cornell_box.obj",
                              "/usr/src/app/", true);
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    printf("Failed to load/parse .obj.\n");
    return;
  }

  std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
  std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
  std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2) << std::endl;
  std::cout << "# of materials : " << (materials.size()) << std::endl;


  materials.push_back(tinyobj::material_t());

  //create device
  g_device = rtcNewDevice("");
  g_scene = rtcNewScene(g_device);
  rtcSetSceneFlags(g_scene,RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
  rtcSetSceneBuildQuality(g_scene,RTC_BUILD_QUALITY_LOW);
  //iterate over shapes
  for (size_t s = 0; s < shapes.size(); s++) { 
    std::vector<unsigned int> mat_ids;
    //create geo
    RTCGeometry geom = rtcNewGeometry(g_device,RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetGeometryBuildQuality(geom,RTC_BUILD_QUALITY_LOW);
    
    Vertex*   vertices  = (Vertex*  ) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),attrib.vertices.size());
    Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),shapes[s].mesh.indices.size()/3);
    //int* materialids = (int*) rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT,sizeof(int),shapes[s].mesh.indices.size()/3);

    for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) 
    {
      tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
      tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
      tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

      int current_material_id = shapes[s].mesh.material_ids[f];

      if ((current_material_id < 0) ||
          (current_material_id >= static_cast<int>(materials.size()))) 
      {
        // Invaid material ID. Use default material.
        current_material_id = materials.size() - 1;  // Default material is added to the last item in `materials`.
      }
      //cerr<<"mat id:"<<current_material_id<<endl;
      mat_ids.push_back(current_material_id);
      triangles[f].v0 = idx0.vertex_index;
      triangles[f].v1 = idx1.vertex_index;
      triangles[f].v2 = idx2.vertex_index;      
    }
    materialids.push_back(mat_ids);

    for (size_t v = 0; v < attrib.vertices.size()/3;v++)
    {
      vertices[v].x = attrib.vertices[v*3];
      vertices[v].y = attrib.vertices[v*3+1];
      vertices[v].z = attrib.vertices[v*3+2];
    }
/*
    for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) 
    {
      
      cerr<<vertices[triangles[f].v0].x<<","<<vertices[triangles[f].v0].y<<","<<vertices[triangles[f].v0].z<<endl;
      cerr<<vertices[triangles[f].v1].x<<","<<vertices[triangles[f].v1].y<<","<<vertices[triangles[f].v1].z<<endl;
      cerr<<vertices[triangles[f].v2].x<<","<<vertices[triangles[f].v2].y<<","<<vertices[triangles[f].v2].z<<endl;
    }
*/
    rtcCommitGeometry(geom);
    unsigned int geomID = rtcAttachGeometry(g_scene,geom);
    rtcReleaseGeometry(geom);
  } 
  rtcCommitScene(g_scene);                          
}


int onCancel(AMQPMessage * message ) {
	cout << "cancel tag="<< message->getDeliveryTag() << endl;
	return 0;
}

int rayMessageHandler( AMQPMessage * message  ) 
{
  uint32_t j = 0;
  Timer tmr;
  numPacketsIn++;
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

    pac.next(oh);
    oh.get().convert(ps);
    pac.next(oh);
    oh.get().convert(tray);

    Ray ray(Vec3f(tray.o),Vec3f(tray.d),0.0,inf);

    rayworkqueue.push( RayJob(ray,ps,tray));

  }
  totalPacketsInTime += tmr.elapsed();

  if (numPacketsIn%10000 == 0)
  {
    std::cout<<"total packets In: "<<numPacketsIn<<endl;
    std::cout<<"total time: "<<totalPacketsInTime << endl;
    std::cout<<"packets in / sec: "<<numPacketsIn / totalPacketsInTime << endl;
  }
  return 0;
}

int  occlusionMessageHandler( AMQPMessage * message  ) 
{
  uint32_t j = 0;
  Timer tmr;
  numPacketsIn++;
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
    float len;
    Vec3f rad;

    pac.next(oh);
    oh.get().convert(ps);
    pac.next(oh);
    oh.get().convert(tray);
    pac.next(oh);
    oh.get().convert(len);
    pac.next(oh);
    oh.get().convert(rad);    

    Ray ray(Vec3f(tray.o),Vec3f(tray.d),0.0001,len);

    occlusionworkqueue.push( OcclusionJob(ray,ps,tray,rad));
  }
  totalPacketsInTime += tmr.elapsed();
  if (numPacketsIn%10000 == 0)
  {
    std::cout<<"total packets in: "<<numPacketsIn<<endl;
    std::cout<<"total time: "<<totalPacketsInTime << endl;
    std::cout<<"packets in / sec: "<<numPacketsIn / totalPacketsInTime << endl;
  }
  return 0;
}

int main(int argc, char *argv[])
{
  cout << "Subscriber started" << endl;

  std::string host ="rabbitmq";
  std::string queuename = "rayqueue";
  bool occlusion = false;
  if(strcmp(argv[1], "occlusion") == 0)
  {
    queuename = "occlusionqueue";
    occlusion = true;
    cout << "Occlusion server started" << endl;
  } else { 
    cout << "Ray server started" << endl;
  }

  //added delay for rabbit mq start to avoid failing with socket error
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
  //setup_scene();
  setup_obj_scene();
  //thread for tracing rays
  const int numthreads = 1;
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
		queue->Bind( "ptex", queuename);

		queue->setConsumerTag("");
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