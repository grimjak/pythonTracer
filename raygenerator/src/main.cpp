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
/*
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
*/
#include <msgpack.hpp>

#include <tbb/concurrent_queue.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathColor.h>


#include <OpenImageIO/fmath.h>

#define IMATH
#include "messaging.h"

#include <influxdb.hpp>
#include <netdb.h>
#include <unistd.h>
#include <limits.h>

#include <random>
#include "sobol.h"

#define FILTER_TABLE_SIZE 1024

using namespace std;
//using namespace rapidjson;
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
unsigned int msgBatchSize = 20;


static Vec3f O = Vec3f(274,274,-440);
static int w = 640;
static int h = 480;
static float r = (float)w/(float)h;
static float filterwidth = 6.0;
static int samples = 32;

AMQPExchange *ex;
AMQPQueue * queue;
char hostname[HOST_NAME_MAX];
std::string rayqueue = "rayqueue";

msgpack::sbuffer ss;
msgpack::packer<msgpack::sbuffer> pk(&ss);

hostent * record = gethostbyname("influxdb");
in_addr * address = (in_addr * )record->h_addr;
string ip_address = inet_ntoa(* address);
influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");

typedef struct ShadeJob {
  PixelSample ps; 
  Vec3f P;
  Vec3f N;
  TRay tray;
  ShadeJob(){};
  ShadeJob(PixelSample p,Vec3f pp,Vec3f nn,TRay t):ps(p),P(pp),N(nn),tray(t)
  {};
} ShadeJob;


concurrent_queue<ShadeJob> shadeworkqueue;

static float filter_func_gaussian(float v, float width)
{
  v *= 6.0f/width;
  return expf(-2.0f*v*v);
}

void util_cdf_evaluate(const int resolution,
                       const float from,
                       const float to,
                       const float width,
                       vector<float> &cdf)
{
	const int cdf_count = resolution + 1;
	const float range = to - from;
	cdf.resize(cdf_count);
	cdf[0] = 0.0f;
	/* Actual CDF evaluation. */
	for(int i = 0; i < resolution; ++i) {
		float x = from + range * (float)i / (resolution - 1);
		float y = filter_func_gaussian(x,width);
		cdf[i + 1] = cdf[i] + fabsf(y);
	}
	/* Normalize the CDF. */
	for(int i = 0; i <= resolution; i++) {
		cdf[i] /= cdf[resolution];
	}
}

/* Invert pre-calculated CDF function. */
void util_cdf_invert(const int resolution,
                     const float from,
                     const float to,
                     const vector<float> &cdf,
                     const bool make_symmetric,
                     vector<float> &inv_cdf)
{
  const float inv_resolution = 1.0f / (float)resolution;
  const float range = to - from;
  inv_cdf.resize(resolution);
  if(make_symmetric) {
    const int half_size = (resolution - 1) / 2;
    for(int i = 0; i <= half_size; i++) {
      float x = i / (float)half_size;
      int index = upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin();
      float t;
      if(index < cdf.size() - 1) {
        t = (x - cdf[index])/(cdf[index+1] - cdf[index]);
      } else {
        t = 0.0f;
        index = cdf.size() - 1;
      }
      float y = ((index + t) / (resolution - 1)) * (2.0f * range);
      inv_cdf[half_size+i] = 0.5f*(1.0f + y);
      inv_cdf[half_size-i] = 0.5f*(1.0f - y);
    }
  }
  else {
    for(int i = 0; i < resolution; i++) {
      float x = from + range * (float)i * inv_resolution;
      int index = upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin();
      float t;
      if(index < cdf.size() - 1) {
        t = (x - cdf[index])/(cdf[index+1] - cdf[index]);
      } else {
        t = 0.0f;
        index = resolution;
      }
      inv_cdf[i] = (index + t) * inv_resolution;
    }
  }
}

/* Evaluate inverted CDF of a given functor with given range and resolution. */
void util_cdf_inverted(const int resolution,
                       const float from,
                       const float to,
                       const bool make_symmetric,
                       const float width,
                       vector<float> &inv_cdf)
{
	vector<float> cdf;
	/* There is no much smartness going around lower resolution for the CDF table,
	 * this just to match the old code from pixel filter so it all stays exactly
	 * the same and no regression tests are failed.
	 */
	util_cdf_evaluate(resolution - 1, from, to, width, cdf);
	util_cdf_invert(resolution, from, to, cdf, make_symmetric, inv_cdf);
}


//generate a single sample for a given set of coordinates
//x and y in NDC, 0 to 1
//need to define O
TRay generateSample(float x, float y)
{
  Vec3f Q = Vec3f(O.x+x,O.y+y,O.z+1);
  Vec3f D = (Q-O).normalize();
  return TRay(O,D);
}

//generate one iterations worth of samples
//need to define w and h and filterwidth
// iterate over pixels, need to keep track of index
void iterate(int &index, int iteration,float *offsets, vector<float> *filter_table , int* indices)
{
  /*std::string host ="rabbitmq";
  std::string rayqueue = "rayqueue";

  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  AMQP amqp(host);
  AMQPExchange *ex = amqp.createExchange("ptex");
  ex->Declare("ptex","direct");

  AMQPQueue * queue = amqp.createQueue(rayqueue);
  queue->Declare();
	queue->Bind( "ptex", rayqueue);

  //need an offset per pixel
  msgpack::sbuffer ss;
  msgpack::packer<msgpack::sbuffer> pk(&ss);

  hostent * record = gethostbyname("influxdb");
  in_addr * address = (in_addr * )record->h_addr;
	string ip_address = inet_ntoa(* address);
  influxdb_cpp::server_info si(ip_address, 8086, "db", "influx", "influx");*/
  Timer tmr;
  float thisRayTime = 0;
  float thisPacketTime = 0;
  unsigned int batch = 0;

 // for (int i = 0; i<w; i++)
 // {
 //   for (int j = 0; j<h; j++)
 //   {
  for (int idx = 0; idx<w*h; idx++)
  {
      int j = indices[idx] / w;
      int i = indices[idx] - (j*w);
      numRays++;
      tmr.reset();
      float rx = sobol::sample(++index,1);
      float ry = sobol::sample(index,2);
      float offset = offsets[j*w+i];
      rx = (rx+offset);if (rx > 1) rx-=1;
      ry = (ry+offset);if (ry > 1) ry-=1;

      rx = (*filter_table)[int(rx*FILTER_TABLE_SIZE-1)];
      ry = (*filter_table)[int(ry*FILTER_TABLE_SIZE-1)];

      float x = 2*((float)i / (float)w - 0.5);
      float y =2*((float)j / (float)h - 0.5); 
      y/=r;

      x += rx/w;
      y += ry/w;
      TRay r = generateSample(x,y);
      int o = rand()%(256);

      PixelSample ps = PixelSample(o,iteration+1,i,j,Vec3f(1,1,1));
      thisRayTime = tmr.elapsed();
      totalRayTime+=thisRayTime;
      
      tmr.reset();
      pk.pack(ps);
      pk.pack(r);

      batch++;
      if(batch==msgBatchSize)
      {
        ex->Publish(ss.data(),ss.size(),rayqueue);
        ss.clear();
        batch=0;
      }

      numPacketsOut++;
      thisPacketTime = tmr.elapsed();
      totalPacketsOutTime+= thisPacketTime;
    
      if ((numRays%1000 == 0) && (numRays > 0))
      {
        int success = influxdb_cpp::builder()
          .meas("raygenerator")
          .tag("name", "totalRays")
          .tag("hostname", hostname)
          .field("numrays", numRays)
          .field("totalRayTime", totalRayTime)
          .field("raysPerSecond", 1.0 / thisRayTime)
          .field("percentComplete", (float)numRays / (w*h*samples))
          .post_http(si);
      }
      if ((numPacketsOut%1000 == 0) && (numPacketsOut > 0))
      {
        int success = influxdb_cpp::builder()
          .meas("rayserver")
          .tag("name", "totalPacketsOut")
          .tag("hostname", hostname)
          .field("num", numPacketsOut)
          .field("totalTime", totalPacketsOutTime)
          .field("packetsPerSecond", 1.0 / thisPacketTime)
          .post_http(si);
      }
    }
}

int onCancel(AMQPMessage * message ) {
	cout << "cancel tag="<< message->getDeliveryTag() << endl;
	return 0;
}

int  raygenMessageHandler( AMQPMessage * message  ) 
{
/*  uint32_t j = 0;
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
    
    

    shadeworkqueue.push( ShadeJob(ps,P,N,tray));
  }*/
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
  float offsets[w*h];
  int indices[w*h];

  int width = 6;
  vector<float> filter_table(FILTER_TABLE_SIZE);
  util_cdf_inverted(FILTER_TABLE_SIZE,
                    0.0f,
                    width * 0.5f,
                    true,
                    width,
                    filter_table);

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0,1);

  for(int i = 0; i < w*h; i++)
    offsets[i] = dist(e2);

  for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
      indices[j*w+i] = j*w+i;  
    
  random_shuffle(&indices[0],&indices[(h*w)-1]);


  gethostname(hostname, HOST_NAME_MAX);

  AMQP amqp(host);
  ex = amqp.createExchange("ptex"); //global
  ex->Declare("ptex","direct");

  queue = amqp.createQueue(rayqueue);
  queue->Declare();
	queue->Bind( "ptex", rayqueue);



  for(int s = 0; s < samples; s++)
  {
    cerr << "sample: " << s << endl;
    iterate(index,s,&offsets[0],&filter_table, &indices[0]);
  }
  cerr << "index = " << index << endl;

  //Wait for a command message which sets the camera and which pixels to generate samples for
  //Need to tag pixel samples so they can be routed to the correct writer

/*
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
	}*/
  cout << "OK"<<endl;    
}