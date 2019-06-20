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

#define FILTER_TABLE_SIZE 1024

using namespace std;
using namespace rapidjson;
using namespace tbb;

typedef float Float;
typedef Imath_2_2::Vec3<Float>     Vec3;
typedef Imath_2_2::Matrix33<Float> Matrix33;
typedef Imath_2_2::Matrix44<Float> Matrix44;
typedef Imath_2_2::Color3<Float>   Color3;
typedef Imath_2_2::Vec2<Float>     Vec2;

static Vec3 O = Vec3(274,274,-440);
//static Vec3 O = Vec3(0,0.35,-1);

static int w = 640;
static int h = 480;
static float r = (float)w/(float)h;
static float filterwidth = 6.0;
static int samples = 1;


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
  Vec3 Q = Vec3(O.x+x,O.y+y,O.z+1);
  Vec3 D = (Q-O).normalize();
  return TRay(O,D);
}

//generate one iterations worth of samples
//need to define w and h and filterwidth
// iterate over pixels, need to keep track of index
void iterate(int &index, int iteration,float *offsets, vector<float> *filter_table)
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
  float offsets[w*h];

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
    
  for(int s = 0; s < samples; s++)
  {
    iterate(index,s,&offsets[0],&filter_table);
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