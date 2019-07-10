import numpy as np
import numpy.ma as ma
import sys
from skimage.io import imsave

from math import sqrt, cos, sin, floor
#import random
import time
from PIL import Image

import pika
import msgpack 
import timeit

from influxdb import InfluxDBClient
from influxdb import SeriesHelper

# InfluxDB connections settings
influxhost = 'influxdb'
port = 8086
user = 'influx'
password = 'influx'
dbname = 'db'

while True:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        if connection.is_open:
            print('OK')
            break
    except:
        print("no connection")
        time.sleep(1)

channel = connection.channel()

w = 512
h = 288
M_PI = 3.14159265358979323846
M_INVPI = 1/ M_PI
msgBatchSize = 144

influxclient = InfluxDBClient(influxhost, port, user, password, dbname)

class MySeriesHelper(SeriesHelper):
    """Instantiate SeriesHelper to write points to the backend."""

    class Meta:
        """Meta class stores time series helper configuration."""

        # The client should be an instance of InfluxDBClient.
        client = influxclient

        # The series name must be a string. Add dependent fields/tags
        # in curly brackets.
        series_name = 'events.stats.{server_name}'

        # Defines all the fields in this time series.
        fields = ['samplesWritten', 'percentComplete', 'pixelsPerSecond']

        # Defines all the tags for the series.
        tags = ['server_name']

        # Defines the number of data points to store prior to writing
        # on the wire.
        bulk_size = 5

        # autocommit must be set to True when using bulk_size
        autocommit = True

class PixelSample():
    def __init__(self,x=0,y=0,t=np.ones(3),o=0,w=1):
        self.i = x
        self.j = y
        self.t = t #througput
        self.o = o #offset for samples
        self.w = w
    def to_dict(self):
        return {'i':self.i,'j':self.j,'t':self.t.tolist(),'o':self.o,'w':self.w}
    def from_dict(self,dict):
        self.i = dict['i']
        self.j = dict['j']
        self.t = np.array(dict['t'])
        self.o = dict['o']
        self.w = dict['w']

#def decode_PixelSample(obj):
#    if b''

class QueueWrapper():
    def __init__(self,key):
        self.key = key
        channel.queue_declare(queue=key)

    def put(self,data):
        channel.basic_publish(exchange='ptex',routing_key=self.key,body=json.dumps(data))


def normalize(x):
    x /= np.linalg.norm(x)
    return x

#need some kind of pixel complete signal 
def setup_writer():
    unpacker = msgpack.Unpacker(raw=False)
    totalCount = 0
    while True:
        count = 0   
        pixelCount = 0
        start = time.perf_counter()
        frame = 1
        framecount = 0
        dupcount = 0

        for method_frame, properties, body in channel.consume('radiancequeue'):
            #start = timeit.timeit() 
            count += 1
            # Acknowledge the message
            channel.basic_ack(method_frame.delivery_tag)
            unpacker.feed(body)

            #is it better to keep hold of the entire framebuffer and average the samples whenever we write?
            for b in range(msgBatchSize):
                ps = unpacker.unpack()
                rad = np.array(unpacker.unpack())
                depth = unpacker.unpack()
                i = h-ps[3]-1
                j = ps[2]
                iteration = ps[1]-1
                if not np.isnan(imgbuffer[i,j,iteration,0]) :
                    print(i,j,iteration,imgbuffer[i,j,iteration],rad)
                    dupcount += 1
                    print ("shouldn't have duplication", str(dupcount))
                #imgbuffer[i,j].append(rad)
                imgbuffer[i,j,iteration] = rad
                #if iteration > imgbufferSamples[i,j]:
                #    imgbufferSamples[i,j] = iteration
                pixelCount += 1
                framecount += 1
                totalCount+=1

            if time.perf_counter()-start > 1 and pixelCount > 0: 
                MySeriesHelper(server_name='renderwriter', samplesWritten=totalCount, percentComplete=totalCount / (w*h*samples), pixelsPerSecond=pixelCount/(time.perf_counter()-start))
                MySeriesHelper.commit()
                
                print("120s elapsed, writing")
                #build output image from buffer                
                outimg = np.nanmean(imgbuffer,axis=2).clip(0,1)
                print("mean")
                outimg = np.power(outimg,1.0/2.2)
                #outimg = np.squeeze(np.stack((outimg,)*3, axis=-1)) #add back in the missing colour axes
                print("gamma and avg")

                outimg = (255.0*outimg).astype(np.uint8)
                print("types")

                Image.fromarray(outimg).save("static/tmp_512x384_{}.png".format(frame), mode='RGB')
                print("written")

                start = time.perf_counter()
                pixelCount = 0
                if framecount > w*h*((float)(frame*frame)*0.05) :
                    frame += 1
                    framecount = 0
        #var = maxs-mins
        #imsave('var.png', var.clip(0,1))
        print("out of message loop", str(count))
    # Cancel the consumer and return any pending messages
    requeued_messages = channel.cancel()

samples = 1024

img = np.zeros((h, w, 3))
#imgbuffer = ma.masked_all((h,w,samples,1))
#imgbuffer.soften_mask()
imgbuffer = np.zeros((h,w,samples,3))
#imgbuffer = np.empty((h,w,samples,1))
imgbuffer.fill(np.nan)

weights = np.ones((h,w,1))
mins = np.full((h, w, 3),[1e6,1e6,1e6])
maxs = np.zeros((h, w, 3))

var = np.zeros((h,w,1))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    setup_writer()
