import numpy as np
import sys
#import matplotlib.pyplot as plt
#import io
from skimage.io import imsave
#from threading import Thread
#import queue

from math import sqrt, cos, sin, floor
#import random
import time
#import json
#from PIL import Image

from flask import Flask
from flask import send_file
from flask_socketio import SocketIO, emit

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

w = 640
h = 480
M_PI = 3.14159265358979323846
M_INVPI = 1/ M_PI

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
        fields = ['samplesWritten', 'percentComplete', 'timePerPixel']

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

        for method_frame, properties, body in channel.consume('radiancequeue'):
            #start = timeit.timeit() 
            count += 1
            # Acknowledge the message
            channel.basic_ack(method_frame.delivery_tag)
            unpacker.feed(body)

            ps = unpacker.unpack()
            rad = np.array(unpacker.unpack())
            depth = unpacker.unpack()
   
            if (depth == 1): 
                weights[h-ps[3]-1,ps[2]] = ps[1] #doesn't work in shadowed areas unless we write 0
                totalCount+=1
                pixelCount+=1
            img[h-ps[3]-1,ps[2]] += rad
            if rad.max() < (mins[h-ps[3]-1,ps[2]]).max() : mins[h-ps[3]-1,ps[2]] = rad
            if rad.max() > (maxs[h-ps[3]-1,ps[2]]).max() : maxs[h-ps[3]-1,ps[2]] = rad

            if time.perf_counter()-start > 1 and pixelCount > 0: #make sure we don't miss the last few pixels, change to time based?
                break
        
        MySeriesHelper(server_name='renderwriter', samplesWritten=count, percentComplete=totalCount / (w*h*samples), timePerPixel=(time.perf_counter()-start)/pixelCount)
        MySeriesHelper.commit()

        print("received 10000 radiance updates, writing")
        #imsave('renders/tmp.png', (img/weights).clip(0,1), plugin='pil', format_str='png')
        imsave('static/tmp.png', (img/weights.clip(1,samples)).clip(0,1))

        #var = maxs-mins
        #imsave('var.png', var.clip(0,1))



    # Cancel the consumer and return any pending messages
    requeued_messages = channel.cancel()

samples = 4
filterwidth = 2.0


img = np.zeros((h, w, 3))
weights = np.zeros((h,w,1))
mins = np.full((h, w, 3),[1e6,1e6,1e6])
maxs = np.zeros((h, w, 3))

var = np.zeros((h,w,1))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    setup_writer()
