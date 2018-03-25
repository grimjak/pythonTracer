import numpy as np
import sys
import matplotlib.pyplot as plt
import io
from skimage.io import imsave
from threading import Thread
import queue

from math import sqrt, cos, sin, floor
import random
import time
import json
#from PIL import Image

from flask import Flask
from flask import send_file
from flask_socketio import SocketIO, emit

import pika

import ghalton

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'test2 app'

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


stopped = False
quit = False

class Ray():
    def __init__(self, O=np.zeros(3), D=np.zeros(3), PDF=1, depth=0):
        self.o = O
        self.d = D
        self.pdf = 1.0
        self.depth = 0
        return
    def to_dict(self):
        return {'o':self.o.tolist(),'d':self.d.tolist(),'pdf':self.pdf,'depth':self.depth}
    def from_dict(self,dict):
        self.o = np.array(dict['o'])
        self.d = np.array(dict['d'])
        self.pdf = dict['pdf']
        self.depth = dict['depth']


class Sampler():
    def __init__(self,size):
        self.size = size
        self.index = random.randrange(1,size)
        self.sequencer = ghalton.Halton(2) #2D set of LDN
        self.points = self.sequencer.get(size)
    def sample2d(self,offset=0):
        self.index += 1
        return self.points[(self.index+offset)%self.size]

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

class QueueWrapper():
    def __init__(self,key):
        self.key = key
        channel.queue_declare(queue=key)

    def put(self,data):
        channel.basic_publish(exchange='',routing_key=self.key,body=json.dumps(data))


def normalize(x):
    x /= np.linalg.norm(x)
    return x

@app.route('/exit')
def exit():
    quit = True
    return('quitting')

def setup_writer():
    while True:
        count = 0        
        for method_frame, properties, body in channel.consume('radiancequeue'):
            count += 1
            # Acknowledge the message
            channel.basic_ack(method_frame.delivery_tag)

            radiancecontext = json.loads(body)
            ps = PixelSample()
            ps.from_dict(radiancecontext['ps'])
            rgb = np.array(radiancecontext['rgb'])
            weights[h-ps.j-1,ps.i] = ps.w
            img[h-ps.j-1,ps.i] += rgb #need to add this to another queue that can then write to disk?
            if count > 1000:
                break
        print("received 100 radiance updates, writing")
        #imsave('tmp.png', (img/weights).clip(0,1), plugin='pil', format_str='png')
        imsave('tmp.png', (img/weights).clip(0,1))


    # Cancel the consumer and return any pending messages
    requeued_messages = channel.cancel()

depth_max = 4  # Maximum number of light reflections.
samples = 4
filterwidth = 2.0
smp = Sampler(w*h*samples)

O = np.array([0., 0.35, -1.])  # Camera.

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

img = np.zeros((h, w, 3))
weights = np.zeros((h,w,1))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

camerarayqueue =  QueueWrapper('camerarayqueue')
rayqueue = QueueWrapper('rayqueue')
shadequeue = QueueWrapper('shadequeue')
occlusionqueue = QueueWrapper('occlusionqueue')

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    setup_writer()
