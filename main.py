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
        channel.exchange_declare(exchange="ptex", exchange_type="direct")
        channel.queue_declare(queue=key)
        channel.queue_bind(exchange="ptex",queue=key,routing_key=key)

    def put(self,data):
        channel.basic_publish(exchange='ptex',routing_key=self.key,body=json.dumps(data))

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def sample(ps,ray):
 #   trace_camera_ray(ps,ray,smp)
    jsonmessage = {'ray':ray.to_dict(),'ps':ps.to_dict()}
    #camerarayqueue.put(jsonmessage)
    rayqueue.put(jsonmessage)


def render_scene():
    #bin rays based on direction (and origin?)
    #go through rays in ordered batches producing a list of hit points
    #shade hit points, secondary rays are fed back into ray bins

    col = np.zeros(3)  # Current color.
    Q = np.array([0., 0., 0.])  # Camera pointing to.

    # Loop through all pixels.
    for s in range(samples):   
        for i, x in enumerate(np.linspace(S[0], S[2], w)):
            if i % 10 == 0:
                print (i / float(w) * 100, "%")
            for j, y in enumerate(np.linspace(S[1], S[3], h)):
                offsetu = filterwidth * smp.sample2d()[0]
                offsetv = filterwidth * smp.sample2d()[1]
                Q[:2] = (x+(offsetu-(filterwidth/2.0))/w, y+(offsetv-(filterwidth/2.0))/h)

                D = normalize(Q - O)
                ray = Ray(O, D)

                ps = PixelSample(i,j,o=random.randint(0,256),w=s+1)
                #weights[h-j-1,i] += 1 
                sample(ps,ray)

                if stopped:
                    break

                if quit:
                    exit()

@app.route('/render')  
def startRender():
    print("render")

    print("start render")
  #  render_scene()
    if not t.is_alive():
        print("starting render")
        t.start()
    else:
        stopped = True
        t.join()
        t.start()

    return 'render started'

    #outstring = io.BytesIO()
    #img = np.clip(img,0,1)
    #imsave(outstring, img.clip(0,1), plugin='pil', format_str='png')
   ## outstring.seek(0)
  #  return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

cameraray_threads = 1
ray_threads = 1
shade_threads = 1
occlusion_threads = 1
depth_max = 4  # Maximum number of light reflections.
samples = 4
filterwidth = 2.0
smp = Sampler(w*h*samples)

t = Thread(target = render_scene)


O = np.array([0., 0.35, -1.])  # Camera.

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

camerarayqueue =  QueueWrapper('camerarayqueue')
rayqueue = QueueWrapper('rayqueue')
shadequeue = QueueWrapper('shadequeue')
occlusionqueue = QueueWrapper('occlusionqueue')

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    print (sys.argv)
    if sys.argv[-1] == "control":    
        #run the rest server an wait for commands
        app.run(debug=True,host='0.0.0.0')
