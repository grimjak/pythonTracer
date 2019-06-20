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

def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = sqrt(1.0 - u1)
    sin_theta = sqrt(u1)
    phi = 2.0 * M_PI * u2
    return np.array([cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta])

def environment(dir):
    return np.array([1.0,1.0,1.0])

def shade_diffuse(ps,P,N,ray,Ci):
    # Diffuse 
    w = N 
    u = np.array([0.00319,1.0,0.0078])
    u = np.cross(u,w)
    u = normalize(u)
    v = np.cross(u,w)
    rnd = smp.sample2d(ps.o)
    sample_d = cosine_weighted_sample_on_hemisphere(rnd[0],rnd[1])
    d = normalize(sample_d[0]*u+sample_d[1]*v+sample_d[2] * w)
    pdf = np.dot(N,sample_d) * M_INVPI

    ps.t *= Ci / ray.pdf
    return Ci,Ray(P+N*0.0001,d,pdf)

def shade_specular(ps,P,N,ray,Ci):
    #specular
    d = normalize(ray.d - 2 * np.dot(ray.d, N) * N)
    pdf = 1
    return Ci,Ray(P+N*0.0001,d,pdf)

def shade(ps,P,N,ray,Cs):
    #calculate direct lighting
    rayToLight = Ray(P+N*0.0001,(L-P))
    rayToLight.d = normalize(rayToLight.d)
    #calculate radiance here and pass it to be added if ray hits?
    rad = ps.t * ((Cs * color_light) * (np.dot(N,rayToLight.d) * M_INVPI))
    #jsonmessage = {'ps':ps.to_dict(),'P':P.tolist(),'N':N.tolist(),'ray':rayToLight.to_dict(),'Cs':Cs.tolist()}
    jsonmessage = {'ps':ps.to_dict(),'ray':rayToLight.to_dict(),'rad':rad.tolist()}    
    occlusionqueue.put(jsonmessage)

    #generate the next ray
    Cs,newray = shade_diffuse(ps,P,N,ray,Cs)
    newray.depth = ray.depth+1
    #add to the ray queue
    if newray.depth < depth_max:
        jsonmessage = {'ps':ps.to_dict(),'ray':newray.to_dict()}
        rayqueue.put(jsonmessage)

def shade_callback(ch, method, properties, body):
    shadecontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(shadecontext['ps'])
    P = np.array(shadecontext['P'])
    N = np.array(shadecontext['N'])
    ray = Ray()
    ray.from_dict(shadecontext['ray'])
    Cs = np.array(shadecontext['Cs'])
    shade(ps,P,N,ray,Cs)


@app.route('/exit')
def exit():
    quit = True
    return('quitting')

@app.route('/flush')
def flush():
    channel.basic_consume(camera_ray_callback,queue='camerarayqueue',no_ack=True)

def setup_listeners():
    channel.basic_consume(shade_callback,queue='shadequeue',no_ack=True)
    channel.start_consuming()

depth_max = 4  # Maximum number of light reflections.
samples = 16
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

camerarayqueue =  QueueWrapper('camerarayqueue')
rayqueue = QueueWrapper('rayqueue')
shadequeue = QueueWrapper('shadequeue')
occlusionqueue = QueueWrapper('occlusionqueue')

if __name__ == '__main__':
    print (sys.argv)
    setup_listeners()