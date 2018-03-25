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
        channel.queue_declare(queue=key)

    def put(self,data):
        channel.basic_publish(exchange='',routing_key=self.key,body=json.dumps(data))


def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(ray, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(ray.d, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - ray.o, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(ray, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(ray.d, ray.d)
    OS = ray.o - S
    b = 2 * np.dot(ray.d, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect(ray, obj):
    if obj['type'] == 'plane':
        return intersect_plane(ray, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(ray, obj['position'], obj['radius'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N
    
def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

#version of trace_ray which just returns geometric properties
def trace_ray(ray):
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(ray,obj)
        if t_obj < t:
            t,obj_idx = t_obj,i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
        # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = ray.o + ray.d * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M) 
    #color = np.array([0.18,0.18,0.18])  
    return obj,M,N,color

#version of trace_ray which just returns geometric properties
def transmission(ray):
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(ray,obj)
        if t_obj < np.inf:
            return False
    return True

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
        diffuse_c=.75, specular_c=.5, reflection=.25)

#def occlusion(ps,P,N,ray,Ci):
def occlusion(ps,ray,rad):
    #calculate direct lighting
    if transmission(ray):
       # rgb = ps.t * ((Ci * color_light) * (np.dot(N,ray.d) * M_INVPI))
        message = {'ps':ps.to_dict(),'rgb':rad.tolist()}
        radiancequeue.put(message)
             
#trace a ray and add the hit point to the shade queue
def trace(ps,ray):
    traced = trace_ray(ray)
    if not traced:
        return

    obj, P, N, Cs = traced
    Cs = np.ones(3)
    jsonmessage = {'ps':ps.to_dict(),'P':P.tolist(),'N':N.tolist(),'ray':ray.to_dict(),'Cs':Cs.tolist()}
    shadequeue.put(jsonmessage)
 
def trace_camera_ray(ps,ray):
    traced = trace_ray(ray)
    if not traced:
        return
    obj,P,N, Cs = traced
    jsonmessage = {'ps':ps.to_dict(),'P':P.tolist(),'N':N.tolist(),'ray':ray.to_dict(),'Cs':Cs.tolist()}
    shadequeue.put(jsonmessage)
    #shadequeue.put((ps,M,N,ray,Ci))

def camera_ray_callback(ch, method, properties, body):
    raycontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(raycontext['ps'])
    ray = Ray()
    ray.from_dict(raycontext['ray'])
    trace_camera_ray(ps,ray)

def ray_callback(ch, method, properties, body):
    raycontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(raycontext['ps'])
    ray = Ray()
    ray.from_dict(raycontext['ray'])
    trace(ps,ray)

def occlusion_callback(ch, method, properties, body):
    occlusioncontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(occlusioncontext['ps'])
    ray = Ray()
    ray.from_dict(occlusioncontext['ray'])
    rad = np.array(occlusioncontext['rad'])
    occlusion(ps,ray,rad)


@app.route('/exit')
def exit():
    quit = True
    return('quitting')

def setup_listeners():
    #channel.basic_consume(camera_ray_callback,queue='camerarayqueue',no_ack=True)
    channel.basic_consume(ray_callback,queue='rayqueue',no_ack=True)
    channel.basic_consume(occlusion_callback,queue='occlusionqueue',no_ack=True)
    channel.start_consuming()

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0.1, 0.1, 0.9]),
         add_sphere([-.75, .1, 2.25], .6, [0.18, 0.18, 0.18]),
         add_sphere([-2.75, .1, 3.5], .6, [0.9, .572, .184]),
         add_plane([0., -.5, 0.], [0.0, 1.0, 0.0]),
    ]

depth_max = 4  # Maximum number of light reflections.
samples = 4
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

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    setup_listeners()
  