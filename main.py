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
channel.queue_declare(queue='camerarayqueue')
channel.queue_declare(queue='rayqueue')
channel.queue_declare(queue ='shadequeue')
channel.queue_declare(queue ='occlusionqueue')


w = 640
h = 480
M_PI = 3.14159265358979323846
M_INVPI = 1/ M_PI

#camerarayqueue = queue.Queue()

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
        self.pdf = np.array(dict['pdf'])
        self.depth = np.array(dict['depth'])


class Sampler():
    def __init__(self,size):
        self.size = size
        self.index = random.randrange(1,size)
        self.sequencer = ghalton.Halton(2) #2D set of LDN
        self.points = self.sequencer.get(size)
    def sample2d(self):
        self.index += 1
        return self.points[self.index%self.size]

class PixelSample():
    def __init__(self,x=0,y=0,t=1.0):
        self.i = x
        self.j = y
        self.t = t #througput
    def to_dict(self):
        return self.__dict__
    def from_dict(self,dict):
        self.i = dict['i']
        self.j = dict['j']
        self.t = dict['t']

class QueueWrapper():
    def __init__(self,key):
        self.key = key
        channel.queue_declare(queue=key)

    def put(self,data):
        datalist = []
        for d in data:
            if type(d) is 'numpy.ndarray':
                datalist.append(d.tolist())
            else:
                datalist.append(d.to_dict())

        channel.basic_publish(exchange='',routing_key=self.key,body=json.dumps(datalist))
    def get(self,data): #rebuild output tuple
        outdata = ()
        for d in data:
            outdata+d
        return outdata



def normalize(x):
    x /= np.linalg.norm(x)
    return x

def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = sqrt(1.0 - u1)
    sin_theta = sqrt(u1)
    phi = 2.0 * M_PI * u2
    return np.array([cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta])

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


def environment(dir):
    return np.array([1.0,1.0,1.0])

#for a given hit point what's the current sample weight 
#add new rays to the ray queue
def shade_diffuse(ps,P,N,ray,Ci,smp):
    # Diffuse 
    w = N 
    u = np.array([0.00319,1.0,0.0078])
    u = np.cross(u,w)
    u = normalize(u)
    v = np.cross(u,w)
    rnd = smp.sample2d()
    sample_d = cosine_weighted_sample_on_hemisphere(rnd[0],rnd[1])
    d = normalize(sample_d[0]*u+sample_d[1]*v+sample_d[2] * w)
    pdf = np.dot(N,sample_d) * M_INVPI

    #update the radiance of the pixel sample directly with any emmision
    #then shoot another ray
    ps.t *= Ci / ray.pdf
    return Ci,Ray(P+N*0.0001,d,pdf)

def shade_specular(ps,P,N,ray,Ci,smp):
    #specular
    d = normalize(ray.d - 2 * np.dot(ray.d, N) * N)
    pdf = 1
    return Ci,Ray(P+N*0.0001,d,pdf)

def occlusion(t,i,j,P,N,ray,Ci,smp):
    #calculate direct lighting
    if transmission(ray):
        rgb = t * ((Ci * color_light) * (np.dot(N,ray.d) * M_INVPI))
        img[h-j-1,i] += rgb
             

#trace a ray and add the hit point to the shade queue
def trace(ps,ray,smp):
    traced = trace_ray(ray)
    if not traced:
        return

   # if ray.depth > depth_max:
        #camera ray

    obj, M, N, col_ray = traced
    #shade(ps,M,N,ray,col_ray,smp)
    shadequeue.put((ps,M,N,ray,col_ray,smp))


def shade(ps,M,N,ray,col_ray,smp):
    #calculate direct lighting
    rayToLight = Ray(M+N*0.0001,(L-M))
    rayToLight.d = normalize(rayToLight.d)
    #occlusion(ps,M,N,rayToLight,col_ray,smp) 
    occlusionqueue.put((ps.t,ps.i,ps.j,M,N,rayToLight,col_ray,smp))

    #generate the next ray
    Cs,newray = shade_diffuse(ps,M,N,ray,col_ray,smp)
    newray.depth = ray.depth+1
    #add to the ray queue
    rayqueue.put((ps,newray,smp))
   # trace(ps,newray,smp)

def trace_camera_ray(ps,ray,smp):
    traced = trace_ray(ray)
    if not traced:
        return
    obj,M,N, col_ray = traced
    shadequeue.put((ps,M,N,ray,col_ray,smp))

#def camera_ray_queue_handler():
##    print ("camera_ray_queue_handler active")
#    while True:
#        raycontext = camerarayqueue.get()
#        ps,ray,smp = raycontext
#        trace_camera_ray(ps,ray,smp)

def camera_ray_callback(ch, method, properties, body):
    print("[x] Received: ")
    raycontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(raycontext[0])
    ray = Ray()
    ray.from_dict(raycontext[1])
    trace_camera_ray(ps,ray,smp)

def ray_queue_handler():
    while True:
        raycontext = rayqueue.get()
        ps,newray,smp = raycontext
        trace(ps,newray,smp)
        rayqueue.task_done()

def shade_queue_handler():
    while True:
        shadecontext = shadequeue.get()
        ps,P,N,ray,Cs,smp = shadecontext
        shade(ps,P,N,ray,Cs,smp)
        shadequeue.task_done()

def occlusion_queue_handler():
    while True:
        occlusioncontext = occlusionqueue.get()
        t,i,j,P,N,ray,col_ray,smp = occlusioncontext
        occlusion(t,i,j,P,N,ray,col_ray,smp)
        occlusionqueue.task_done()

def sample(ps,ray):
    smp = Sampler(256)
 #   trace_camera_ray(ps,ray,smp)
    #camerarayqueue.put((ps,ray,smp))
    camerarayqueue.put((ps,ray))

#    print(json.dumps(ps.__dict__))
   # channel.basic_publish(exchange='',routing_key='camerarayqueue',body=json.dumps({'ray':ray.__dict__(),'ps':ps.__dict__}))
 #   return

 ##   traced = trace_ray(ray)
#    if not traced:
#       return
#    obj,M,N, col_ray = traced
    #shade(ps,M,N,ray,col_ray,smp)
#    shadequeue.put((ps,M,N,ray,col_ray,smp))


def render_scene():
    #bin rays based on direction (and origin?)
    #go through rays in ordered batches producing a list of hit points
    #shade hit points, secondary rays are fed back into ray bins
    #samples = 4

    #smp = Sampler(w*h*samples)
    col = np.zeros(3)  # Current color.
    Q = np.array([0., 0., 0.])  # Camera pointing to.

    # Loop through all pixels.
    for s in range(samples):   
        for i, x in enumerate(np.linspace(S[0], S[2], w)):
            if i % 10 == 0:
                print (i / float(w) * 100, "%")
            for j, y in enumerate(np.linspace(S[1], S[3], h)):
            #for s in range(samples): 
                offsetu = filterwidth * smp.sample2d()[0]
                offsetv = filterwidth * smp.sample2d()[1]
                Q[:2] = (x+(offsetu-(filterwidth/2.0))/w, y+(offsetv-(filterwidth/2.0))/h)

                D = normalize(Q - O)
                ray = Ray(O, D)

                ps = PixelSample(i,j)
                weights[h-j-1,i] += 1 
                sample(ps,ray)

                if stopped:
                    break

                if quit:
                    exit()
                #img[h - j - 1, i, :] += np.clip(ps.rgb, 0, 1) / samples
                #update_pixel(ps)
                #img[h-j-1,i] = (img[h-j-1,i] / weights[h-j-1,i]) + (np.clip(ps.rgb,0,1) / weights[h-j-1,i])
    wait(6000)


@app.route('/render')  
def startRender():
    print("render")
    #t = Thread(target = render_scene)
    #start trace worker threads
    #for i in range(cameraray_threads):
     #   worker = Thread(target = camera_ray_queue_handler)
    #    worker.setDaemon(True)
    #    worker.start()

    #start trace worker threads
    for i in range(ray_threads):
        worker = Thread(target = ray_queue_handler)
        worker.setDaemon(True)
        worker.start()

    #start shade worker threads
    for i in range(shade_threads):
        worker = Thread(target = shade_queue_handler)
        worker.setDaemon(True)
        worker.start()

    #start shade worker threads
    for i in range(occlusion_threads):
        worker = Thread(target = occlusion_queue_handler) 
        worker.setDaemon(True)
        worker.start()

    print("start render")
  #  render_scene()
    if not t.is_alive():
        print("starting render")
        t.start()
    else:
        stopped = True
        t.join()
        t.start()
    outstring = io.BytesIO()
    #img = np.clip(img,0,1)
    imsave(outstring, img.clip(0,1), plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/update')
def update():
    outstring = io.BytesIO()
    #img = np.clip(img,0,1)
    imsave(outstring, (img/weights).clip(0,1), plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/exit')
def exit():
    quit = True
    return('quitting')

@app.route('/flush')
def flush():
    channel.basic_consume(camera_ray_callback,queue='camerarayqueue',no_ack=True)


def setup_listeners():
    channel.basic_consume(camera_ray_callback,queue='camerarayqueue',no_ack=True)
    channel.start_consuming()

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0.1, 0.1, 0.9]),
         add_sphere([-.75, .1, 2.25], .6, [0.18, 0.18, 0.18]),
         add_sphere([-2.75, .1, 3.5], .6, [0.9, .572, .184]),
         add_plane([0., -.5, 0.], [0.0, 1.0, 0.0]),
    ]

cameraray_threads = 2
ray_threads = 2
shade_threads = 2
occlusion_threads = 2
depth_max = 5  # Maximum number of light reflections.
samples = 16
filterwidth = 2.0
smp = Sampler(w*h*samples)
t = Thread(target = render_scene)


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

#smp = Sampler()
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)


camerarayqueue = QueueWrapper('camerarayqueue')
rayqueue = queue.Queue()
shadequeue = queue.Queue()
occlusionqueue = queue.Queue()

if __name__ == '__main__':
    print (sys.argv)
    if sys.argv[-1] == "control":    
        #run the rest server an wait for commands
        app.run(debug=True,host='0.0.0.0')
    elif sys.argv[-1] == "server":
        setup_listeners()
