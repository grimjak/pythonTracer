import numpy as np
import matplotlib.pyplot as plt
import io
from skimage.io import imsave
from threading import Thread
import queue

from math import sqrt, cos, sin
import random
#from PIL import Image

from flask import Flask
from flask import send_file
from flask.ext.socketio import SocketIO, emit

import ghalton

app = Flask(__name__)
@app.route('/')

def hello_world():
    return 'test2 app'

w = 640
h = 480
M_PI = 3.14159265358979323846
M_INVPI = 1/ M_PI

rayqueue = queue.Queue()
shadequeue = queue.Queue()

class Ray():
    def __init__(self, O, D, PDF=1):
        self.o = O
        self.d = D
        self.pdf = 1.0
        return

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
    def __init__(self):
        self.rgb = np.array([0.0,0.0,0.0])
        self.t = 1.0 #througput
        self.d = 0 #current depth, do we need to track this?

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
def trace(ray):
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
def shade(ps,P,N,ray,Ci,smp):
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

    #update pixel sample throughput based on surface colour and pdf
    #pdf should be applied to the result of the hit not the first hit
    #should pdf be on the ray?
    #ps.t *= Cs / ray.pdf
    return Ci,Ray(P+N*0.0001,d,pdf)

def sample(ps,ray):
    depth = 0
    smp = Sampler(256)
    # Loop through initial and secondary rays.
    while depth < depth_max:
        traced = trace(ray)
        if not traced:
            #miss, add environment
           # ps.rgb += ps.t * environment(ray.d)
            break
        obj, M, N, col_ray = traced
        depth += 1

        #calculate direct lighting
        rayToLight = Ray(M,(L-M))
        traced = trace(ray)
        if traced:
            Cs, rayToLight = shade(ps,M,N,rayToLight,col_ray,smp) #don't want to affect overall pixel sample throughput with direct lighting
            ps.rgb += (ps.t * Cs * color_light) / rayToLight.pdf

        Cs,ray = shade(ps,M,N,ray,col_ray,smp)
        ps.t *= Cs / ray.pdf
        # Reflection: create a new ray.
       # ray.o, ray.d = M + N * .0001, normalize(ray.d - 2 * np.dot(ray.d, N) * N)
       # pdf = np.dot(N,ray.d)

    return ps


def render_scene():
    #bin rays based on direction (and origin?)
    #go through rays in ordered batches producing a list of hit points
    #shade hit points, secondary rays are fed back into ray bins
    samples = 1

    smp = Sampler(w*h*samples)
    col = np.zeros(3)  # Current color.
    Q = np.array([0., 0., 0.])  # Camera pointing to.

    # Loop through all pixels.
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        if i % 10 == 0:
            print (i / float(w) * 100, "%")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            col[:] = 0
            for s in range(samples): 
                offset = smp.sample2d()
                Q[:2] = (x+(offset[0]-0.5)/w, y+(offset[1]-0.5)/h)
                #Q[:2] = (x, y)

                D = normalize(Q - O)
                #depth = 0
                ray = Ray(O, D)
                #reflection = 1.
                #col = sample(O,D)
                ps = PixelSample()
                ps = sample(ps,ray)
                #col = np.array([1,0,0])
                img[h - j - 1, i, :] += np.clip(ps.rgb, 0, 1) / samples

@app.route('/fullrender')  
def fullRender():  
    render_scene()
    outstring = io.BytesIO()
    imsave(outstring, img, plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/render')  
def startRender():
    t = Thread(target = render_scene)
    t.start()   
    #render_scene()
    outstring = io.BytesIO()
    imsave(outstring, img, plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/update')
def update():
    outstring = io.BytesIO()
    imsave(outstring, img, plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0.1, 0.1, 0.9]),
         add_sphere([-.75, .1, 2.25], .6, [0.18, 0.18, 0.18]),
         add_sphere([-2.75, .1, 3.5], .6, [0.9, .572, .184]),
         add_plane([0., -.5, 0.], [0.0, 1.0, 0.0]),
    ]

depth_max = 1  # Maximum number of light reflections.

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

#smp = Sampler()
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')