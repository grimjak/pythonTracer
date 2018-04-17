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
returnchannel = connection.channel()
'''channel.queue_declare(queue='camerarayqueue')
channel.queue_declare(queue='rayqueue')
channel.queue_declare(queue ='shadequeue')
channel.queue_declare(queue ='occlusionqueue')
'''




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

def occlusion(ps,P,N,ray,Ci):
    #calculate direct lighting
    if transmission(ray):
        rgb = ps.t * ((Ci * color_light) * (np.dot(N,ray.d) * M_INVPI))
        message = {'ps':ps.to_dict(),'rgb':rgb.tolist()}
        radiancequeue.put(message)
        #returnchannel.basic_publish(exchange='',routing_key='radiancequeue',body=json.dumps(message))

        #img[h-ps.j-1,ps.i] += rgb #need to add this to another queue that can then write to disk?
             

#trace a ray and add the hit point to the shade queue
def trace(ps,ray):
    traced = trace_ray(ray)
    if not traced:
        return

   # if ray.depth > depth_max:
        #camera ray

    obj, P, N, Cs = traced
    jsonmessage = {'ps':ps.to_dict(),'P':P.tolist(),'N':N.tolist(),'ray':ray.to_dict(),'Cs':Cs.tolist()}

    #shade(ps,M,N,ray,col_ray,smp)
    shadequeue.put(jsonmessage)


def shade(ps,P,N,ray,Cs):
    #calculate direct lighting
    rayToLight = Ray(P+N*0.0001,(L-P))
    rayToLight.d = normalize(rayToLight.d)
    #occlusion(ps,M,N,rayToLight,col_ray,smp) 
    #calculate radiance here and pass it to be added if ray hits?
    jsonmessage = {'ps':ps.to_dict(),'P':P.tolist(),'N':N.tolist(),'ray':rayToLight.to_dict(),'Cs':Cs.tolist()}
    occlusionqueue.put(jsonmessage)
    #occlusionqueue.put((ps,P,N,rayToLight,Cs))

    #generate the next ray
    Cs,newray = shade_diffuse(ps,P,N,ray,Cs)
    newray.depth = ray.depth+1
    #add to the ray queue
    if newray.depth < depth_max:
        jsonmessage = {'ps':ps.to_dict(),'ray':newray.to_dict()}
        rayqueue.put(jsonmessage)
        #rayqueue.put((ps,newray))
   # trace(ps,newray,smp)
 
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

def occlusion_callback(ch, method, properties, body):
    occlusioncontext = json.loads(body)
    ps = PixelSample()
    ps.from_dict(occlusioncontext['ps'])
    P = np.array(occlusioncontext['P'])
    N = np.array(occlusioncontext['N'])
    ray = Ray()
    ray.from_dict(occlusioncontext['ray'])
    Cs = np.array(occlusioncontext['Cs'])
    occlusion(ps,P,N,ray,Cs)

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
    time.sleep(6000)

def radiance_queue_handler():
    for method_frame, properties, body in channel.consume('radiancequeue'):
        radiancecontext = json.loads(body)
        ps = PixelSample()
        ps.from_dict(radiancecontext['ps'])
        rgb = np.array(radiancecontext['rgb'])
        img[h-ps.j-1,ps.i] += rgb #need to add this to another queue that can then write to disk?

def radiance_callback(ch, method, properties, body):
    imsave('tmp.png', (img/weights).clip(0,1), plugin='pil', format_str='png')

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

    worker = Thread(target = radiance_queue_handler)
    worker.setDaemon(True)
 #   worker.start()

    outstring = io.BytesIO()
    #img = np.clip(img,0,1)
    imsave(outstring, img.clip(0,1), plugin='pil', format_str='png')
    outstring.seek(0)
    return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/update')
def update():
    #jsonmessage = ['write']
    #radiancequeue.put(jsonmessage)    

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
    channel.basic_consume(ray_callback,queue='rayqueue',no_ack=True)
    channel.basic_consume(shade_callback,queue='shadequeue',no_ack=True)
    channel.basic_consume(occlusion_callback,queue='occlusionqueue',no_ack=True)
    #channel.basic_consume(radiance_callback,queue='radiancequeue',no_ack=True)

    channel.start_consuming()

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

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0.1, 0.1, 0.9]),
         add_sphere([-.75, .1, 2.25], .6, [0.18, 0.18, 0.18]),
         add_sphere([-2.75, .1, 3.5], .6, [0.9, .572, .184]),
         add_plane([0., -.5, 0.], [0.0, 1.0, 0.0]),
    ]

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

'''
camerarayqueue =  queue.Queue()
rayqueue = queue.Queue()
shadequeue = queue.Queue()
occlusionqueue = queue.Queue()
'''
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
    elif sys.argv[-1] == "server":
        setup_listeners()
    elif sys.argv[-1] == "writer":
        setup_writer()
