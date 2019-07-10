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
from flask import render_template
from flask_socketio import SocketIO, emit

import pika



app = Flask(__name__)

@app.route('/')
def files(filename):
    return filename

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

@app.route('/render')  
def startRender():
    print("render")

    return render_template('renderTemplate.html',name="test")

    #outstring = io.BytesIO()
    #img = np.clip(img,0,1)
    #imsave(outstring, img.clip(0,1), plugin='pil', format_str='png')
   ## outstring.seek(0)
  #  return send_file(outstring, attachment_filename='test.png',mimetype='image/png')

@app.route('/renders/<filename>')
def render_file(filename):
    return filename

depth_max = 4  # Maximum number of light reflections.
samples = 16
filterwidth = 6.0

O = np.array([0., 0.35, -1.])  # Camera.

r = float(w) / h

radiancequeue = QueueWrapper('radiancequeue')


if __name__ == '__main__':
    print (sys.argv)
    #run the rest server an wait for commands
    app.run(debug=True,host='0.0.0.0')
