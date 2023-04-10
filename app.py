from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D 
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import cv2
import base64
from skimage.io import imread
import skimage.exposure

app = Flask(__name__)

gm_exp = tf.Variable(3., dtype=tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                           axis=[1,2], 
                           keepdims=False)+1.e-8)**(1./gm_exp)
    return pool

# load the model
path_to_model = 'model.hdf5'
print("Loading the model...")
model = load_model(path_to_model, compile=False)
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
print("Done!")

# define the classes
classes = {
    0: 'OSCC',
    1: 'Normal'
}


