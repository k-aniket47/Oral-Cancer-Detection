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

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    image_duplicate=img.copy()
    hsv = cv2.cvtColor(image_duplicate,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h_new = (h + 90) % 180
    hsv_new = cv2.merge([h_new,s,v])
    bgr_new = cv2.cvtColor(hsv_new,cv2.COLOR_HSV2BGR)
    ave_color = cv2.mean(bgr_new)[0:3]
    color_img = np.full_like(image_duplicate, ave_color)
    blend = cv2.addWeighted(image_duplicate, 0.5, color_img, 0.5, 0.0)
    result = skimage.exposure.rescale_intensity(blend, in_range='image', out_range=(0,255)).astype(np.uint8)

    img = cv2.resize(result, (299, 299), interpolation=cv2.INTER_CUBIC)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="Normal"
    else:
        preds="OSCC"

    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
