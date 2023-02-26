import pathlib
import cv2
from flask_restful import Api, Resource
from flask import Flask, jsonify, request
from io import BytesIO
from base64 import b64decode
import base64
from decimal import *
import pandas as pd
from operator import pos
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import os
import io
import json
import numpy as np
from numpy import genfromtxt
import PIL
from PIL import Image
import tensorflow as tf
from numpy import asarray
from mtcnn.mtcnn import MTCNN


# from bson import json_util
# from fastai.vision.all import *
temp = pathlib.WindowsPath
pathlib.PosixPath = pathlib.WindowsPath


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    image.save("images/extractedFace.png", format='PNG')

    return image


def img_to_encoding(image_path, model):

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def decode_base64_image(base64_string):

    try:
        decoded_image = base64_string.split(',')[1]
        decoded_image = base64.b64decode(decoded_image)
        image_to_bytes = io.BytesIO(decoded_image)
        image = Image.open(image_to_bytes)

        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def save_img(image_object):
    try:

        img_string = image_object["image"]["dataURL"]
        img = decode_base64_image(img_string)
        img.save("images/sentData.png", format='PNG')

    except Exception as e:
        print(f"Error converting image: {e}")
        return None


print('running')
K.set_image_data_format('channels_last')
json_file = open('keras-facenet-h5/model.json', 'r')
print(json_file)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('keras-facenet-h5/model.h5')
FRmodel = model


def who_is_it(image_path, model):

    json_file = r"C:\Users\user\Desktop\Model-test\image_encodings.json"
    # encoding = img_to_encoding(image_path, model)
    encoding = img_to_encoding(image_path, model)
    min_dist = 100

    with open(json_file, 'r') as f:
        database = json.load(f)

    for (name, encodings) in database.items():
        for db_enc in encodings:

            dist = np.linalg.norm(encoding - db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
       print("Not in the database. min_dist = " + str(min_dist))
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


class Identify(Resource):

    def post(self):

        postedData = request.get_json()
        save_img(postedData)
        extract_face("images/sentData.png", required_size=(160, 160))

       
        min_dist, name = who_is_it('images/extractedFace.png', model)
        

        retJson = {
            "status": 200,
            "msg": "Recognized !",
            "name": name,
            "distance": min_dist
            
        }

        return jsonify(retJson)
