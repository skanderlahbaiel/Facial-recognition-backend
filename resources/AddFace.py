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
from datetime import datetime
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
import re
import json
from resources.identify import img_to_encoding, extract_face, save_img, decode_base64_image
from resources.identify import FRmodel


# from bson import json_util
# from fastai.vision.all import *
temp = pathlib.WindowsPath
pathlib.PosixPath = pathlib.WindowsPath
json_file = "Model-test/image_encodings.json"


def img_to_encoding_to_json(image_path, name, model, json_file):
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = model.predict_on_batch(x_train)
        embedding = embedding / np.linalg.norm(embedding, ord=2)

        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                file_contents = f.read()
                if file_contents.strip():
                    database = json.loads(file_contents)
                else:
                    database = {}
        else:
            database = {}

        # Add the embedding for the current image to the dictionary
        if name in database:
            database[name].append(embedding.tolist())
        else:
            database[name] = [embedding.tolist()]

        # Write the updated dictionary back to the JSON file
        with open(json_file, 'w') as f:
            json.dump(database, f, indent=4)

        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

   


def save_img(image_object):
    try:

        img_string = image_object["image"]["dataURL"]
        name = image_object["image"]["name"]
        img = decode_base64_image(img_string)
        img_path = extract_face_and_save(img, name, required_size=(160, 160))
        return img_path
    except Exception as e:
        print(f"Error converting image: {e}")
        return None


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


def extract_face_and_save(img, name, required_size=(160, 160)):
    # convert to array
    pixels = asarray(img)
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
    current_year = datetime.now()
    # remove spaces and special characters using regular expressions
    formatted_datetime = re.sub('[^A-Za-z0-9]+', '', str(current_year))
    image.save(f"images/{name+formatted_datetime}.png", format='PNG')
    path = f"images/{name+formatted_datetime}.png"

    return path

class Addface(Resource):

    def post(self):

        #   You should send data in this json form:
        #   {"image":{
        #   "dataURL":"",
        #   "name": ""}}

        postedData = request.get_json()
        img_path = save_img(postedData)
        print(img_path)
        
        name = postedData["image"]["name"]
        img_to_encoding_to_json(img_path, name, FRmodel, r"C:\Users\user\Desktop\Model-test\image_encodings.json")

        retJson = {
            "status": 200,
            "msg": "Success !",
            "data": postedData["image"]["name"]
        }

        return jsonify(retJson)
