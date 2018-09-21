from flask import Flask, jsonify, abort, make_response, request
from io import BytesIO
from PIL import Image
from PIL import ImageOps
import requests
import base64
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import load_model

model = load_model('weights.09-0.04.hdf5')

# Label names
names = ["pulleys","helmets","crampons","harnesses","insulated_jackets","axes","rope","boots","hardshell_jackets","carabiners","tents","gloves"]
# Load model 
# MODEL_NAME  ="AdventureWorksModel"
# model = Sequential()
# model.add(InputLayer(input_shape=(128,128,3))) #入力層
# model.add(Conv2D(64, 3, padding='valid')) #畳み込み層
# model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)) #プーリング層
# model.add(Conv2D(32, 3, padding='valid')) #畳み込み層
# model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)) #プーリング層

# #最終的に1次元に変換して
# model.add(Flatten())
# #↓全結合層で12個のラベルに変換する
# model.add(Dense(12, activation='softmax'))
# #コンパイル　引数はドキュメンコピペ
# model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# model.load_weights('kohhei_challenge4_weight.hdf5')

# Get name from prediction result
def find_name(prediction_result):
    for score, name in zip(prediction_result, names):
        if score == np.max(prediction_result):
            return name, score

def prepare_img(img):
    resized_img = resize(img)
    normalized_img = normalize(resized_img)
    return normalized_img

# Resize file to 128x128
def resize(img):    
    try:
        # Create thumbnail. Image.thumbnail replace the file.
        img.thumbnail((128, 128), Image.LANCZOS)        
        # Adding white padding
        w,h = img.size
        border=(0, 0, 128-w, 128-h) #'left,top,right,bottom'
        color='#FFFFFF'
        padding_im= ImageOps.expand(img, border, color)   
        return padding_im
    except:
        print("failed to resize")
        
# Normalize image
def normalize(img):    
    try:        
        norm_img = ImageOps.autocontrast(img,cutoff=0.2)
        return norm_img
    except:
        print("failed normalize")

# API
api = Flask(__name__)

# Predict using serialized image
@api.route('/predict/img', methods=['POST'])
def predict_img():
    try:
        x = []
        # Get image from serialized json and prepare data
        post_data = base64.b64decode(request.get_json()["image"])
        im = Image.open(BytesIO(post_data)).convert('RGB')
        arr = np.array(prepare_img(im))
        x.append(arr)
        x = np.asarray(x, dtype=np.float64)
        x /= 255
        # Run prediction and get label
        result = model.predict(x)
        name, score = find_name(result[0])
    except:
        abort(404)

    result = {
        "result":True,
        "data":{
            "product":name,
            "score":str(score)
            }
        }

    return make_response(jsonify(result))

# Predict using URL
@api.route('/predict/url', methods=['POST'])
def predict_url():
    try:
        x = []
        # Get image from URL and prepare data
        url = request.get_json()["url"]
        response = requests.get(url)    
        im = Image.open(BytesIO(response.content)).convert('RGB')
        arr = np.array(prepare_img(im))
        x.append(arr)
        # Run prediction and get label
        x = np.asarray(x, dtype=np.float64)
        x /= 255
        result = model.predict(x)
        name, score = find_name(result[0])
    except:
        abort(404)

    result = {
        "result":True,
        "data":{
           "product":name,
           "score":str(score)
            }
        }

    return make_response(jsonify(result))

@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    api.run(host='0.0.0.0', port=8080)