from flask import Flask, jsonify, request 
import tensorflow as tf
import base64
import cv2
from tensorflow import keras
import numpy as np
from flask_cors import CORS, cross_origin
from PIL import Image



def load_object_model():
    # Path to the saved model file or directory
    model_path = './classifier.h5'

    # Load the saved model
    model = keras.models.load_model(model_path)

    return model

def load_color_model():
    # Path to the saved model file or directory
    model_path = './color_classifier.h5'

    # Load the saved model
    model = keras.models.load_model(model_path)

    return model

def load_cash_model():
    # Path to the saved model file or directory
    model_path = './cashReader.h5'

    # Load the saved model
    model = keras.models.load_model(model_path)

    return model


app = Flask(__name__)

model1=load_object_model()
model2=load_color_model()
model3=load_cash_model()

CORS(app)

data_cat= ['Bed', 'Chair', 'Sofa', 'Swivelchair', 'Table']
labels=['Black','Beige','Blue','Green','Red','White']
cashReader=['1 EG','5 EG','10 EG','20 EG','50 EG','100 EG','200 EG']

@app.route('/api',  methods=['PUT'])

def classify():
    inputchar=request.get_data()
    imagedata=base64.b64decode(inputchar)
    filename='something.jpg'
    with open(filename,'wb') as f:
        f.write(imagedata)
    
    
    answer = predict_image(filename,model1)
    return answer

    


@app.route('/api2',  methods=['PUT'])

def classify_color():
    inputchar=request.get_data()
    imagedata=base64.b64decode(inputchar)
    filename='something.jpg'
    with open(filename,'wb') as f:
        f.write(imagedata)
    
    predictions = predict_color(filename,model2)
        
    return predictions

@app.route('/api3',  methods=['PUT'])

def predict():
    inputchar=request.get_data()
    imagedata=base64.b64decode(inputchar)
    filename='something.jpg'
    with open(filename,'wb') as f:
        f.write(imagedata)
    
    predictions = predict_cash(filename,model2)
        
    return predictions

def predict_image(filename, model):
    img_ = keras.utils.load_img(filename, target_size=(180, 180))
    img_array = keras.utils.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_processed)
    score = tf.nn.softmax(prediction)
   
    print('object in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))
    #returning the prediction answer
    return data_cat[np.argmax(score)]
   


def predict_color(filename, model):
    img_ = keras.utils.load_img(filename, target_size=(180, 180))
    img_array = keras.utils.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_processed)
    score = tf.nn.softmax(prediction)
   
    print('color in image is {} with accuracy of {:0.2f}'.format(labels[np.argmax(score)],np.max(score)*100))
    #returning the prediction answer
    return labels[np.argmax(score)]

def predict_cash(filename, model):
    img_ = keras.utils.load_img(filename, target_size=(180, 180))
    img_array = keras.utils.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_processed)
    score = tf.nn.softmax(prediction)
   
    print('color in image is {} with accuracy of {:0.2f}'.format(cashReader[np.argmax(score)],np.max(score)*100))
    #returning the prediction answer
    return cashReader[np.argmax(score)]
   

# Run the application
if __name__ == '__main__':
    app.run(debug=True)