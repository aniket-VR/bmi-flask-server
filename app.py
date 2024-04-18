from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
import logging
import face_recognition 
import pandas as pd
import time
import os
def my_face_encoding(image_path):
    print(image_path)
    logging.info("Getting face encoding for image %s", image_path)
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print("no face found !!!")
        logging.warning("No face found in image %s", image_path)
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()
HEIGHT_PATH = "models/height_model.h5"
WEIGHT_PAHT = "models/weight_model.h5"
height_model = tf.keras.models.load_model(HEIGHT_PATH)
weight_model = tf.keras.models.load_model(WEIGHT_PAHT)


def predict_height_weight_BMI(input_img,height_model,weight_model):
    logging.info("Predicting height, weight, and BMI for image %s", input_img)
    start_time = time.time()
    test_array = np.expand_dims(np.array(my_face_encoding(input_img)),axis=0)
    height = np.ndarray.item(np.exp(height_model.predict(test_array)))
    weight = np.ndarray.item(np.exp(weight_model.predict(test_array)))
    bmi = weight / (height)**2
    end_time = time.time()
    runtime = end_time - start_time
    logging.info("Predicted height: %f, weight: %f, BMI: %f, runtime : %s", height, weight, bmi, runtime)
    return {'height':height,"weight":weight,"bmi":bmi,'runtime':runtime}


height_model = tf.keras.models.load_model(HEIGHT_PATH)
weight_model = tf.keras.models.load_model(WEIGHT_PAHT)
app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\Users\\aniket\\Desktop\\BMINewServer\\UPLOAD_FOLDER'  # Replace 'path/to/upload/folder' with the actual path where you want to save the uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["POST"]) 
def predict():

    file = request.files['image']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    input_image = 'UPLOAD_FOLDER/'+file.filename
    return predict_height_weight_BMI(input_image,height_model,weight_model)
    
    

if __name__ =='__main__':
    app.run(debug=True)