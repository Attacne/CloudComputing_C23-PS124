from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import tempfile
from werkzeug.utils import secure_filename
from google.cloud import storage
import os
import io
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from predict.prediction import predict_new_images, skincare_tips

main = Flask(__name__)

model = load_model('model.h5') #load model

def load_model():
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded")
    return model

#endpoint index/homepage
@main.route('/', methods=["GET"])
def index():
    return '<h1> <center> Welcome to T2T API Homepage, port 4000 </center> </h1>'

@main.route('/predict', methods=['POST'])
def predict_skin_condition():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    # Save the uploaded image 
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image = (temp_file.name), 

    image.save(image)

    # Open and preprocess the image
    img = Image.open(image)
    img = predict_new_images(img)

    # Make predictions
    class_labels = ['Blackhead', 'Papules', 'Pustules', 'Nodules', 'Whitehead', 'Healthy skin']
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    skincare_tip = skincare_tips[predicted_class]

    # Return the predicted class and skincare tip
    return jsonify({"predicted_class": predicted_class, "skincare_tip": skincare_tip})

if __name__ == '__main__':
    main.run(debug=True, host='0.0.0.0')