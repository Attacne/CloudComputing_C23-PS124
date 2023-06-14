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

main = Flask(__name__)

model = load_model('model.h5') #load model

def load_model():
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded")
    return model

def predict_new_images(model, class_dict):
    train_dir = os.path.join("Capstone Attacne/train")
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1/255,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 12,
                                                    class_mode = 'categorical',
                                                    target_size = (128, 128))

    class_dict = {0: 'Blackhead', 1: 'Papules', 2: 'Pustules', 3: 'Nodules', 4:'Whitehead', 5: 'Healthy skin'}
    image_path = "uploaded/contoh.jpg" #location tempat MD Upload Image

    predict_image = Image.open(image_path)
    predict_image = tf.image.resize(predict_image, [128, 128])

    x = tf.keras.preprocessing.image.img_to_array(predict_image)/255
    x = np.expand_dims(x, axis=0)

    predict = model.predict(x)
    class_prediction = np.argmax(predict)
    prediction = class_dict[class_prediction]
    return prediction

def skincare_tips(prediction):
    
    # Define the class labels
    class_labels = ['Blackhead', 'Papules', 'Pustules', 'Nodules', 'Whitehead', 'Healthy skin']

    # Define the skincare tips for each class
    skincare_tips = {
        'Blackhead': 'Bersihkan wajah secara rutin dan gunakan produk skincare yang mengandung salicylic acid',
        'Whitehead': 'Lakukan exfoliasi secara rutin dengan alat dan produk yang lembut. Hindari produk-produk yang dapat menyumbat pori-pori kamu',
        'Papules': 'Aplikasikan produk yang dapat mengurangi inflamasi pada wajah, seperti menggunakan produk yang mengandung benzoyl peroxide atau salicylic acid',
        'Pustules': 'Gunakan produk-produk yang tidak mengandung minyak dan non-comedogenic',
        'Nodules': 'Jerawat jenis nodules perlu konsultasi dengan ahli dermatologi',
        'Healthy skin': 'Pertahankan kondisi wajahmu dengan tetap rutin menggunakan skincare yang cocok dengan kulitmu. Jangan lupa untuk selalu double cleansing, menggunakan moisturizer, dan memakain sunscreen untuk menjaga kulit tetap lembap dan terlindung dari sinar matahari.'
    }

    # Print the message and skincare tips
    if prediction == 'Healthy skin':
      print("Wajah kamu terlihat sehat dan bebas acne")
      print(skincare_tips[prediction])
    else: 
      print("Kamu mengalami", prediction, "pada wajahmu.")
      print("Berikut adalah hal yang dapat kamu lakukan untuk merawat wajahmu:")
      print(skincare_tips[prediction])
class_dict = {0: 'Blackhead', 1: 'Papules', 2: 'Pustules', 3: 'Nodules', 4:'Whitehead', 5: 'Healthy skin'}
predicted_class = predict_new_images(model, class_dict)
skincare_tips(predicted_class)