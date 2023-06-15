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

# Inisialisasi klien penyimpanan Google Cloud
service_account = 'attacne-project-6bd247a71c17.json'
client = storage.Client.from_service_account_json(service_account)

# Inisialisasi flask
main = Flask(__name__)

#load model dari file model
bucket_name = 'attacne_dataset'
model_path = 'model.h5'
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(model_path)
model_path_local = 'model.h5'
blob.download_to_filename(model_path_local)
loaded_model = tf.keras.models.load_model(model_path_local)

model = tf.keras.models.load_model(model_path_local) #load model

def load_model():
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded")
    return model

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

# Define the allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#endpoint index/homepage
@main.route('/', methods=["GET"])
def index():
    return '<h1> <center> Welcome to Attacne API Homepage, port 5000 </center> </h1>'

@main.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image file found'})

    file = request.files['file']

    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed file types are PNG, JPG, and JPEG'})

    # Save the file to a temporary directory
    filename = secure_filename(file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)

    # Load and preprocess the image
    img = image.load_img(temp_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    skincare_tip = skincare_tips[predicted_label]

    # Return the prediction and skincare tip
    return jsonify({'prediction': predicted_label, 'skincare_tip': skincare_tip})

if __name__ == '__main__':
    main.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))