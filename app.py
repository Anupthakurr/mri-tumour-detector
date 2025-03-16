from flask import Flask, render_template, request, send_from_directory, g
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

import gdown

# Ensure the models directory exists
MODEL_PATH = 'models/model.h5'
MODEL_URL = "https://drive.google.com/uc?id=180O_Lof0WV_sex4lmyppbEpe-sWBqBwa"  # Replace with your actual file ID

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Download model only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model... This will happen only once.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Initialize Flask app
app = Flask(__name__)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load model lazily when needed
def get_model():
    if 'model' not in g:
        print("Loading model into memory...")
        g.model = load_model(MODEL_PATH)
    return g.model

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode="rgb")  # Ensure correct format
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    model = get_model()  # Load model only when needed
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)

            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Set port dynamically
    app.run(host='0.0.0.0', port=port)  # Removed debug=True for production

