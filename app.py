from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'crop_disease_model_tf.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Define your class labels (same order as during training)
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Bacterial_spot', 'tomato_healthy']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/disease-prediction')
def prediction_page():
    return render_template('disease_prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('disease_prediction.html', prediction="No file selected")

    file = request.files['image']
    if file.filename == '':
        return render_template('disease_prediction.html', prediction="No file selected")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))  # Use model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100

    result = f"{predicted_class} ({confidence:.2f}% confidence)"
    return render_template('disease_prediction.html', prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

