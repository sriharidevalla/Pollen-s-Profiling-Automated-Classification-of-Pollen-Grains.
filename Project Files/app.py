import os
from flask import Flask, render_template, request, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

class_names = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia',
    'serjania', 'syagrus', 'tridax', 'urochloa'
]

app = Flask(__name__)

MODEL_PATH = 'cnn.hdf5'
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
IMG_SIZE = (128, 128)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = 'No file selected.'
            return render_template('prediction.html', error=error)
        file = request.files['file']
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Unknown class (index {pred_idx})"
        confidence = float(preds[pred_idx]) if pred_idx < len(preds) else 0.0
        prediction = {
            'predicted_type': pred_class,
            'confidence': confidence,
            'img_file': filename  # Only the filename, served via uploads route
        }
        return render_template('prediction.html', prediction=prediction)
    return render_template('prediction.html', error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
