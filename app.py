from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo de clasificación
model = load_model('colibri_classifier_model.h5')
class_labels = {0: 'Colibrí Inca ventrivioleta (Coeligena helianthea)',
                1: 'Colibrí picoespada (Ensifera ensifera)'}

def predict_class(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    predicted_class_index = np.argmax(preds, axis=1)[0]
    return class_labels[predicted_class_index], float(np.max(preds))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'label': None, 'probability': None})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'label': None, 'probability': None})
    
    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    
    label, probability = predict_class(file_path)
    return jsonify({'label': label, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
