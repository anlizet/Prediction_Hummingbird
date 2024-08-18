from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Cargar el modelo
model = load_model('colibri_classifier_model.h5')

# Diccionario de etiquetas
class_labels = {
    0: 'Colibrí Inca ventrivioleta (Coeligena helianthea)',
    1: 'Colibrí picoespada (Ensifera ensifera)'
}

def predict_with_labels(img_path, model, class_labels):
    # Cargar y preprocesar la imagen para la predicción
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predecir la clase
    preds = model.predict(x)

    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_index = np.argmax(preds, axis=1)[0]

    # Obtener la etiqueta de la clase predicha y probabilidad
    predicted_class_label = class_labels.get(predicted_class_index, 'Desconocida')
    probability = preds[0][predicted_class_index]
    return (predicted_class_label, probability)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    img_path = 'temp_image.jpg'
    file.save(img_path)


    try:
        predicted_label, probability = predict_with_labels(img_path, model, class_labels)
        return jsonify({'label': predicted_label, 'probability': float(probability)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:

        import os
        os.remove(img_path)

if __name__ == '__main__':
    app.run(debug=True)

