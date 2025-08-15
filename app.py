from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis Flutter

# Configuration du modèle
MODEL_URL = "https://drive.google.com/uc?export=download&confirm=t&id=12zWlffsA0K6czhT1ohELJVN7UoT7AXpH"
MODEL_PATH = "alzheimer_model_float32.tflite"

# Télécharger le modèle au démarrage
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé depuis Google Drive")
    except Exception as e:
        print(f"❌ Erreur de téléchargement : {str(e)}")
        exit(1)

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        # 1. Lire et prétraiter l'image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  # Adaptez à la taille attendue par votre modèle
        image_array = np.array(image) / 255.0  # Normalisation
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # 2. Faire la prédiction
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # 3. Renvoyer le résultat
        return jsonify({
            "prediction": predictions.tolist()[0],
            "message": "Prédiction réussie"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API Alzheimer opérationnelle"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

