import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os
import requests
import shutil
# Vérifiez la version de TensorFlow
print(f"Version de TensorFlow: {tf.__version__}")
print(f"Version de TensorFlow Lite: {tf.lite.__version__}")

# Chargement du modèle avec vérification
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("⚡ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur de chargement: {str(e)}")
    print("Essayez de re-convertir votre modèle avec la même version de TF")
    exit(1)

app = Flask(__name__)
CORS(app)

# Hugging Face URL du modèle
HF_MODEL_URL = "https://huggingface.co/aroussya/alzheimer-tflite/resolve/main/alzheimer_model_float32.tflite?download=true"

# Dossier pour stocker le modèle
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model_float32.tflite")
TEMP_PATH = os.path.join(MODEL_DIR, "temp.tflite")

def download_model():
    """Télécharge le modèle depuis Hugging Face si pas déjà présent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        print("✅ Modèle déjà présent")
        return True
    
    try:
        print("⏳ Téléchargement du modèle depuis Hugging Face...")
        with requests.get(HF_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(TEMP_PATH, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        shutil.move(TEMP_PATH, MODEL_PATH)
        print("✅ Modèle téléchargé avec succès")
        return True
    except Exception as e:
        print(f"❌ Échec du téléchargement Hugging Face: {e}")
        return False

# Télécharger le modèle avant le démarrage de l'API
if not download_model():
    print("❌ Impossible de charger le modèle")
    exit(1)

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("⚡ Modèle TensorFlow Lite chargé avec succès!")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({
            "prediction": predictions.tolist()[0],
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API Alzheimer opérationnelle"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

