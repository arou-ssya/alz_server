from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import urllib.request
import shutil

app = Flask(__name__)
CORS(app)

# Configuration robuste
MODEL_ID = "1a9yahheIt8Bc2c_ssCo2DItvucAqRK4P"
MODEL_DIR = os.path.join(os.getcwd(), "model")  # Chemin absolu
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model.tflite")
TEMP_PATH = os.path.join(MODEL_DIR, "temp.tflite")  # Fichier temporaire

def download_with_retry():
    """Téléchargement avec plusieurs tentatives et méthodes"""
    try:
        # Tentative 1: gdown standard
        try:
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", TEMP_PATH, quiet=False)
            if validate_model(TEMP_PATH):
                return True
        except Exception as e:
            print(f"⚠️ Échec gdown: {e}")

        # Tentative 2: Téléchargement direct
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(
                f"https://drive.google.com/uc?export=download&id={MODEL_ID}",
                headers=headers
            )
            with urllib.request.urlopen(req) as response, open(TEMP_PATH, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            if validate_model(TEMP_PATH):
                return True
        except Exception as e:
            print(f"⚠️ Échec téléchargement direct: {e}")

        # Tentative 3: wget (si disponible)
        try:
            os.system(f'wget --no-check-certificate "https://drive.google.com/uc?export=download&id={MODEL_ID}" -O {TEMP_PATH}')
            if validate_model(TEMP_PATH):
                return True
        except Exception as e:
            print(f"⚠️ Échec wget: {e}")

        return False
    except Exception as e:
        print(f"❌ Erreur lors des tentatives de téléchargement: {e}")
        return False

def validate_model(model_path):
    """Validation du modèle téléchargé"""
    try:
        # Vérification basique
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 100_000:
            return False
            
        # Vérification avec TensorFlow
        tf.lite.Interpreter(model_path=model_path)
        return True
    except:
        return False

# Initialisation du modèle
def setup_model():
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if os.path.exists(MODEL_PATH) and validate_model(MODEL_PATH):
            print("✅ Modèle déjà présent et valide")
            return True
            
        print("⏳ Téléchargement du modèle...")
        
        if download_with_retry():
            shutil.move(TEMP_PATH, MODEL_PATH)
            print("✅ Modèle téléchargé et validé")
            return True
            
        print("❌ Toutes les méthodes de téléchargement ont échoué")
        return False
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        return False

# Initialisation au démarrage
if not setup_model():
    print("❌ Impossible de charger le modèle - Vérifiez l'accès à Google Drive")
    exit(1)

# Chargement final
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("⚡ Modèle TensorFlow Lite chargé avec succès!")
except Exception as e:
    print(f"❌ Échec du chargement final: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        # Traitement de l'image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Prédiction
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


