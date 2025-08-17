import os
import time
import importlib.metadata
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import requests
import shutil
import sys
import tensorflow as tf
import numpy as np

# Configuration des versions requises
REQUIRED_VERSIONS = {
    'tensorflow': '2.15.0',  # Version mise à jour pour compatibilité TFLite
    'numpy': '1.24.3',
    'flask': '2.3.2',
    'flask-cors': '3.0.10',
    'pillow': '10.4.0',
    'requests': '2.26.0'
}

def check_versions():
    """Vérifie que toutes les versions des dépendances sont correctes"""
    try:
        current_versions = {}
        for lib in REQUIRED_VERSIONS.keys():
            try:
                current_versions[lib] = importlib.metadata.version(lib)
            except importlib.metadata.PackageNotFoundError:
                print(f"❌ Package {lib} non installé")
                return False

        for lib, required_version in REQUIRED_VERSIONS.items():
            if current_versions[lib] != required_version:
                print(f"❌ Version incorrecte de {lib}. Requis: {required_version}, Actuelle: {current_versions[lib]}")
                return False
        
        print("✅ Toutes les versions sont correctes")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des versions: {str(e)}")
        return False

# Vérification initiale
if not check_versions():
    sys.exit(1)

# Initialisation de Flask
app = Flask(__name__)
CORS(app)

# Configuration Hugging Face
HF_MODEL_URL = "https://huggingface.co/aroussya/alzheimer-tflite/resolve/main/alzheimer_model_float32.tflite"
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model.tflite")

def download_model():
    """Télécharge le modèle depuis Hugging Face avec gestion des erreurs améliorée"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Vérifie si le modèle existe déjà et a une taille valide
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                if file_size > 100000:  # > 100KB
                    print(f"✅ Modèle déjà présent ({file_size/1e6:.2f} MB)")
                    return True
                else:
                    os.remove(MODEL_PATH)
            
            print(f"⏳ Tentative {attempt + 1} de téléchargement...")
            
            headers = {"User-Agent": "Flask-App/1.0"}
            response = requests.get(HF_MODEL_URL, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Téléchargement temporaire
            temp_path = f"{MODEL_PATH}.tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filtre les chunks vides
                        f.write(chunk)
            
            # Validation avant déplacement final
            if os.path.getsize(temp_path) < 100000:
                raise ValueError("Fichier modèle trop petit")
            
            shutil.move(temp_path, MODEL_PATH)
            print(f"✅ Modèle téléchargé ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Tentative {attempt + 1} échouée: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return False

# Initialisation du modèle
def load_model():
    """Charge le modèle avec vérification de compatibilité"""
    try:
        if not download_model():
            return None

        # Vérification de la version de TensorFlow
        print(f"ℹ️ Version de TensorFlow: {tf.__version__}")
        
        # Chargement du modèle
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        
        # Vérification de la version du modèle
        model_version = interpreter._model_version
        print(f"ℹ️ Version du modèle TFLite: {model_version}")
        
        if model_version > 12:
            print("⚠️ Attention: Le modèle nécessite une version plus récente de TFLite")
            print("Solution: Mettez à jour TensorFlow ou utilisez un modèle plus ancien")
            return None
        
        interpreter.allocate_tensors()
        print("⚡ Modèle TensorFlow Lite chargé avec succès!")
        return interpreter
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle: {str(e)}")
        return None

# Initialisation globale
interpreter = load_model()
if interpreter is None:
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        # Traitement de l'image
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({
            "prediction": prediction.tolist()[0],
            "status": "success",
            "model_version": "1.0",
            "framework": f"TensorFlow {tf.__version__}"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "tensorflow_version": tf.__version__,
        "flask_version": importlib.metadata.version('flask'),
        "port": int(os.environ.get('PORT', 10000))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Serveur Flask démarré sur le port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
