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

# Installation des dépendances
requirements = {
    'tensorflow': '2.13.0',
    'numpy': '1.24.3',
    'flask': '2.3.2',
    'flask-cors': '3.0.10',
    'pillow': '9.5.0',
    'requests': '2.26.0'
}

def install_dependencies():
    for pkg, ver in requirements.items():
        try:
            __import__(pkg)
            print(f"✅ {pkg} déjà installé")
        except ImportError:
            print(f"⏳ Installation de {pkg}...")
            os.system(f"pip install {pkg}=={ver} --quiet")

install_dependencies()

# Vérification des versions (souple)
REQUIRED_VERSIONS = requirements

def check_versions():
    current_versions = {
        'tensorflow': tf.__version__,
        'numpy': np.__version__,
        'flask': importlib.metadata.version('flask')
        # 'requests': importlib.metadata.version('requests')  # ignoré pour éviter blocage
    }
    
    for lib, required_version in REQUIRED_VERSIONS.items():
        if lib not in current_versions:
            continue
        current_version = current_versions[lib]
        if current_version != required_version:
            print(f"⚠ Version différente de {lib}. Requis: {required_version}, Actuelle: {current_version} — Ignoré")
    
    print("✅ Vérification des versions terminée")
    return True  # Toujours True pour éviter blocage

check_versions()  # On lance la vérification sans bloquer

app = Flask(__name__)
CORS(app)

# Configuration Hugging Face
HF_MODEL_URL = "https://huggingface.co/aroussya/alzheimer-tflite/resolve/main/alzheimer_model_float32.tflite"
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model.tflite")

def download_model():
    """Télécharge le modèle depuis Hugging Face"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                if file_size > 100000:
                    print(f"✅ Modèle déjà présent ({file_size/1e6:.2f} MB)")
                    return True
                else:
                    os.remove(MODEL_PATH)
            
            print(f"⏳ Tentative {attempt + 1} de téléchargement...")
            
            headers = {"User-Agent": "Flask-App/1.0"}
            response = requests.get(HF_MODEL_URL, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_path = f"{MODEL_PATH}.tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            shutil.move(temp_path, MODEL_PATH)
            print(f"✅ Modèle téléchargé ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Tentative {attempt + 1} échouée: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return False

# Initialisation
if not download_model():
    sys.exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("⚡ Modèle TensorFlow Lite chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {str(e)}")
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({
            "prediction": prediction.tolist()[0],
            "status": "success",
            "model_version": "1.0"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "tensorflow_version": tf.__version__,
        "flask_version": importlib.metadata.version('flask')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Serveur démarré sur http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
