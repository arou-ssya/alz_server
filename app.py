import os
import gdown
import pkg_resources
required = {
    'flask': '2.1.3',
    'werkzeug': '2.1.2'
}
for pkg, ver in required.items():
    pkg_resources.require(f"{pkg}=={ver}")

from flask import Flask

app = Flask(__name__)

# Config Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1Qm-lh5Fxw_7ojUYVKY81YHcmQ7uOAIRH"  # ID extrait de votre lien
MODEL_PATH = "alzheimer_model_float32.tflite"

# Téléchargement au démarrage
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Modèle téléchargé depuis Google Drive")
else:
    print("✅ Modèle déjà présent")

@app.route('/')
def home():
    return "Modèle Alzheimer chargé avec succès!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Port obligatoire pour Render

