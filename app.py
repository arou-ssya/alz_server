from flask import Flask
import os
import gdown

app = Flask(__name__)

# Configuration Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&confirm=t&id=1Qm-lh5Fxw_7ojUYVKY81YHcmQ7uOAIRH"
MODEL_PATH = "model/alzheimer_model_float32.tflite"

# Téléchargement du modèle
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé avec succès")
    except Exception as e:
        print(f"❌ Erreur de téléchargement : {str(e)}")
        # Solution de secours
        os.system(f"wget --no-check-certificate '{MODEL_URL}' -O {MODEL_PATH}")
else:
    print("✅ Modèle déjà présent")

@app.route('/')
def home():
    return "API Alzheimer opérationnelle"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
