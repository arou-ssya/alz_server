FROM python:3.8-slim

WORKDIR /app

# Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et le modèle
COPY app.py .
COPY alzheimer_model_float32.tflite .

# Exposer le port
EXPOSE 10000

# Lancer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
