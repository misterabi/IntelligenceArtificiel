# Utilisation d'une image Python officielle de base
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . .

# Installer les dépendances Python à partir du requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port nécessaire pour votre application (par exemple, 8000 pour FastAPI, 8501 pour Streamlit)
EXPOSE 8000

# Commande de démarrage de l'application (à adapter selon votre application)
CMD ["python", "src/app/back.py"]
