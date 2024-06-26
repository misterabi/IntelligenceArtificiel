# Utilisation d'une image Python officielle de base
FROM python:3.12

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Installer les dépendances Python à partir du requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port nécessaire pour votre application FastAPI
EXPOSE 8501

# Commande de démarrage de l'application
CMD ["streamlit", "run", "front.py"]
