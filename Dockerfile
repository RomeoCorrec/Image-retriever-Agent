# 1. L'IMAGE DE BASE
FROM python:3.11-slim

# 2. INSTALLATION DES DÉPENDANCES SYSTÈME
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. LE DOSSIER DE TRAVAIL
WORKDIR /app

# 4. OPTIMISATION DE L'INSTALLATION (LA CORRECTION EST ICI)
COPY requirements.txt .
RUN pip install --upgrade pip

# ÉTAPE A : On installe PyTorch (Version CPU seulement) - C'est le plus gros morceau
# Cela évite de télécharger les drivers Nvidia inutiles dans Docker
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ÉTAPE B : On installe TensorFlow et Keras (Le 2ème monstre)
RUN pip install --no-cache-dir tensorflow tf-keras

# ÉTAPE C : On installe le reste des dépendances IA lourdes
RUN pip install --no-cache-dir deepface transformers ultralytics

# ÉTAPE D : On installe tout le reste (FastAPI, Streamlit, etc.)
# Pip verra que les gros sont déjà là et passera vite dessus
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 5. CONFIGURATION DES DOSSIERS
RUN mkdir -p /app/uploads && mkdir -p /app/models

# 6. VARIABLES D'ENVIRONNEMENT
ENV DEEPFACE_HOME=/app/models
ENV HF_HOME=/app/models
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 7. COPIE DU CODE
COPY . .

# 8. OUVERTURE DES PORTS
EXPOSE 8000
EXPOSE 8501

CMD ["bash"]