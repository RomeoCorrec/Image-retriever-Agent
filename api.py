import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from pathlib import Path

# Imports locaux
from fonctions import (
    load_clip_model_processor, # Pour pré-charger CLIP
    add_image_with_person_name_from_path, 
    add_face_with_person_name_from_path
)
from tools_gem import retrieve_images_by_persons_names_and_image_description

# 1. Setup & Connexions
load_dotenv()

# Configurer Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configurer Qdrant
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
print(f"Connexion à Qdrant sur : {qdrant_url}")
client = QdrantClient(url=qdrant_url) # Pas de clé en local

# Pré-charger CLIP au démarrage de l'API pour éviter les lenteurs
print("Chargement de CLIP...")
load_clip_model_processor()

app = FastAPI(title="Image Retriever API (Gemini Powered)")

# --- Modèles de données ---
class AgentRequest(BaseModel):
    query: str 

class ImageRequest(BaseModel):
    image_path: str

class FaceRequest(BaseModel):
    image_path: str
    person_name: str

# --- Wrapper pour l'outil ---
# C'est l'astuce : On crée une version de la fonction qui n'a PAS besoin de l'argument 'client'
# car on utilise le client global défini ci-dessus.
def search_tool(image_description: str, person_names: list = None):
    """
    Outil de recherche d'images. Utilise ça pour trouver des photos.
    """
    print(f"DEBUG: Appel de l'outil avec desc='{image_description}' et noms={person_names}")
    return retrieve_images_by_persons_names_and_image_description(
        image_description=image_description,
        client=client, # On injecte le client global ici
        person_names=person_names
    )

# --- Initialisation de l'Agent Gemini ---

# Chargement du prompt système (ton fichier prompts.txt)
system_instruction = Path("prompts.txt").read_text(encoding="utf-8")
# On nettoie un peu le template car Gemini n'a pas besoin des accolades {QDRANT_URL} dans le texte
system_instruction = system_instruction.replace("{QDRANT_URL}", qdrant_url).replace("{QDRANT_KEY}", "Locally Managed")

# Création du modèle avec l'outil
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[search_tool], # On lui donne notre wrapper
    system_instruction=system_instruction
)

# On lance une session de chat (automatique function calling activé)
chat_session = model.start_chat(enable_automatic_function_calling=True)

# Ajoute ces imports en haut de api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import shutil
from pathlib import Path

# ... (Le début de ton fichier reste pareil : setup, load_dotenv, etc.) ...

# Crée un dossier pour stocker les images reçues s'il n'existe pas
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """Sauvegarde le fichier uploadé sur le disque et retourne son chemin"""
    try:
        # On définit le chemin de destination
        dest_path = UPLOAD_DIR / upload_file.filename
        
        # On copie les octets du fichier reçu vers le disque
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return str(dest_path.absolute()) # On retourne le chemin absolu
    finally:
        upload_file.file.close()

# --- NOUVELLES ROUTES ---

@app.post("/add_image")
# Note : On n'utilise plus ImageRequest ici, mais UploadFile
def add_image_endpoint(file: UploadFile = File(...)):
    try:
        # 1. On sauvegarde l'image physiquement
        saved_path = save_upload_file(file)
        
        # 2. On traite l'image avec le chemin local
        nbr, names, scores, added = add_image_with_person_name_from_path(saved_path, client)
        if added:
            return {"status": "success", "detected_people": names, "filename": file.filename,
                    "numbers_of_detected_peoples":nbr,
                    "scores":scores}    
        else:
            return
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_face")
def add_face_endpoint(
    file: UploadFile = File(...), 
    person_name: str = Form(...) # Form(...) permet de récupérer du texte en plus du fichier
):
    try:
        # 1. On sauvegarde l'image
        saved_path = save_upload_file(file)
        
        # 2. Normalisation du nom
        normalized_name = person_name.lower().strip()
        
        # 3. Apprentissage
        success = add_face_with_person_name_from_path(
            saved_path, 
            normalized_name, 
            client
        )
        
        if success:
            return {"status": "success", "message": f"Visage de {normalized_name} ajouté."}
        else:
            raise HTTPException(status_code=400, detail="Échec de l'ajout (pas de visage ou plusieurs).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... (La route /ask_agent reste inchangée) ...