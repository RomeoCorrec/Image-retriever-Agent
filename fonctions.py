from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from smolagents import LiteLLMModel
from huggingface_hub import login
from deepface import DeepFace
import uuid
from qdrant_client.http import models

from pathlib import Path
# Fonction pour se connecter à Hugging Face
def login_huggingface(token):
    login(token)


# Fonction pour se charger le modèle CLIP et le processeur
def load_clip_model_processor(model_id = "openai/clip-vit-large-patch14"):
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


# Chargement du modèle Ollama via SmolAgents
def load_ollama_model(model_id="ollama_chat/qwen2:7b", api_base="http://127.0.0.1:11434", num_ctx=4096):
    model = LiteLLMModel(
            model_id=model_id,
            api_base=api_base,
            num_ctx=num_ctx,
        )
    
    return model


# Fonction pour encoder une image
def embed_image(path, model_id = "openai/clip-vit-large-patch14"):
    clip_model, processor = load_clip_model_processor(model_id)
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)  # normalisation
    return vec.cpu().numpy()[0]


# Fonction pour encoder un texte
def embed_text_single(text, model_id = "openai/clip-vit-large-patch14"):
    clip_model, processor = load_clip_model_processor(model_id)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        v = clip_model.get_text_features(**inputs)
    v = v / v.norm(p=2, dim=-1, keepdim=True)
    return v.cpu().numpy()[0].tolist()


def load_filled_prompt(template_path: str, QDRANT_URL: str, QDRANT_KEY: str, task: str) -> str:
    """
    Charge un template de prompt et remplace les variables entre accolades.
    Exemple : load_filled_prompt('prompts/qdrant_react_agent_prompt.txt', task="find images of romeo in China")
    """
    template = Path(template_path).read_text(encoding="utf-8")
    return template.format(QDRANT_URL=QDRANT_URL, QDRANT_KEY=QDRANT_KEY, task=task)


# Function pour ajouter une image avec les noms des personnes détectées
def add_image_with_person_name_from_path(image_path, client, add_faces_vector=True, images_collection="images_collection", faces_collection="faces"):
    embeddings_query = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        detector_backend="yolov8",   # ou mtcnn, mediapipe, opencv
        align=True,
    )

    names = []
    scores = []
    for emb in embeddings_query:
        embeddings_query_vector = emb['embedding']
        hits = client.query_points(
            collection_name=faces_collection,
            query=embeddings_query_vector,
            limit=1,
            with_payload=True,
        )
        
        for point in hits.points:
            name = point.payload.get("name")
            names.append(name)
            scores.append(point.score)

            if add_faces_vector and point.score < 0.8:  # seuil de similarité pour ajouter un nouveau visage
                # Ajouter le vecteur du visage dans la collection de visages
                uid = uuid.uuid4()
                client.upsert(
                    collection_name=faces_collection,
                    points=[
                        models.PointStruct(
                            id=str(uid),
                            vector=embeddings_query_vector,
                            payload={
                                "name": name,
                                "image_path": image_path
                            }
                        )
                    ]
                )
    
    # Vérification si l'image est déjà dans la collection d'images
    existing = client.scroll(
        collection_name=images_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="path",
                    match=models.MatchValue(value=image_path)
                )
            ]
        ),
        limit=1
    )
    if len(existing[0]) > 0:
        print(f"Image déjà présente : {image_path}")
        return names, scores  # on quitte la fonction
        
    # Ajout de l'image dans la collection d'images
    uid = uuid.uuid4()
    vec = embed_image(image_path)

    client.upsert(
        collection_name=images_collection,
        points=[
            models.PointStruct(
                id=str(uid),
                vector=vec.tolist(),
                payload={
                    "person": names,
                    "path": image_path
                }
            )
        ]
    )
    return names, scores

# Fonction qui ajoute uniquement le vecteurs d'un seul visage avec le nom fournis, a partir d'une image
def add_face_with_person_name_from_path(image_path, person_name, client, faces_collection="faces"):

    embeddings_query = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        detector_backend="yolov8",   # ou mtcnn, mediapipe, opencv
        align=True,
    )

    if len(embeddings_query) == 0:
        print(f"Aucun visage détecté dans l'image : {image_path}")
        return False
    elif len(embeddings_query) > 1:
        print(f"Plusieurs visages détectés dans l'image : {image_path}. Veuillez fournir une image avec un seul visage.")
        return False

    emb = embeddings_query[0]
    embeddings_query_vector = emb['embedding']
    
    # # Vérification si le visage est déjà dans la collection de visages
    # existing = client.scroll(
    #     collection_name=faces_collection,
    #     scroll_filter=models.Filter(
    #         must=[
    #             models.FieldCondition(
    #                 key="name",
    #                 match=models.MatchValue(value=person_name)
    #             )
    #         ]
    #     ),
    #     limit=1
    # )
    # if len(existing[0]) > 0:
    #     print(f"Visage déjà présent pour la personne : {person_name}")
    #     return False  # on quitte la fonction
        
    # Ajout du visage dans la collection de visages
    uid = uuid.uuid4()

    done = client.upsert(
        collection_name=faces_collection,
        points=[
            models.PointStruct(
                id=str(uid),
                vector=embeddings_query_vector,
                payload={
                    "name": person_name,
                    "image_path": image_path
                }
            )
        ]
    )
    return done