from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from smolagents import LiteLLMModel
from huggingface_hub import login

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
