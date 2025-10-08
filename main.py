from smolagents import CodeAgent, FinalAnswerTool
from dotenv import load_dotenv
import os

from fonctions import *
from tools import *


def main():

    load_dotenv(dotenv_path=".env")
    # Connexion à Hugging Face
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login_huggingface(hf_token)

    # Chargement du modèle Ollama via SmolAgents
    ollama_model = load_ollama_model()

    # Connexion à Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_KEY")

    # Définition des outils pour l'agent
    final_answer_tool = FinalAnswerTool()
    
    tools = [
        final_answer_tool,
        retrieve_images_by_persons_names_and_image_description,
        connect_to_qdrant
    ]

    # Création de l'agent
    agent = CodeAgent(
        tools=tools,
        model=ollama_model,
        additional_authorized_imports=["qdrant_client"],  # ✅ autorise l’import
        verbosity_level=1,
    )

    # Demander a l'utilisateur une tâche
    task = input("Quelle image souhaitez-vous trouver ? ")

    prompt = load_filled_prompt(
        template_path='prompts.txt',
        QDRANT_URL=qdrant_url,
        QDRANT_KEY=qdrant_api_key,
        task=task
    )

    # Exécution de l'agent avec la tâche fournie par l'utilisateur
    agent.run(task=prompt)

if __name__ == "__main__":
    main()