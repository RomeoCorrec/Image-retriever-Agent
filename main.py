from smolagents import CodeAgent, FinalAnswerTool
from dotenv import load_dotenv
import os
import argparse

from fonctions import *
from tools import *


def main():

    parser = argparse.ArgumentParser(description="Une CLI simple d'exemple")
    parser.add_argument("--task", type=str, choices=["add_image", "add_face", "image_retrieval"], required=True, help="La tâche que l'agent doit accomplir")
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")
    # Connexion à Hugging Face
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login_huggingface(hf_token)

    # Chargement du modèle Ollama via SmolAgents
    ollama_model = load_ollama_model()

    # Connexion à Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_KEY")

    if args.task == "image_retrieval":
        print("Mode image retrieval")

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

    elif args.task == "add_image":
        print("Mode add image")
        # Connexion à Qdrant
        client = connect_to_qdrant(qdrant_url, qdrant_api_key)

        # Demander a l'utilisateur le chemin de l'image
        image_path = input("Chemin de l'image à ajouter : ")

        # Ajouter l'image avec les noms des personnes détectées
        add_image_with_person_name_from_path(image_path, client)

    elif args.task == "add_face":
        print("Mode add face")
        # Connexion à Qdrant
        client = connect_to_qdrant(qdrant_url, qdrant_api_key)

        # Demander a l'utilisateur le chemin de l'image
        image_path = input("Chemin de l'image de la personne à ajouter : ")
        person_name = input("Nom de la personne : ")

        # Ajouter le visage avec le nom de la personne
        add_face_with_person_name_from_path(image_path, person_name, client)

if __name__ == "__main__":
    main()