from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image, ImageOps
from smolagents import tool
import matplotlib.pyplot as plt

from fonctions import *

@tool
def connect_to_qdrant(url: str, api_key: str) -> QdrantClient:
    """
    Connect to Qdrant and return the client instance.

    Args:
        url (str): The URL of the Qdrant instance to connect to.
        api_key (str): The API key used for authentication with Qdrant.

    Returns:
        QdrantClient: An instance of the connected Qdrant client.
    """
    client = QdrantClient(url=url, api_key=api_key)
    return client


@tool
def retrieve_images_by_persons_names_and_image_description(
    image_description: str,
    client: QdrantClient,
    person_names: list = None,
    top_k: int = 3,
    plot_found_images: bool = False
) -> dict:
    """Récupère les images en fonction des noms de personnes et d'une description d'image.
    Args:
        image_description (str): Description textuelle de l'image recherchée.
        client (QdrantClient): Instance du client Qdrant.
        person_names (list, optional): Liste des noms de personnes à filtrer. Defaults to None.
        top_k (int, optional): Nombre maximum d'images à récupérer. Defaults to 3.
        plot_found_images (bool, optional): Si True, affiche les images trouvées. Defaults to False.
    Returns:
        dict: Dictionnaire avec les chemins des images trouvées et leurs scores.
    """

    if person_names is None or len(person_names) == 0:
        filter = None
    else:
        must_conditions = [
            models.FieldCondition(
                key="person",
                match=models.MatchValue(value=name)
            ) for name in person_names
        ]

        filter = models.Filter(must=must_conditions)

    # Encoder la description textuelle
    vec = embed_text_single(image_description, model_id="openai/clip-vit-large-patch14")
    hits = client.query_points(
        collection_name="images_collection",
        query=vec,
        limit=top_k,
        with_payload=True,
        query_filter=filter
    )

    results_paths = dict()

    for point in hits.points:
        if point.score > 0.20:  # seuil de confiance
            results_paths[point.payload.get("path")] = point.score
    
    if len(results_paths) == 0:
        print("Aucune image trouvée avec un score suffisant.")
        
    if plot_found_images and len(results_paths) > 0:
        fig, axes = plt.subplots(1, len(results_paths), figsize=(5 * len(results_paths), 5))
        if len(results_paths) == 1:
            axes = [axes]  # Assurer que axes est toujours une liste
        for ax, (path, score) in zip(axes, results_paths.items()):
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Score: {score:.2f}")
        plt.show()

    return results_paths