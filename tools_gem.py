from qdrant_client import QdrantClient
from qdrant_client.http import models
from fonctions import embed_text_single

# Plus d'import de smolagents !

def retrieve_images_by_persons_names_and_image_description(
    image_description: str,
    client: QdrantClient, # Gemini ignorera cet argument s'il n'est pas dans la docstring, on gérera l'injection manuellement
    person_names: list = None,
    top_k: int = 3
) -> dict: # On peut renvoyer un dict, Gemini le lira très bien
    """
    Récupère les images en fonction des noms de personnes et d'une description.
    
    Args:
        image_description: Description visuelle de la scène (ex: "à la plage").
        person_names: Liste des noms de personnes à filtrer (ex: ["romeo"]).
        top_k: Nombre max de résultats.
    """
    
    # ... (Le reste du code de la fonction reste IDENTIQUE à ce que tu avais) ...
    
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

    try:
        # Note: assure-toi que embed_text_single est bien importé
        vec = embed_text_single(image_description)
        hits = client.query_points(
            collection_name="images_collection",
            query=vec,
            limit=top_k,
            with_payload=True,
            query_filter=filter
        )
    except Exception as e:
        return {"error": str(e)}

    results = {}
    for point in hits.points:
        if point.score > 0.22: 
            results[point.payload.get("path")] = point.score
    
    if not results:
        return {"message": "Aucune image trouvée."}
        
    return results