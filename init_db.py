from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

load_dotenv()

# Connexion
url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=url)

print(f"Initialisation de la base de données sur {url}...")

# 1. Création de la collection IMAGES (CLIP Large = 768 dimensions)
client.recreate_collection(
    collection_name="images_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)
print("✅ Collection 'images_collection' créée.")

# 2. Création de la collection VISAGES (Facenet512 = 512 dimensions)
client.recreate_collection(
    collection_name="faces",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
)
print("✅ Collection 'faces' créée.")