# Image Retriever Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Open%20Source-green.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/vector_db-Qdrant-red.svg)](https://qdrant.tech/)
[![CLIP](https://img.shields.io/badge/model-CLIP-orange.svg)](https://github.com/openai/CLIP)

An intelligent AI-powered image retrieval system that uses CLIP embeddings and Qdrant vector database to find images based on natural language descriptions and person names.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Technical Details](#technical-details)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

This project implements a multimodal retrieval agent that combines:
- **CLIP (Contrastive Language-Image Pre-training)** for encoding images and text into the same vector space
- **Qdrant** vector database for efficient similarity search
- **SmolAgents** framework with Ollama for intelligent query processing
- Natural language understanding to extract person names and visual descriptions from user queries

## Features

- 🔍 **Natural Language Search**: Describe images in plain language (English or French)
- 👤 **Person-based Filtering**: Search for images containing specific people
- 🤖 **Intelligent Query Processing**: AI agent automatically parses queries to extract names and descriptions
- 🎯 **Semantic Similarity**: Uses CLIP embeddings for accurate image-text matching
- 📊 **Confidence Scoring**: Returns results with similarity scores
- 🖼️ **Visual Display**: Optional image visualization with matplotlib

## How It Works

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  User Input: "Find images of Roméo in China"                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent (Ollama LLM) - Query Processing                          │
│  • Extracts person names: ["roméo"]                             │
│  • Normalizes to lowercase: ["romeo"]                           │
│  • Generates description: "person in China"                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool: connect_to_qdrant()                                      │
│  • Connects to vector database                                  │
│  • Returns authenticated client                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLIP Text Encoder                                              │
│  • Input: "person in China"                                     │
│  • Output: 768-dimensional vector                               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Qdrant Vector Search                                           │
│  • Searches images_collection                                   │
│  • Applies person filter: person = "romeo"                      │
│  • Finds top-K similar vectors (cosine similarity)              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Filter Results (score > 0.20)                                  │
│  • Ranks by similarity score                                    │
│  • Returns image paths + scores                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output to User                                                 │
│  • /path/to/romeo_china_1.jpg (0.87)                            │
│  • /path/to/romeo_china_2.jpg (0.73)                            │
│  • /path/to/romeo_china_3.jpg (0.65)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **User Query**: The user provides a natural language query (e.g., "Find images of Roméo in China")

2. **Query Processing**: The AI agent (Ollama LLM):
   - Extracts person names from the query
   - Converts names to lowercase for database matching
   - Generates a clean description without person names
   - Validates and structures the query parameters

3. **Database Connection**: Agent connects to Qdrant using provided credentials

4. **Text Embedding**: The clean description is encoded using CLIP's text encoder into a 768-dimensional vector

5. **Vector Search**: Qdrant performs similarity search:
   - Compares query vector against all image vectors
   - Uses cosine similarity metric
   - Applies HNSW algorithm for fast approximate search

6. **Person Filtering** (if applicable): 
   - Filters results by person names in metadata
   - Uses exact matching on lowercase names

7. **Threshold Filtering**: Only returns results with similarity score > 0.20

8. **Results**: Top matching images are returned with confidence scores

### Embedding Process

**For Images** (during indexing):
```
Image File → PIL.Image → CLIP Image Encoder → 768-d vector → Qdrant
```

**For Text** (during search):
```
Text Query → CLIP Text Encoder → 768-d vector → Qdrant Search
```

**For Faces** (during indexing, optional):
```
Image File → DeepFace → Face Detection → FaceNet512 → 512-d vector → Qdrant
```

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (default: `http://127.0.0.1:11434`)
- Qdrant instance (local or cloud)
- Hugging Face account (for CLIP model access)

## Installation

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/RomeoCorrec/Image-retriever-Agent.git
cd Image-retriever-Agent
```

2. **Install Python dependencies**:
```bash
pip install smolagents qdrant-client transformers torch pillow matplotlib python-dotenv huggingface-hub deepface
```

Or create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install smolagents qdrant-client transformers torch pillow matplotlib python-dotenv huggingface-hub deepface
```

3. **Install and start Ollama**:
```bash
# Download and install Ollama from https://ollama.ai
# Pull the required model
ollama pull qwen2:7b
```

4. **Start Ollama server** (if not already running):
```bash
ollama serve
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Hugging Face Token (required for CLIP model)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_KEY=your_qdrant_api_key_here
```

### Getting API Keys:

- **Hugging Face Token**: Sign up at [huggingface.co](https://huggingface.co) and create a token in Settings > Access Tokens
- **Qdrant**: 
  - For cloud: Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
  - For local: Run `docker run -p 6333:6333 qdrant/qdrant`

## Data Preparation

Before you can search for images, you need to index them in Qdrant. This involves:

### 1. Setting Up Qdrant Collections

Create two collections in your Qdrant instance:

**images_collection**: Stores image embeddings with metadata
- Vector size: 768 (CLIP ViT-Large/14 dimension)
- Distance metric: Cosine similarity

**faces** (optional): Stores face embeddings for person identification
- Vector size: 512 (FaceNet512 dimension)
- Distance metric: Cosine similarity

### 2. Indexing Images

Use the utility functions in `fonctions.py` to index your images:

```python
from fonctions import add_image_with_person_name_from_path, add_face_with_person_name_from_path
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_API_KEY")

# Index an image with automatic face detection
image_path = "/path/to/your/image.jpg"
detected_names, scores = add_image_with_person_name_from_path(
    image_path=image_path,
    client=client,
    add_faces_vector=True
)

# Or add a face reference for a known person
add_face_with_person_name_from_path(
    image_path="/path/to/reference/face.jpg",
    person_name="romeo",  # Must be lowercase
    client=client
)
```

### 3. Image Metadata Format

Each image in Qdrant is stored with:
- **vector**: 768-dimensional CLIP embedding
- **payload**: 
  - `path`: Absolute path to the image file
  - `person`: List of person names detected in the image (lowercase)

**Example payload:**
```json
{
  "path": "/home/user/photos/vacation.jpg",
  "person": ["romeo", "alice"]
}
```

## Usage

### Running the Application

```bash
python main.py
```

The application will prompt you for a query:
```
Quelle image souhaitez-vous trouver ?
```

### Example Queries

```text
# Search by person and description
Find images of Roméo in China

# Search by description only
Images of a sunset over mountains

# Multiple people
Photos avec Alice et Bob au restaurant

# Complex description
Pictures of a person standing near the Eiffel Tower at night
```

### Expected Output

When you run a query, the agent will:
1. Parse the query and extract person names
2. Connect to Qdrant
3. Search for matching images
4. Return results with confidence scores

**Example output:**
```
Final answer: 
Found images:
- /home/user/photos/romeo_china_2023.jpg (Score: 0.87)
- /home/user/photos/beijing_trip.jpg (Score: 0.73)
- /home/user/photos/great_wall.jpg (Score: 0.65)
```

If no images are found with sufficient similarity (score > 0.20), the agent will inform you:
```
Aucune image trouvée avec un score suffisant.
Final answer: No matching images found. Try rephrasing your query or adjusting the similarity threshold.
```

### Query Format

The agent intelligently processes queries to:
- Identify person names (case-insensitive)
- Convert names to lowercase for database filtering
- Extract visual descriptions
- Handle both English and French queries

## Project Structure

```
Image-retriever-Agent/
│
├── main.py                 # Entry point - initializes agent and handles user interaction
├── tools.py                # Agent tools for Qdrant connection and image retrieval
├── fonctions.py            # Utility functions for CLIP, Ollama, and prompt loading
├── prompts.txt             # Agent instruction template
├── .env                    # Environment variables (not in repo)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| **main.py** | Application entry point | `main()` - Orchestrates the entire workflow |
| **tools.py** | SmolAgents tools | `connect_to_qdrant()`, `retrieve_images_by_persons_names_and_image_description()` |
| **fonctions.py** | Utility functions | Model loaders, embedding functions, indexing helpers |
| **prompts.txt** | Agent instructions | Template with query parsing logic |
| **.env** | Configuration | API keys and environment variables |

### Data Flow

```
User → main.py → Agent (Ollama) → tools.py → fonctions.py → Qdrant
                                       ↓
                                  CLIP Model
```

## Core Components

### main.py
**Purpose**: Entry point for the application

**Key Functions**:
- Loads environment variables from `.env`
- Initializes Hugging Face authentication
- Sets up Ollama model for query processing
- Creates the CodeAgent with available tools
- Handles user input and task execution loop

**Agent Configuration**:
```python
agent = CodeAgent(
    tools=[final_answer_tool, retrieve_images_by_persons_names_and_image_description, connect_to_qdrant],
    model=ollama_model,
    additional_authorized_imports=["qdrant_client"],
    verbosity_level=1
)
```

### tools.py
**Purpose**: Defines agent tools for database interaction

**Tools**:

1. **`connect_to_qdrant(url: str, api_key: str) -> QdrantClient`**
   - Establishes connection to Qdrant vector database
   - Returns authenticated client instance
   - Used at the start of every search operation

2. **`retrieve_images_by_persons_names_and_image_description(...) -> dict`**
   - **Parameters**:
     - `image_description` (str): Clean visual description without person names
     - `client` (QdrantClient): Connected Qdrant client
     - `person_names` (list, optional): Lowercase person names to filter by
     - `top_k` (int, default=3): Maximum number of results
     - `plot_found_images` (bool, default=False): Display images with matplotlib
   - **Returns**: Dictionary mapping image paths to similarity scores
   - **Process**:
     1. Encodes description using CLIP text encoder
     2. Searches Qdrant for similar image vectors
     3. Filters by person names if provided
     4. Returns results above 0.20 similarity threshold

### fonctions.py
**Purpose**: Utility functions for model loading and data processing

**Functions**:

- **`login_huggingface(token: str)`**: Authenticates with Hugging Face Hub
- **`load_clip_model_processor(model_id: str)`**: Loads CLIP model and processor
  - Default: `"openai/clip-vit-large-patch14"`
  - Returns: (model, processor) tuple
- **`load_ollama_model(model_id: str, api_base: str, num_ctx: int)`**: Initializes Ollama LLM
  - Default model: `"ollama_chat/qwen2:7b"`
  - Default API: `"http://127.0.0.1:11434"`
- **`embed_image(path: str, model_id: str) -> np.ndarray`**: Generates 768-d CLIP embedding for image
- **`embed_text_single(text: str, model_id: str) -> list`**: Generates 768-d CLIP embedding for text
- **`load_filled_prompt(template_path: str, ...) -> str`**: Loads and fills prompt template with variables
- **`add_image_with_person_name_from_path(...)`**: Indexes image with automatic face detection
- **`add_face_with_person_name_from_path(...)`**: Adds face reference for person identification

### prompts.txt
**Purpose**: System prompt template for the agent

**Instructions included**:
- How to parse user queries
- Extract and normalize person names (to lowercase)
- Generate clean visual descriptions
- Use tools in correct sequence
- Handle edge cases (no results, multiple people, etc.)
- Retry logic for failed searches

**Template variables**:
- `{QDRANT_URL}`: Qdrant instance URL
- `{QDRANT_KEY}`: Qdrant API key
- `{task}`: User's search query

## Technical Details

### Architecture

```
User Query → Agent (Ollama) → Query Parser
                                    ↓
                         [Person Names] + [Description]
                                    ↓
                         CLIP Text Encoder (768-d vector)
                                    ↓
                         Qdrant Vector Search
                                    ↓
                         Filter by Person Names (optional)
                                    ↓
                         Top-K Results (score > 0.20)
                                    ↓
                         Image Paths + Scores → User
```

### Models Used
- **CLIP**: `openai/clip-vit-large-patch14` for multimodal embeddings
  - Image encoder: Vision Transformer (ViT-Large/14)
  - Text encoder: Transformer
  - Embedding dimension: 768
  - Normalization: L2 normalized vectors
- **LLM**: `qwen2:7b` via Ollama for query understanding
  - Context window: 4096 tokens
  - Used for intelligent query parsing and name extraction
- **Face Recognition** (optional): `Facenet512` via DeepFace
  - Embedding dimension: 512
  - Used for person identification in images

### Vector Database
- **Collection**: `images_collection`
- **Similarity Metric**: Cosine similarity
- **Similarity Threshold**: 0.20 (configurable in `tools.py`)
- **Metadata**: Stores image paths and person names
- **Face Collection**: `faces` (optional, for person identification)

### Performance Characteristics

**Search Speed**:
- Query encoding: ~100-200ms (CLIP text encoding)
- Vector search: <10ms for collections up to 100K images (with Qdrant)
- Total latency: ~200-500ms per query (excluding LLM processing)

**Accuracy**:
- CLIP embeddings enable semantic understanding beyond keyword matching
- Similarity threshold of 0.20 balances precision and recall
- Person filtering uses exact matching on lowercase names

**Scalability**:
- Supports millions of images with Qdrant's HNSW index
- Memory usage: ~3KB per indexed image (768 float32 + metadata)
- Recommended: Use Qdrant Cloud or dedicated server for >10K images

### Dependencies
- `smolagents`: Agent framework for tool orchestration
- `qdrant-client`: Vector database client
- `transformers`: CLIP model from Hugging Face
- `torch`: PyTorch for CLIP inference
- `pillow`: Image loading and processing
- `matplotlib`: Image visualization
- `python-dotenv`: Environment variable management
- `huggingface-hub`: Model access and authentication
- `deepface`: Face recognition (optional, for person indexing)

## Customization

### Adjust Similarity Threshold
In `tools.py`, line 69:
```python
if point.score > 0.20:  # Change this value
```
- **Lower values** (e.g., 0.15): More results, lower precision
- **Higher values** (e.g., 0.30): Fewer results, higher precision

### Change Number of Results
When calling the agent, modify `top_k` parameter (default: 3)

In `tools.py`, update the function call:
```python
retrieve_images_by_persons_names_and_image_description(
    image_description="...",
    client=client,
    person_names=["alice"],
    top_k=10  # Return up to 10 results
)
```

### Use Different Models

#### CLIP Model
In `fonctions.py`, change the model in embedding functions:
```python
# Available CLIP models:
# - openai/clip-vit-base-patch32 (smaller, faster)
# - openai/clip-vit-large-patch14 (default, best quality)
# - openai/clip-vit-large-patch14-336 (higher resolution)

def embed_image(path, model_id="openai/clip-vit-base-patch32"):
    # ... your code
```

#### Ollama Model
In `fonctions.py`, modify `load_ollama_model()`:
```python
def load_ollama_model(
    model_id="ollama_chat/llama2",  # Try: llama2, mistral, mixtral
    api_base="http://127.0.0.1:11434",
    num_ctx=4096
):
    # ... your code
```

### Enable Image Visualization

Set `plot_found_images=True` in the retrieval call to display results:
```python
retrieve_images_by_persons_names_and_image_description(
    image_description="...",
    client=client,
    plot_found_images=True  # Shows images with matplotlib
)
```

### Advanced: Custom Filters

Modify the filter logic in `tools.py` to add custom metadata filtering:
```python
# Example: Filter by date range, location, tags, etc.
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="person",
            match=models.MatchValue(value=name)
        ) for name in person_names
    ] + [
        models.FieldCondition(
            key="location",  # Add custom metadata
            match=models.MatchValue(value="Paris")
        )
    ]
)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Connection refused to http://127.0.0.1:11434
   ```
   **Solution**:
   - Ensure Ollama is running: `ollama serve`
   - Check the API base URL in `fonctions.py` matches your Ollama instance
   - Verify the model is installed: `ollama list`
   - If using a different host: Update `api_base` in `load_ollama_model()`

2. **Hugging Face Authentication Failed**
   ```
   Error: Invalid or missing Hugging Face token
   ```
   **Solution**:
   - Verify your token in `.env` is correct
   - Ensure you have accepted CLIP model terms on [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)
   - Check token permissions (needs read access)
   - Test token: `huggingface-cli whoami`

3. **Qdrant Connection Error**
   ```
   Error: Could not connect to Qdrant at <URL>
   ```
   **Solution**:
   - Verify URL and API key in `.env` are correct
   - Check Qdrant instance is running and accessible
   - For local Qdrant: Ensure Docker container is running
   - Test connection: `curl http://localhost:6333/health`
   - Check firewall rules for cloud instances

4. **No Images Found**
   ```
   Aucune image trouvée avec un score suffisant.
   ```
   **Solution**:
   - Verify images are indexed in Qdrant collection `images_collection`
   - Check collection exists: Use Qdrant dashboard or API
   - Try lowering the similarity threshold in `tools.py` (line 69)
   - Rephrase your query with more descriptive terms
   - Ensure person names are lowercase in the database

5. **CUDA/GPU Errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**:
   - CLIP runs on CPU by default (no CUDA required)
   - If forcing GPU: Reduce batch size or use smaller CLIP model
   - Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

6. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'smolagents'
   ```
   **Solution**:
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Ensure virtual environment is activated
   - Check Python version: `python --version` (requires 3.8+)

7. **DeepFace Face Detection Issues**
   ```
   Face could not be detected in the image
   ```
   **Solution**:
   - Ensure image has clear, visible faces
   - Try different detector backend: `opencv`, `mtcnn`, `mediapipe`
   - Check image quality and lighting
   - For indexing: Use images with single, frontal faces for best results

### Performance Issues

**Slow Query Processing**:
- First query is slow due to model loading (expected)
- Subsequent queries should be faster
- Consider keeping models in memory for production use

**High Memory Usage**:
- CLIP model requires ~1.5GB RAM
- Reduce by using smaller model: `clip-vit-base-patch32`
- For large collections: Use Qdrant Cloud with pagination

### Debug Mode

Enable verbose output in `main.py`:
```python
agent = CodeAgent(
    tools=tools,
    model=ollama_model,
    verbosity_level=2  # 0=quiet, 1=normal, 2=debug
)
```

### Getting Help

If you encounter issues not listed here:
1. Check the [GitHub Issues](https://github.com/RomeoCorrec/Image-retriever-Agent/issues)
2. Enable debug mode and share the logs
3. Verify all prerequisites are met
4. Test individual components (Qdrant, Ollama, CLIP) separately

## API Reference

### Agent Tools

#### `connect_to_qdrant(url: str, api_key: str) -> QdrantClient`

Establishes connection to Qdrant vector database.

**Parameters**:
- `url` (str): Qdrant instance URL (e.g., `"http://localhost:6333"` or cloud URL)
- `api_key` (str): API key for authentication

**Returns**: `QdrantClient` - Authenticated Qdrant client instance

**Example**:
```python
client = connect_to_qdrant(
    url="https://xyz.cloud.qdrant.io",
    api_key="your-api-key-here"
)
```

#### `retrieve_images_by_persons_names_and_image_description(...) -> dict`

Performs multimodal image search with optional person filtering.

**Parameters**:
- `image_description` (str, required): Visual description of the desired image
- `client` (QdrantClient, required): Connected Qdrant client
- `person_names` (list[str], optional): List of person names to filter by (lowercase)
- `top_k` (int, optional): Maximum number of results to return (default: 3)
- `plot_found_images` (bool, optional): Whether to display results (default: False)

**Returns**: `dict` - Mapping of image paths to similarity scores

**Example**:
```python
results = retrieve_images_by_persons_names_and_image_description(
    image_description="person standing on a beach at sunset",
    client=client,
    person_names=["alice", "bob"],
    top_k=5,
    plot_found_images=True
)
# Returns: {'/path/img1.jpg': 0.85, '/path/img2.jpg': 0.72, ...}
```

### Utility Functions

#### `embed_image(path: str, model_id: str = "openai/clip-vit-large-patch14") -> np.ndarray`

Generates CLIP embedding for an image.

**Parameters**:
- `path` (str): Path to image file
- `model_id` (str, optional): CLIP model identifier

**Returns**: `np.ndarray` - 768-dimensional normalized embedding vector

#### `embed_text_single(text: str, model_id: str = "openai/clip-vit-large-patch14") -> list`

Generates CLIP embedding for text.

**Parameters**:
- `text` (str): Text description
- `model_id` (str, optional): CLIP model identifier

**Returns**: `list` - 768-dimensional normalized embedding vector

#### `add_image_with_person_name_from_path(...)`

Indexes an image with automatic face detection and person identification.

**Parameters**:
- `image_path` (str): Path to image file
- `client` (QdrantClient): Connected Qdrant client
- `add_faces_vector` (bool, optional): Whether to add face embeddings (default: True)
- `images_collection` (str, optional): Collection name (default: "images_collection")
- `faces_collection` (str, optional): Face collection name (default: "faces")

**Returns**: `tuple[list, list]` - (detected_names, confidence_scores)

#### `add_face_with_person_name_from_path(...)`

Adds a face reference for person identification.

**Parameters**:
- `image_path` (str): Path to image with single face
- `person_name` (str): Person's name (will be converted to lowercase)
- `client` (QdrantClient): Connected Qdrant client
- `faces_collection` (str, optional): Collection name (default: "faces")

**Returns**: `bool` - True if successful, False otherwise

## FAQ

### General Questions

**Q: What types of images does this system work best with?**  
A: The system works well with:
- Photos of people in various settings
- Landscape and nature photography
- Indoor and outdoor scenes
- Objects and products

It performs best when images are clear, well-lit, and contain recognizable visual elements.

**Q: Can I search in languages other than English?**  
A: The agent accepts queries in English and French. However, CLIP performs best with English descriptions, so the agent translates and normalizes queries internally.

**Q: How accurate is the person detection?**  
A: Person filtering relies on face recognition (FaceNet512). Accuracy depends on:
- Face visibility and angle
- Image quality
- Number of reference faces indexed
- Lighting conditions

**Q: Do I need a GPU?**  
A: No, the system runs on CPU. However, a GPU can speed up CLIP inference for large batches of images.

### Technical Questions

**Q: How many images can I index?**  
A: Qdrant can handle millions of vectors. Practical limits depend on:
- Available memory (server-side)
- Response time requirements
- Qdrant configuration (HNSW parameters)

For collections >100K images, consider Qdrant Cloud or a dedicated server.

**Q: What's the difference between `images_collection` and `faces`?**  
A: 
- `images_collection`: Stores CLIP embeddings of full images (768-d)
- `faces`: Stores FaceNet embeddings of detected faces (512-d)

The `faces` collection enables person identification, which filters the `images_collection`.

**Q: Can I use this without Ollama?**  
A: The current implementation requires Ollama for query parsing. However, you could modify `main.py` to use a different LLM provider (OpenAI, Anthropic, etc.) by changing the `LiteLLMModel` configuration.

**Q: How do I backup my indexed images?**  
A: Qdrant supports snapshots:
```bash
# Create snapshot
curl -X POST 'http://localhost:6333/collections/images_collection/snapshots'

# List snapshots
curl 'http://localhost:6333/collections/images_collection/snapshots'

# Download snapshot
curl 'http://localhost:6333/collections/images_collection/snapshots/<snapshot-name>' -o backup.snapshot
```

**Q: Can I integrate this into a web application?**  
A: Yes! You can:
1. Expose the retrieval functions as a REST API (using Flask/FastAPI)
2. Use the agent tools directly in your backend
3. Return image URLs instead of file paths
4. Add authentication and rate limiting

Example FastAPI integration:
```python
from fastapi import FastAPI
from tools import retrieve_images_by_persons_names_and_image_description, connect_to_qdrant

app = FastAPI()

@app.get("/search")
def search_images(query: str, persons: str = None, limit: int = 3):
    client = connect_to_qdrant(url=QDRANT_URL, api_key=QDRANT_KEY)
    person_list = persons.split(",") if persons else None
    results = retrieve_images_by_persons_names_and_image_description(
        image_description=query,
        client=client,
        person_names=person_list,
        top_k=limit
    )
    return {"results": results}
```

### Usage Questions

**Q: How do I update an indexed image?**  
A: Delete the old entry and re-index:
```python
# Find the image point ID
hits = client.scroll(
    collection_name="images_collection",
    scroll_filter=models.Filter(
        must=[models.FieldCondition(
            key="path",
            match=models.MatchValue(value="/old/path.jpg")
        )]
    )
)

# Delete it
client.delete(
    collection_name="images_collection",
    points_selector=[hit.id for hit in hits[0]]
)

# Re-index with updated path/metadata
add_image_with_person_name_from_path(new_path, client)
```

**Q: Can I search for multiple criteria (e.g., "sunset AND beach")?**  
A: Yes! CLIP naturally handles complex descriptions. The more specific your query, the better the results:
```
"sunset over ocean beach with palm trees"
```

**Q: How do I prevent duplicate images?**  
A: The `add_image_with_person_name_from_path` function checks if an image path already exists before indexing.

## Limitations

- **Person names must be pre-indexed**: The system can only filter by people whose faces have been added to the `faces` collection
- **Text in images**: CLIP doesn't read text within images (e.g., signs, documents)
- **Abstract concepts**: Works best with concrete visual elements rather than abstract ideas
- **Similarity threshold**: Fixed at 0.20 by default; may need tuning for your dataset
- **Real-time indexing**: Large batches of images take time to process (CLIP + face detection)
- **Language**: While multilingual, performance is optimized for English descriptions

## Contributing

Contributions are welcome! We appreciate your help in making this project better.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: 
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed
4. **Test your changes**: Ensure everything works as expected
5. **Commit your changes**: `git commit -m "Add: description of your changes"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**: Describe your changes and their benefits

### Areas for Contribution

- **New features**: Additional search filters, batch processing, web UI
- **Optimizations**: Performance improvements, caching, model quantization
- **Documentation**: More examples, tutorials, translations
- **Bug fixes**: Check the [Issues](https://github.com/RomeoCorrec/Image-retriever-Agent/issues) page
- **Testing**: Unit tests, integration tests, test datasets

### Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings for functions
- Keep functions focused and modular

### Reporting Bugs

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Error messages and stack traces
- Relevant code snippets

## License

This project is open source. Please check with the repository owner for specific license terms.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the multimodal embedding model
- [Qdrant](https://qdrant.tech/) for the vector database
- [SmolAgents](https://github.com/huggingface/smolagents) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM inference

## Contact

For questions or support, please open an issue on the GitHub repository.
