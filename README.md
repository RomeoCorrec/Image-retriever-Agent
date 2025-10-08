# Image Retriever Agent

An intelligent AI-powered image retrieval system that uses CLIP embeddings and Qdrant vector database to find images based on natural language descriptions and person names.

## Overview

This project implements a multimodal retrieval agent that combines:
- **CLIP (Contrastive Language-Image Pre-training)** for encoding images and text into the same vector space
- **Qdrant** vector database for efficient similarity search
- **SmolAgents** framework with Ollama for intelligent query processing
- **DeepFace** with Facenet512 for face detection and person identification
- Natural language understanding to extract person names and visual descriptions from user queries

## Features

- 🔍 **Natural Language Search**: Describe images in plain language (English or French)
- 👤 **Person-based Filtering**: Search for images containing specific people
- 🤖 **Intelligent Query Processing**: AI agent automatically parses queries to extract names and descriptions
- 🎯 **Semantic Similarity**: Uses CLIP embeddings for accurate image-text matching
- 📊 **Confidence Scoring**: Returns results with similarity scores
- 🖼️ **Visual Display**: Optional image visualization with matplotlib
- 🧑 **Face Recognition**: Automatic face detection and recognition using DeepFace with Facenet512

## How It Works

1. **User Query**: The user provides a natural language query (e.g., "Find images of Roméo in China")
2. **Query Processing**: The AI agent:
   - Extracts person names from the query
   - Converts names to lowercase
   - Generates a clean description without person names
3. **Embedding**: The description is encoded using CLIP's text encoder
4. **Vector Search**: Qdrant searches for similar image embeddings
5. **Filtering**: Results are filtered by person names (if specified)
6. **Results**: Top matching images are returned with confidence scores

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (default: `http://127.0.0.1:11434`)
- Qdrant instance (local or cloud)
- Hugging Face account (for CLIP model access)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/RomeoCorrec/Image-retriever-Agent.git
cd Image-retriever-Agent
```

2. **Install dependencies**:
```bash
pip install smolagents
pip install qdrant-client
pip install transformers
pip install torch
pip install pillow
pip install matplotlib
pip install python-dotenv
pip install huggingface-hub
pip install deepface
```

3. **Install and start Ollama**:
```bash
# Download and install Ollama from https://ollama.ai
# Pull the required model
ollama pull qwen2:7b
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
├── main.py              # Entry point - initializes agent and handles user interaction
├── tools.py             # Agent tools for Qdrant connection and image retrieval
├── fonctions.py         # Utility functions for CLIP, Ollama, and prompt loading
├── prompts.txt          # Agent instruction template
├── .env                 # Environment variables (not in repo)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Core Components

### main.py
- Loads environment variables
- Initializes Hugging Face authentication
- Sets up Ollama model
- Creates the CodeAgent with tools
- Handles user input and task execution

### tools.py
- `connect_to_qdrant()`: Establishes connection to Qdrant database
- `retrieve_images_by_persons_names_and_image_description()`: Performs multimodal search with optional person filtering

### fonctions.py
- `login_huggingface()`: Authenticates with Hugging Face
- `load_clip_model_processor()`: Loads CLIP model and processor
- `load_ollama_model()`: Initializes Ollama LLM
- `embed_image()`: Generates CLIP embeddings for images
- `embed_text_single()`: Generates CLIP embeddings for text
- `load_filled_prompt()`: Loads and fills prompt template
- `add_image_with_person_name_from_path()`: Adds images to the database with automatic face detection and person name association
- `add_face_with_person_name_from_path()`: Registers a single face with a given person name in the faces collection

### prompts.txt
System prompt template that instructs the agent on:
- How to parse user queries
- Extract person names
- Generate clean descriptions
- Use tools correctly
- Handle edge cases

## Technical Details

### Models Used
- **CLIP**: `openai/clip-vit-large-patch14` for multimodal embeddings
- **LLM**: `qwen2:7b` via Ollama for query understanding
- **Face Recognition**: `Facenet512` via DeepFace for face detection and recognition

### Vector Database
- **Collections**: 
  - `images_collection`: Stores image embeddings with associated person names and file paths
  - `faces`: Stores face embeddings for person identification
- **Similarity Threshold**: 0.20 (configurable in `tools.py`)
- **Metadata**: Stores image paths and person names

### Dependencies
- `smolagents`: Agent framework
- `qdrant-client`: Vector database client
- `transformers`: CLIP model
- `torch`: PyTorch for CLIP
- `pillow`: Image processing
- `matplotlib`: Image visualization
- `python-dotenv`: Environment variable management
- `huggingface-hub`: Model access
- `deepface`: Face detection and recognition framework

## Customization

### Adjust Similarity Threshold
In `tools.py`, line 69:
```python
if point.score > 0.20:  # Change this value
```

### Change Number of Results
When calling the agent, modify `top_k` parameter (default: 3)

### Use Different Models
In `fonctions.py`:
- CLIP model: Change `model_id` parameter in embedding functions
- Ollama model: Modify `load_ollama_model()` parameters
- Face recognition model: Modify `model_name` in `add_image_with_person_name_from_path()` and `add_face_with_person_name_from_path()` (options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace)

### Database Indexing Functions

The repository includes utility functions for indexing images with face detection:

#### `add_image_with_person_name_from_path(image_path, client, ...)`
Automatically detects faces in an image, identifies persons by comparing against the `faces` collection, and adds the image to the `images_collection` with associated person names.

**Features:**
- Detects up to 10 faces per image
- Matches detected faces against known persons in the database
- Prevents duplicate image entries
- Returns detected person names and confidence scores

#### `add_face_with_person_name_from_path(image_path, person_name, client, ...)`
Registers a reference face for a person in the `faces` collection.

**Features:**
- Requires exactly one face in the image
- Stores face embedding with associated person name
- Used to build the face recognition reference database

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check the API base URL in `fonctions.py`

2. **Hugging Face Authentication Failed**
   - Verify your token in `.env`
   - Ensure you have accepted CLIP model terms on Hugging Face

3. **Qdrant Connection Error**
   - Verify URL and API key in `.env`
   - Check Qdrant instance is running and accessible

4. **No Images Found**
   - Verify images are indexed in Qdrant
   - Try adjusting the similarity threshold
   - Rephrase your query

5. **DeepFace/Face Detection Issues**
   - Ensure OpenCV is properly installed
   - Check that face images are clear and properly oriented
   - Verify the detector backend (opencv, mtcnn, mediapipe) is available

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source. Please check with the repository owner for specific license terms.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the multimodal embedding model
- [Qdrant](https://qdrant.tech/) for the vector database
- [SmolAgents](https://github.com/huggingface/smolagents) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [DeepFace](https://github.com/serengil/deepface) for face recognition capabilities

## Contact

For questions or support, please open an issue on the GitHub repository.
