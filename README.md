# Image Retriever Agent

An AI-powered image search system that finds images using natural language descriptions and person names.

## What It Does

This tool allows you to search for images in your collection by simply describing what you're looking for in plain language. For example: "Find images of Roméo in China" or "Photos of a sunset over mountains".

## How It Works

1. You provide a natural language query (in English or French)
2. An AI agent extracts person names and visual descriptions from your query
3. The description is converted into a vector using CLIP (a multimodal AI model)
4. The system searches a Qdrant vector database for similar images
5. Results are filtered by person names (if specified) and returned with confidence scores

## Technologies Used

- **CLIP** (`openai/clip-vit-large-patch14`) - Converts images and text into comparable vectors
- **Qdrant** - Vector database for fast similarity search
- **SmolAgents** - Framework for building AI agents
- **Ollama** (`qwen2:7b`) - Local LLM for intelligent query processing
- **Python** - Core programming language with libraries: PyTorch, Transformers, Pillow

## Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- Qdrant instance (local or cloud)
- Hugging Face account

### Installation

```bash
# Clone the repository
git clone https://github.com/RomeoCorrec/Image-retriever-Agent.git
cd Image-retriever-Agent

# Install dependencies
pip install smolagents qdrant-client transformers torch pillow matplotlib python-dotenv huggingface-hub

# Install Ollama and pull the model
ollama pull qwen2:7b
```

### Configuration

Create a `.env` file:

```env
HUGGINGFACE_TOKEN=your_token_here
QDRANT_URL=your_qdrant_url
QDRANT_KEY=your_qdrant_key
```

### Usage

```bash
python main.py
```

Then enter your query when prompted, for example:
- "Find images of Roméo in China"
- "Images of a sunset over mountains"
- "Photos avec Alice et Bob au restaurant"

## Project Structure

```
├── main.py          # Entry point and agent initialization
├── tools.py         # Qdrant connection and image retrieval tools
├── fonctions.py     # Utility functions for CLIP, Ollama, and embeddings
├── prompts.txt      # Agent instruction template
└── .env             # Configuration (not in repo)
```

## License

Open source. Please check with the repository owner for specific license terms.
