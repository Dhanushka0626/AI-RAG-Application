# AI RAG Application 🤖🗺️

An intelligent AI-powered tour guide assistant that uses Retrieval-Augmented Generation (RAG) to provide real-time information about landmarks and tourist destinations. Upload an image of a landmark, and the AI will identify it and answer your questions with accurate, context-aware responses.

## 🎯 Overview

This application combines computer vision, semantic search, and large language models to create an intelligent tour guide. When you upload an image of a landmark:
1. **Image Classification**: The system identifies what landmark you're looking at
2. **Context Retrieval**: It retrieves relevant information about that landmark
3. **AI Response**: Uses an LLM to answer your questions based on verified information

Perfect for tourists, travelers, and educational purposes!

## ✨ Features

- **Image-Based Landmark Recognition**: Upload any landmark image to identify it
- **Intelligent Q&A**: Ask questions about the identified landmark
- **Context-Aware Responses**: Answers are grounded in curated knowledge base data
- **Multi-Source Data**: Knowledge base includes text descriptions and visual embeddings
- **Real-time Processing**: Instant responses with spinner feedback
- **Web Interface**: User-friendly Streamlit interface
- **Vector Database**: Uses ChromaDB for efficient semantic search

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Web interface and application framework
- **Python** - Programming language

### AI & ML Components
- **LangChain** - LLM orchestration and prompt management
- **LangChain-Ollama** - Local LLM integration
- **ChromaDB** - Vector database for embeddings and retrieval
- **Sentence-Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **Img2Vec-PyTorch** - Image embeddings using deep learning
- **PyTorch** - Deep learning framework

### Computer Vision
- **OpenCV** - Image processing
- **Pillow** - Image operations
- **TorchVision** - Vision utilities

### Utilities
- **NumPy** - Numerical operations

## 📦 Prerequisites

Before installation, ensure you have:
- Python 3.8 or higher
- pip (Python package manager)
- **Ollama** installed with Mistral model (for local LLM)
  - Download from: [ollama.ai](https://ollama.ai)
  - Pull model: `ollama pull mistral`
- At least 8GB RAM
- GPU (optional, but recommended for faster processing)

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/Dhanushka0626/AI-RAG-Application.git
cd AI-RAG-Application
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running with Mistral model:
```bash
ollama serve
```

5. Prepare your data directory structure:
```
AI-RAG-Application/
└── data/
    ├── landmark1/
    │   ├── image1.jpeg
    │   ├── image2.jpeg
    │   └── description.txt
    ├── landmark2/
    │   ├── image1.jpeg
    │   └── description.txt
    └── ...
```

## 🚀 Quick Start

### Step 1: Create Vector Databases

First, prepare your data in the `data/` directory with images and text descriptions, then create the vector databases:

```bash
python create_dbs.py
```

This will:
- Create image embeddings using Img2Vec
- Create text embeddings using Sentence-Transformers
- Store everything in ChromaDB for fast retrieval

### Step 2: Run the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### Step 3: Use the Application

1. Upload an image of a landmark
2. The system will classify the landmark
3. Ask questions about it (e.g., "When was it built?", "What's its historical significance?")
4. Receive AI-generated responses based on your knowledge base

## 📁 Project Structure

```
AI-RAG-Application/
├── main.py                 # Streamlit web interface
├── query_data.py          # Core RAG pipeline functions
├── create_dbs.py          # Database creation script
├── util.py                # Utility functions and embeddings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── data/                 # Input data directory
│   └── [landmark_categories]/
│       ├── *.jpeg        # Landmark images
│       └── *.txt         # Text descriptions
└── db/                   # ChromaDB vector database
    ├── chromadb.sqlite3  # Database file
    └── ...
```

## 🔄 How It Works

### Architecture

```
User Uploads Image
        ↓
Image Classification (Img2Vec)
        ↓
Identify Landmark Category
        ↓
Retrieve Similar Text Chunks (Semantic Search)
        ↓
Generate Response (Mistral LLM)
        ↓
Display Response to User
```

### Process Flow

#### 1. **Image Classification** (`classify_img`)
- Takes uploaded image and encodes it using Img2Vec
- Compares with image embeddings in ChromaDB
- Returns the landmark category

#### 2. **Context Retrieval** (`get_most_similar_chunks`)
- Takes user question and embeds it with Sentence-Transformers
- Searches ChromaDB for top-3 most relevant text chunks
- Returns relevant information about the landmark

#### 3. **Response Generation** (`create_response`)
- Uses retrieved context and user question
- Passes to Mistral LLM via LangChain
- Returns AI-generated answer based on knowledge base

## 📊 Data Structure

### Images Collection
- **Name**: `images`
- **Embedding Function**: Img2Vec (PyTorch)
- **Documents**: Landmark images
- **IDs**: `{category}-{image_name}`

### Documents Collection
- **Name**: `documents_{landmark_category}`
- **Embedding Function**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Documents**: Text chunks (300 chars, 100 overlap)
- **Metadata**: Source file information

## ⚙️ Key Components

### `main.py` - Web Interface
- Streamlit UI for image upload
- File handling and image display
- User question input
- Response display with spinner feedback

### `query_data.py` - RAG Pipeline
- Image classification
- Semantic search for relevant chunks
- LLM response generation
- Source attribution

### `create_dbs.py` - Database Setup
- Image embedding and storage
- Document loading and chunking
- ChromaDB collection creation
- Batch processing of landmark categories

### `util.py` - Custom Embeddings
- Custom `ImageEmbeddingFunction` class
- Img2Vec model initialization
- Image preprocessing (RGB conversion)

## 🔐 Configuration

### Model Settings
- **Image Embedding Model**: Img2Vec (ResNet-based)
- **Text Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Model**: Mistral (via Ollama)
- **Chunk Size**: 300 characters
- **Chunk Overlap**: 100 characters
- **Top-K Retrieval**: 3 documents

### Directory Configuration (in `util.py`)
```python
DB_PATH = "./db"        # ChromaDB storage location
DATA_PATH = "./data"    # Input data location
```

## 💡 Usage Examples

### Example 1: Eiffel Tower
1. Upload a photo of the Eiffel Tower
2. Ask: "When was it built?"
3. AI responds with construction date and historical details

### Example 2: Statue of Liberty
1. Upload a photo of the Statue of Liberty
2. Ask: "Who designed this monument?"
3. AI provides designer information and significance

## 🚀 Advanced Features

### Customization Options

**Add New Landmarks:**
1. Create directory in `data/`: `data/my_landmark/`
2. Add images: `data/my_landmark/image1.jpeg`, etc.
3. Add description: `data/my_landmark/description.txt`
4. Run: `python create_dbs.py`
5. Restart Streamlit app

**Modify Chunk Size:**
Edit in `create_dbs.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Change this
    chunk_overlap=100    # And/or this
)
```

**Change LLM Model:**
Edit in `query_data.py`:
```python
model = ChatOllama(model="llama2")  # Change model
```

## 📊 Performance

- **Image Classification**: ~500ms (GPU: ~100ms)
- **Semantic Search**: ~100ms
- **LLM Response**: ~2-5 seconds (depends on model)
- **Total Time**: ~3-7 seconds per query

## ⚠️ Troubleshooting

### "Ollama not running"
- Ensure Ollama is running: `ollama serve`
- Check if Mistral model is installed: `ollama list`

### "Model not found" error
- Pull the model: `ollama pull mistral`

### "CUDA out of memory"
- Reduce batch size or use CPU only
- For CPU mode: Modify embedding functions

### "Image not recognized"
- Ensure image is in correct category
- Check image quality and lighting
- Verify image format (.jpeg)

### "No similar documents found"
- Add more training data to the category
- Rerun `create_dbs.py`
- Check data directory structure

## 🔄 Data Pipeline

```
Raw Data (Images + Text)
        ↓
create_dbs.py
        ↓
Image Embedding (Img2Vec) + Text Chunking
        ↓
ChromaDB Storage
        ↓
main.py
        ↓
query_data.py (RAG Pipeline)
        ↓
User Response
```

## 📝 File Formats

### Supported Image Formats
- JPEG (.jpeg)
- PNG (.png)
- JPG (.jpg)

### Text Files
- Plain text (.txt)
- One file per landmark category
- Can contain multiple paragraphs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Dhanushka0626**
- GitHub: [@Dhanushka0626](https://github.com/Dhanushka0626)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues or questions about the project, please open an issue on the [GitHub Issues](https://github.com/Dhanushka0626/AI-RAG-Application/issues) page.

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Image search/reverse image lookup
- [ ] Batch processing of multiple images
- [ ] Web deployment (Streamlit Cloud)
- [ ] Mobile app integration
- [ ] Real-time camera feed support
- [ ] Response source citations
- [ ] User feedback and model refinement
- [ ] Integration with travel APIs
- [ ] Augmented Reality (AR) features
- [ ] Multi-modal embeddings
- [ ] Fine-tuned landmark-specific models

## 📚 Knowledge Base Format

Structure your data directory like this:

```
data/
├── eiffel_tower/
│   ├── eiffel1.jpeg
│   ├── eiffel2.jpeg
│   └── information.txt
├── statue_of_liberty/
│   ├── liberty1.jpeg
│   ├── liberty2.jpeg
│   └── information.txt
└── big_ben/
    ├── bigben1.jpeg
    ├── bigben2.jpeg
    └── information.txt
```

Each `information.txt` should contain detailed descriptions about the landmark.

---

**Note**: This application is designed for educational and tourist assistance purposes. For critical decisions, always verify information with official sources. The quality of responses depends on the quality and completeness of your knowledge base data.
