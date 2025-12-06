# 🏔️ Kashmir Tourism Assistant

A sophisticated RAG (Retrieval-Augmented Generation) based chatbot powered by Qwen2.5-3B-Instruct for providing comprehensive information about Kashmir tourism, attractions, and travel guidance.

![Kashmir Tourism Assistant](https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=1200)

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **RAG-Powered Responses**: Retrieves relevant context from vector database for accurate answers
- **Beautiful UI**: Stunning Kashmir night sky background with glassmorphic design
- **Real-time Chat**: Instant question display with smooth loading animations
- **Context-Aware**: Maintains conversation history for contextual responses
- **Suggested Questions**: Quick access to common tourism queries
- **Smart Formatting**: Bold headings and organized bullet points
- **Model Caching**: Fast subsequent loads with Streamlit caching
- **Error Handling**: Robust error management and user feedback
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Streamlit**: Web application framework
- **Sentence Transformers**: Text embedding (all-mpnet-base-v2)
- **Transformers**: LLM inference (Qwen2.5-3B-Instruct)
- **FAISS**: Vector similarity search
- **PyTorch**: Deep learning framework

### Frontend
- **Custom CSS**: Modern glassmorphic design
- **HTML**: Structured message display
- **Streamlit Components**: Interactive UI elements

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- 8GB+ RAM recommended
- 10GB+ disk space for models

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/kashmir-tourism-assistant.git
cd kashmir-tourism-assistant
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create requirements.txt

Create a `requirements.txt` file with the following content:

```txt
streamlit==1.28.0
sentence-transformers==2.2.2
transformers==4.35.0
torch==2.1.0
faiss-cpu==1.7.4
numpy==1.24.3
```

**Note**: For GPU support, install `faiss-gpu` instead of `faiss-cpu`.

### Step 5: Prepare Vector Store

Before running the application, you need to create the vector store:

1. Prepare your Kashmir tourism data (text documents, PDFs, etc.)
2. Run the indexing script/notebook to create `vector_store.index` and `vector_store_meta.pkl`

```python
# Example indexing code (create a separate indexing.py)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load your documents
documents = [...]  # Your Kashmir tourism documents

# Create embeddings
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Save index
faiss.write_index(index, "vector_store.index")

# Save metadata
with open("vector_store_meta.pkl", "wb") as f:
    pickle.dump({
        "chunks": documents,
        "metadata": [{"source": f"doc_{i}"} for i in range(len(documents))]
    }, f)
```

## 📁 Project Structure

```
kashmir-tourism-assistant/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── vector_store.index          # FAISS vector index
├── vector_store_meta.pkl       # Document chunks and metadata
├── README.md                   # Project documentation
│
├── data/                       # (Optional) Raw tourism data
│   ├── documents/
│   └── images/
│
├── notebooks/                  # (Optional) Jupyter notebooks
│   └── indexing.ipynb         # Vector store creation
│
└── utils/                      # (Optional) Helper functions
    ├── data_processing.py
    └── embeddings.py
```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Chatbot

1. **Wait for Models to Load**: First-time loading may take a few minutes
2. **Ask Questions**: Type your question in the chat input
3. **Use Suggestions**: Click on suggested questions in the sidebar
4. **Clear Chat**: Use the "Clear Chat" button to start fresh
5. **View History**: Scroll through previous conversations

### Example Questions

- "What are the best tourist spots in Kashmir?"
- "Tell me about adventure sports available"
- "What are the famous pilgrimage sites?"
- "Best time to visit Kashmir?"
- "What do tourists like about Kashmir?"

## ⚙️ Configuration

### Model Configuration

Edit the constants in `app.py`:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
VECTOR_STORE_PATH = "vector_store.index"
METADATA_PATH = "vector_store_meta.pkl"
```

### Response Parameters

Adjust generation parameters in the `generate_response()` function:

```python
output = model.generate(
    **inputs,
    max_new_tokens=300,      # Adjust response length
    temperature=0.4,         # Control randomness (0.0-1.0)
    top_p=0.9,              # Nucleus sampling threshold
    eos_token_id=stop_tokens,
    pad_token_id=tokenizer.eos_token_id
)
```

### Retrieval Configuration

Modify search parameters in `search_similar_chunks()`:

```python
def search_similar_chunks(query, embedding_model, index, chunks, metadata, 
                         top_k=5,              # Number of results
                         score_threshold=0.3):  # Minimum similarity score
```

## 🔧 How It Works

### 1. Document Indexing (Preprocessing)

```
Raw Documents → Text Chunks → Embeddings → FAISS Index
```

- Documents are split into manageable chunks
- Each chunk is converted to a vector embedding
- Embeddings are stored in FAISS for fast retrieval

### 2. Query Processing (Runtime)

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

1. **User Input**: Question is received from the chat interface
2. **Embedding**: Query is converted to vector representation
3. **Retrieval**: Similar document chunks are retrieved from FAISS
4. **Augmentation**: Retrieved context is added to the prompt
5. **Generation**: LLM generates response based on context
6. **Display**: Formatted response is shown to the user

### 3. Response Format

Responses follow a structured format:
- **Bold heading** (if applicable)
- 2-3 sentence paragraph summary
- Bullet points with key information

## 🎨 Customization

### Changing the Background Image

Update the CSS in `app.py`:

```python
st.markdown("""
<style>
    .stApp {
        background-image: url('YOUR_IMAGE_URL');
        ...
    }
</style>
""", unsafe_allow_html=True)
```

### Modifying Color Scheme

Adjust gradient colors in the CSS:

```python
.user-message {
    background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
}
```

### Custom System Prompt

Edit the `build_prompt()` function to modify chatbot behavior:

```python
system_prompt = """
Your custom instructions here...
"""
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Vector store not found
```
Solution: Run the indexing script first to create vector_store.index and vector_store_meta.pkl
```

**Issue**: CUDA out of memory
```
Solution: Reduce batch size or use CPU inference by setting device_map=None
```

**Issue**: Slow response times
```
Solution: 
- Use GPU if available
- Reduce max_new_tokens
- Decrease top_k in retrieval
```

**Issue**: Models taking too long to load
```
Solution: This is normal for first load. Subsequent loads will be faster due to caching.
```

### Debug Mode

Add debug information:

```python
# Add after retrieval
st.sidebar.write(f"Retrieved {len(retrieved_chunks)} chunks")
st.sidebar.write(f"Scores: {retrieved_scores}")
```

## 📊 Performance Optimization

### GPU Acceleration

For faster inference, ensure CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Model Quantization

For lower memory usage:

```python
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16,  # Use half precision
    load_in_8bit=True,          # 8-bit quantization
    device_map="auto"
)
```

## 🙏 Acknowledgments

- **Qwen Team**: For the powerful Qwen2.5-3B-Instruct model
- **Sentence Transformers**: For excellent embedding models
- **Streamlit**: For the amazing web framework
- **FAISS**: For fast similarity search
- **Kashmir Tourism**: For inspiring this project

**Made with ❤️ for Kashmir Tourism**
