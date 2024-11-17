# HUST Chatbot

HUST Chatbot is an intelligent virtual assistant developed for Hanoi University of Science and Technology (HUST), utilizing the RAG (Retrieval-Augmented Generation) architecture to provide accurate and contextually appropriate responses.

## Features

- 🤖 Support for both casual conversation and information retrieval
- 📚 Automatic question classification and query expansion
- 🔍 Relevant information retrieval from database
- 💬 User-friendly chat interface
- ✨ Markdown support in messages

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd hust-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file and configure environment variables:
```
OPENAI_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
CACHE_FOLDER=path_to_cache_folder
```

## Project Structure

```
hust-chatbot/
├── app.py                 # Entry point with Gradio interface
├── base.py               # Abstract base classes
├── chatbot.py            # Main chatbot implementation
├── router.py             # Question classifier
├── responder.py          # Response generators
├── retrieval.py          # Document retrieval
├── reflection.py         # Query expansion
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## System Architecture

The chatbot is built with a modular architecture consisting of main components:

1. **Router**: Classifies questions as either "chitchat" or "query"
2. **Retriever**: Searches for relevant documents from Supabase
3. **Reflection**: Expands queries based on context
4. **Responder**: 
   - ChitchatResponder: Handles casual conversation
   - RAGResponder: Generates responses based on retrieved information

## Usage

Run the application:
```bash
python app.py
```

Access the web interface at `http://localhost:7860`

## Tech Stack

- **Framework**: Gradio
- **LLM**: OpenAI GPT
- **Vector Database**: Supabase
- **Embedding Model**: BAAI/bge-m3
- **Other Libraries**:
  - LangChain
  - SentenceTransformers
  - Python-dotenv
  - PyVi

## Development

### Adding New Features

1. Create new classes inheriting from abstract base classes in `base.py`
2. Implement required abstract methods
3. Register new components in `app.py`

### Code Example

```python
# Example of implementing a custom router
from base import BaseRouter

class CustomRouter(BaseRouter):
    async def classify(self, question: str) -> str:
        # Implementation for question classification
        return "query" if "what" in question.lower() else "chitchat"
```

### Testing

```bash
# Run tests
python -m pytest tests/
```

## API Documentation

### HustChatbot Class

```python
class HustChatbot:
    def __init__(
        self,
        router: BaseRouter,
        retriever: BaseRetriever,
        reflection: BaseReflection,
        chitchat_responder: BaseResponder,
        rag_responder: BaseResponder
    ):
        """
        Initialize the chatbot with its components.
        
        Args:
            router: Component for question classification
            retriever: Component for document retrieval
            reflection: Component for query expansion
            chitchat_responder: Component for casual conversation
            rag_responder: Component for information retrieval responses
        """
```

### Message Format

```python
@dataclass
class Message:
    content: str
    type: str  # 'human' or 'ai'
```

## Configuration

The system supports the following configuration options through environment variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-3.5-turbo # Optional, defaults to gpt-3.5-turbo

# Supabase Configuration
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx

# Model Cache
CACHE_FOLDER=/path/to/cache # For storing embedding models
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Requirements

```
gradio>=4.0.0
langchain>=0.1.0
openai>=1.0.0
python-dotenv>=1.0.0
supabase>=1.0.0
sentence-transformers>=2.2.2
pyvi>=0.1.1
```

## Troubleshooting

Common issues and solutions:

1. **OpenAI API Error**:
   - Check if your API key is valid
   - Ensure you have sufficient credits

2. **Supabase Connection Error**:
   - Verify your Supabase URL and key
   - Check if the vector store is properly set up

3. **Embedding Model Error**:
   - Ensure CACHE_FOLDER is writable
   - Check internet connection for model download

## Acknowledgments

- OpenAI for GPT models
- Supabase for vector database
- BAAI for embedding models
- Gradio team for the UI framework