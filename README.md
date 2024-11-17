# HUST RAG Chatbot

An intelligent RAG (Retrieval Augmented Generation) powered chatbot for Bach Khoa University, designed to assist students with inquiries about academic regulations, scholarships, training programs, documentation, and admissions.

## Features

- 🤖 Automatic classification between chitchat and knowledge-based queries
- 🔍 Information retrieval using vector similarity search
- 💭 Context-aware responses based on conversation history
- 📚 Supabase integration for document storage and retrieval
- 🔄 Intelligent query expansion based on chat context
- 🎯 High accuracy through semantic routing
- 🚀 Modular and extensible architecture

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│    User     │────>│  Semantic   │────>│  Semantic   │────>│  LLMs   │
│   Input     │     │   Cache     │     │   Router    │     │         │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────┘
                          │                     │                  ▲
                          │                     │                  │
                          ▼                     ▼                  │
                    ┌─────────────┐      ┌─────────────┐         │
                    │   Vector    │<─────│  Reflection  │         │
                    │  Database   │      └─────────────┘         │
                    └─────────────┘             │                │
                          │                     │                │
                          │                     ▼                │
                          └──────────────>┌─────────────┐       │
                                         │    RAGs      │───────┘
                                         │   System     │
                                         └─────────────┘
```

## System Requirements

- Python 3.8+
- Supabase account
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/bk-rag-chatbot.git
cd bk-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Project Structure

```
bk-rag-chatbot/
├── src/
│   ├── base.py           # Abstract base classes
│   ├── routers.py        # Question classification
│   ├── retrievers.py     # Document retrieval
│   ├── reflection.py     # Query expansion
│   ├── responders.py     # Response generation
│   └── chatbot.py        # Main chatbot class
├── tests/
│   └── ...              # Test files
├── requirements.txt      # Project dependencies
├── .env                 # Environment variables
└── README.md            # Project documentation
```

## Usage

### Component Extension

#### Adding a Custom Router

```python
from src.base import BaseRouter

class CustomRouter(BaseRouter):
    async def classify(self, question: str) -> str:
        # Implement your custom classification logic
        return "chitchat" or "rag"
```

#### Adding a Custom Retriever

```python
from src.base import BaseRetriever
from typing import List, Dict

class CustomRetriever(BaseRetriever):
    async def retrieve(self, query: str) -> List[Dict]:
        # Implement your custom retrieval logic
        return documents
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation accordingly
- Use type hints
- Keep components modular and single-responsibility

## Performance Optimization

- Implement caching for frequent queries
- Use batch