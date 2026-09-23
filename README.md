# RAG Project — Chat With Your Own Notes

A simple Retrieval-Augmented Generation (RAG) system built in Python. 
It lets you ask natural-language questions about a text document, and 
answers them using only content retrieved from that document — rather 
than relying on the model's own general knowledge.

Built as a hands-on project to learn how embeddings, vector search, and 
LLMs combine to let a model answer questions about information it was 
never trained on.

## How it works

1. **Ingestion** (`ingest.py`) — reads a text file, splits it into 
   chunks, converts each chunk into an embedding using a local sentence-
   transformer model, and stores everything in a persistent ChromaDB 
   vector database.
2. **Query** (`query.py`) — takes a user's question, embeds it using the 
   same model, searches the vector database for the most semantically 
   similar chunks, then sends those chunks along with the question to 
   an LLM (via the Groq API) to generate a grounded answer.

## Files

- `ingest.py` — loads a text file, generates embeddings, stores them in 
  ChromaDB
- `query.py` — interactive CLI for asking questions against the stored 
  data
- `personalchoices.txt` — example source document used for testing

## Tech used

- Python 3
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) 
  — generates embeddings locally, no API required
- [ChromaDB](https://www.trychroma.com/) — vector database for storing 
  and searching embeddings
- [Groq API](https://groq.com) (`openai/gpt-oss-20b`) — generates the 
  final answer from retrieved context

## Setup

1. Clone this repo:
```bash
   git clone https://github.com/iamsabanawaz/rag-project.git
   cd rag-project
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
   pip install groq chromadb sentence-transformers
```

4. Set your Groq API key:
```bash
   export GROQ_API_KEY="your-key-here"
```
   Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

## Usage

First, build the vector database from your document:
```bash
python ingest.py
```

Then ask questions against it:
```bash
python query.py
```

Type `quit` to exit.
