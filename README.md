# auRAG - Automated RAG System

A FastAPI-based Retrieval-Augmented Generation (RAG) system that processes PDF documents and answers questions using Google's Gemini AI and Pinecone vector database.

## Overview

auRAG is a document question-answering system that:
- Downloads and processes PDF documents from URLs
- Splits documents into manageable chunks
- Indexes content using Google's embedding model
- Stores vectors in Pinecone for efficient retrieval
- Answers questions using Google's Gemini 2.5 Flash model
- Provides a RESTful API for easy integration

## Features

- **PDF Processing**: Automatically downloads and processes PDF documents from URLs
- **Smart Chunking**: Uses recursive character splitting for optimal document segmentation
- **Vector Search**: Leverages Pinecone for fast similarity search
- **AI-Powered Q&A**: Uses Google's Gemini 2.5 Flash for accurate answers
- **RESTful API**: Simple HTTP endpoints for easy integration
- **Authentication**: Header-based API key authentication
- **Error Handling**: Comprehensive error handling and logging

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Text Chunking  │───▶│  Vector Storage │
│   (URL Input)   │    │   & Embedding   │    │   (Pinecone)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │◀───│   RAG Chain     │◀───│  Similarity     │
│   (Questions)   │    │  (Gemini 2.5)   │    │    Search       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- Google AI Studio API key (for Gemini)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd auRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=aurag
PINECONE_ENV=us-east-1
GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Starting the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST `/hackrx/run`

Process a document and answer questions.

**Headers:**
- `Authorization: Bearer your_api_key`

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?",
    "What recommendations are provided?"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "answers": [
    "The main topic is...",
    "The key findings include...",
    "The recommendations are..."
  ]
}
```

### Example Usage

```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer your_api_key",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/sample.pdf",
    "questions": [
        "What is the document about?",
        "What are the main points?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## Project Structure

```
auRAG/
├── main.py              # FastAPI application and endpoints
├── model.py             # RAG model implementation (Gemini + Pinecone)
├── processor.py         # Document processing and indexing
├── requirements.txt     # Python dependencies
├── temp_doc.pdf        # Temporary PDF storage
└── Utils/
    └── helper.py       # Utility functions
```

## Key Components

### Document Processing (`processor.py`)
- Downloads PDFs from URLs
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Generates embeddings using Google's embedding model
- Stores vectors in Pinecone with automatic index creation

### RAG Model (`model.py`)
- Uses Google's Gemini 2.5 Flash for question answering
- Implements similarity search with Pinecone
- Loads QA chain for document-based responses

### API Layer (`main.py`)
- FastAPI application with authentication
- Single endpoint for document processing and Q&A
- Comprehensive error handling and logging

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `aurag` |
| `PINECONE_ENV` | Pinecone environment | `us-east-1` |
| `GOOGLE_API_KEY` | Google AI API key | Required |

### Model Configuration

- **Embedding Model**: `models/embedding-001` (Google)
- **LLM Model**: `gemini-2.5-flash` (Google)
- **Chunk Size**: 300 characters with 50 character overlap
- **Similarity Search**: Top 5 most relevant chunks

## Error Handling

The system includes comprehensive error handling for:
- Invalid API keys
- PDF download failures
- Document processing errors
- Vector database issues
- Model inference errors

## Performance Considerations

- Documents are processed once and cached in Pinecone
- Similarity search returns top 5 most relevant chunks
- Uses efficient embedding models for fast processing
- Supports concurrent requests through FastAPI

## Security

- API key authentication required for all endpoints
- Bearer token format validation
- Secure environment variable handling
- Input validation and sanitization 