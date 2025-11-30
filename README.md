# LLM-Powered Intelligent Document Analysis Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OCR](https://img.shields.io/badge/OCR-Tesseract%2BLLM-orange)

A sophisticated intelligent document analysis system that combines OCR, Large Language Models, and semantic search to provide comprehensive document understanding and question-answering capabilities.

## Features

### Document Processing
- **Multi-format Support**: PDF, JPG, PNG, TIFF, TXT documents
- **Advanced OCR**: Tesseract + LLM-enhanced text extraction
- **Handwritten Text Recognition**: Specialized processing for handwritten content
- **PDF Conversion**: Render PDF pages to images for processing

### AI-Powered Analysis
- **Dual LLM Backend**: Google Gemini API + Local Ollama (Gemma3)
- **RAG System**: Retrieval-Augmented Generation with FAISS vector store
- **Structured Extraction**: Automatic identification of titles, authors, dates, keywords
- **Semantic Search**: Intelligent document chunking and similarity search

### User Interfaces
- **Streamlit App**: Interactive notebook-style interface
- **FastAPI Backend**: Production-ready RESTful API
- **Web Interface**: Modern web frontend with real-time processing

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR installed on system
- Ollama (for local AI processing) - Optional
- Google Gemini API key - Optional

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/UmarKhattab09/OCR.git
cd OCR
