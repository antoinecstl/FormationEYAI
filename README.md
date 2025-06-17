# AI PDF Assistant

A ChatGPT-like interface for interacting with your PDF documents using Ollama and Streamlit.

## Features

- 📚 Upload and process multiple PDF documents
- 🤖 Chat with your documents using AI models from Ollama
- 📊 RAG (Retrieval Augmented Generation) capabilities
- 💬 Streaming responses for interactive chat experience

## Requirements

- Python 3.8+
- Ollama installed and running on your machine
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is installed and running, if you have Ollama CLI use :
   ```bash
   ollama serve
   ```
4. Pull the required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text:latest
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app3.py
   ```
   or 
      ```bash
   python3 -m streamlit run app3.py; 
   ```
2. Upload your PDF documents using the sidebar
3. Click "Process PDFs" to analyze the documents
4. Start asking questions about your documents in the chat input

## Troubleshooting

- If you encounter errors when processing PDFs, make sure Ollama is running with `ollama serve`
- If models are not loading, verify they are installed using `ollama list`
- Check the console for detailed error messages
