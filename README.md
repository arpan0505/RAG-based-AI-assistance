#  RAG based AI Assistant

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that functions as an AI assistant capable of answering questions about a collection of video content. It features a user-friendly web interface and an advanced search mechanism to provide accurate, context-aware answers with clickable sources.

## Features

- **Web-Based Chat Interface**: A clean and modern chat UI built with Flask and Bootstrap, featuring real-time streaming of AI responses.
- **Hybrid Search Engine**: Combines semantic (vector) search with traditional keyword (BM25) search for highly accurate and relevant context retrieval.
- **Conversation History**: The assistant remembers previous turns in the conversation, allowing for natural follow-up questions.
- **Clickable, Contextual Sources**: Responses include a list of source video clips. Timestamps are clickable, opening the video at the precise moment in a new tab. Each source also includes a text snippet for immediate context.
- **Automated Data Ingestion**: A simple script (`ingest.py`) processes video files, transcribes the audio using Whisper, and builds the necessary search indexes.
- **Local First**: Runs entirely on your local machine, using local models via Ollama and local storage for search indexes.

## Architecture

- **Backend**: Flask
- **Frontend**: HTML, Bootstrap CSS, vanilla JavaScript
- **Speech-to-Text**: `openai-whisper`
- **Language Model (LLM)**: Powered by a local model running on **Ollama** (e.g., Llama 3.1).
- **Vector Database**: `ChromaDB` for efficient semantic search.
- **Keyword Search**: `rank_bm25` for robust keyword-based retrieval.
- **Video/Audio Processing**: `ffmpeg`

## Setup and Installation

Follow these steps to set up and run the application on your local machine.

### Prerequisites

1.  **Python 3.8+**
2.  **ffmpeg**: You must have `ffmpeg` installed and available in your system's PATH. This is required for audio extraction.
3.  **Ollama**: You need [Ollama](https://ollama.com/) installed and running. You must also have a model downloaded.
    ```sh
    # Example: Pull the Llama 3.1 model
    ollama pull llama3.1
    ```

### Installation Steps

1.  **Clone the Repository** (or ensure all project files are in a single directory).

2.  **Create a Virtual Environment**:
    ```sh
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    -   On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Install all required Python packages from the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  **Add Your Videos**:
    Place all the video files you want to query (e.g., `.mp4`, `.mov`, `.mkv`) inside the `videos/` directory.

2.  **Ingest the Data**:
    Run the ingestion script. This will process your videos, transcribe them, and build the search indexes. This step only needs to be run once per video or when you add new videos.
    ```sh
    python ingest.py
    ```

3.  **Run the Web Application**:
    Start the Flask server. Make sure your Ollama application is already running in the background.
    ```sh
    python app.py
    ```

4.  **Access the AI Assistant**:
    Open your web browser and navigate to **http://127.0.0.1:5000**. You can now start asking questions!

## Project File Structure

```
.
├── app.py             # The main Flask web application
├── ingest.py           # Script for data processing and ingestion
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── videos/             # Directory to store your source video files
├── audios/             # Stores audio files extracted from videos
├── jsons/              # Stores the generated raw transcript files
├── chroma_db/          # Directory for the ChromaDB vector store
├── templates/
│   └── index.html      # Frontend HTML and JavaScript
└── ...
```
