import os
import json
from flask import Flask, render_template, request, Response
import requests
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "video_transcripts"
LLM_MODEL = "llama3.1" 
EMBEDDING_MODEL = "bge-m3"

# --- Initialize ChromaDB and Embedding Function ---
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print("Successfully connected to ChromaDB collection.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None

# --- Prompt Template ---
PROMPT_TEMPLATE = '''
You are a helpful AI assistant for a SQL course. The user is asking a question about the course content, which is provided below as context from video transcripts. 

**Context from Video Transcripts:**
{context}

**User's Question:**
"{question}"

Based on the provided context, answer the user's question. If the context contains the answer, identify the specific video and guide the user to the relevant concept. If the context is not sufficient to answer the question, politely state that the information is not available in the provided video transcripts. Do not make up information.
'''

def generate_response_stream(prompt):
    """
    Sends a request to the language model and streams the response.
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        r.raise_for_status()  # Raise an exception for bad status codes

        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                try:
                    # The response from the streaming API is a series of JSON objects, one per line
                    data = json.loads(chunk.decode('utf-8'))
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    # In case of malformed JSON, just continue
                    continue

    except requests.exceptions.RequestException as e:
        print(f"Error calling the generation API: {e}")
        yield "Error: Could not connect to the language model."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not collection:
        return Response("Error: ChromaDB collection is not available.", status=500)

    data = request.get_json()
    incoming_query = data.get('question')

    if not incoming_query:
        return Response("Error: No question provided.", status=400)

    # 1. Query ChromaDB to get relevant context
    try:
        results = collection.query(
            query_texts=[incoming_query],
            n_results=5  # Retrieve the top 5 most relevant chunks
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        context_parts = []
        for i, doc in enumerate(documents):
            source = metadatas[i].get('source', 'Unknown source')
            context_parts.append(f"Source: {source}\nContent: {doc}")
        
        context = "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return Response("Error: Failed to retrieve context from the database.", status=500)

    # 2. Create the prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=incoming_query)

    # 3. Stream the response
    return Response(generate_response_stream(prompt), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)