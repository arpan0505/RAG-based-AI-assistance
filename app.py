import os
import json
from flask import Flask, render_template, request, Response
import requests
import chromadb

app = Flask(__name__)

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "video_transcripts"
LLM_MODEL = "llama3.1"
MERGE_THRESHOLD_SECONDS = 10 # Threshold to merge consecutive timestamps

# --- Initialize ChromaDB ---
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
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

def merge_source_chunks(documents, metadatas):
    """Merges consecutive chunks from the same source video."""
    if not metadatas:
        return []

    # Group chunks by source video
    source_groups = {}
    for i, meta in enumerate(metadatas):
        source_file = meta.get('source')
        if source_file not in source_groups:
            source_groups[source_file] = []
        source_groups[source_file].append({
            'doc': documents[i],
            'meta': meta
        })

    merged_sources = []
    for source_file, chunks in source_groups.items():
        if not chunks: continue

        # Sort chunks by their start time
        sorted_chunks = sorted(chunks, key=lambda x: x['meta'].get('start_seconds', 0))

        current_start = sorted_chunks[0]['meta'].get('start_time')
        current_end_sec = sorted_chunks[0]['meta'].get('end_seconds', 0)
        current_text = [sorted_chunks[0]['doc']]

        for i in range(1, len(sorted_chunks)):
            chunk = sorted_chunks[i]
            start_sec = chunk['meta'].get('start_seconds', 0)
            
            # If the new chunk is close enough to the previous one, merge them
            if start_sec <= current_end_sec + MERGE_THRESHOLD_SECONDS:
                current_end_sec = max(current_end_sec, chunk['meta'].get('end_seconds', 0))
                current_text.append(chunk['doc'])
            else:
                # Finish the previous merged chunk
                merged_sources.append({
                    "source": source_file,
                    "start": current_start,
                    "end": chunk['meta'].get('end_time'),
                    "summary": " ".join(current_text)
                })
                # Start a new chunk
                current_start = chunk['meta'].get('start_time')
                current_end_sec = chunk['meta'].get('end_seconds', 0)
                current_text = [chunk['doc']]
        
        # Add the last merged chunk
        merged_sources.append({
            "source": source_file,
            "start": current_start,
            "end": sorted_chunks[-1]['meta'].get('end_time'),
            "summary": " ".join(current_text)
        })

    return merged_sources

def generate_response_stream(prompt, sources_list):
    """
    Streams the LLM response and then the final sources list.
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True
        )
        r.raise_for_status()

        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                try:
                    data = json.loads(chunk.decode('utf-8'))
                    token = data.get("response", "")
                    if token:
                        yield json.dumps({"type": "token", "content": token}) + '\n'
                except json.JSONDecodeError: continue
        
        if sources_list:
            yield json.dumps({"type": "sources", "content": sources_list}) + '\n'

    except requests.exceptions.RequestException as e:
        print(f"Error calling the generation API: {e}")
        yield json.dumps({"type": "error", "content": "Error: Could not connect to the language model."}) + '\n'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not collection: return Response("Error: ChromaDB not available.", status=500)

    data = request.get_json()
    query = data.get('question')
    if not query: return Response("Error: No question.", status=400)

    try:
        results = collection.query(query_texts=[query], n_results=10, include=["documents", "metadatas"])
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        context = "\n\n---\n\n".join([f"Source: {m.get('source')}\nContent: {d}" for d, m in zip(documents, metadatas)])
        
        # Merge and prepare sources for display
        merged_sources = merge_source_chunks(documents, metadatas)

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return Response("Error retrieving context.", status=500)

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    return Response(generate_response_stream(prompt, merged_sources), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
