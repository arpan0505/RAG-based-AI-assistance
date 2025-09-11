import os
import json
from flask import Flask, render_template, request, Response, send_from_directory
import requests
import chromadb
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# --- Configuration ---
VIDEO_DIR = "videos"
JSON_DIR = "jsons"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "video_transcripts"
LLM_MODEL = "llama3.1"
MERGE_THRESHOLD_SECONDS = 10
RRF_K = 60  # Constant for Reciprocal Rank Fusion
TOP_N_RESULTS = 7 # Number of results to fetch

VIDEO_DIR_ABSOLUTE = os.path.abspath(VIDEO_DIR)

# --- Global objects for Hybrid Search ---
client = None
collection = None
bm25 = None
corpus_ids = []
corpus_docs = {}
corpus_metadatas = {}

def initialize_hybrid_search():
    """Initializes ChromaDB client and BM25 index from transcript files."""
    global client, collection, bm25, corpus_ids, corpus_docs, corpus_metadatas
    
    # 1. Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Successfully connected to ChromaDB collection.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return

    # 2. Load all documents from JSONs to build corpus for BM25
    print("Initializing BM25 keyword search index...")
    doc_texts = []
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        with open(os.path.join(JSON_DIR, json_file), 'r') as f:
            data = json.load(f)
            for i, segment in enumerate(data.get('segments', [])):
                doc_id = f"{base_name}_{i}"
                text = segment.get('text', '')
                
                if doc_id not in corpus_docs:
                    corpus_ids.append(doc_id)
                    doc_texts.append(text)
                    corpus_docs[doc_id] = text
                    corpus_metadatas[doc_id] = {
                        'source': f"{base_name}.mp4",
                        'start_time': str(segment.get('start')),
                        'end_time': str(segment.get('end')),
                        'start_seconds': segment.get('start'),
                        'end_seconds': segment.get('end')
                    }

    if doc_texts:
        tokenized_corpus = [doc.split(" ") for doc in doc_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index initialized with {len(doc_texts)} documents.")
    else:
        print("No documents found to initialize BM25 index.")

# --- Prompt Template ---
PROMPT_TEMPLATE = '''
You are a helpful AI assistant for a SQL course. A summary of the conversation so far is provided below (if any).

{history}

The user has a new question. Use the following context from video transcripts to answer it. Do not use prior knowledge. If the context is not sufficient, politely state that the information is not available in the provided transcripts.

**Context from Video Transcripts:**
{context}

**User's Question:**
"{question}"

Based on the provided context and the conversation history, answer the user's question.
'''

def merge_source_chunks(documents, metadatas):
    # (This function remains the same as before)
    if not metadatas: return []
    source_groups = {}
    for i, meta in enumerate(metadatas):
        source_file = meta.get('source')
        if source_file not in source_groups: source_groups[source_file] = []
        source_groups[source_file].append({'doc': documents[i], 'meta': meta})
    merged_sources = []
    for source_file, chunks in source_groups.items():
        if not chunks: continue
        sorted_chunks = sorted(chunks, key=lambda x: x['meta'].get('start_seconds', 0))
        merged_chunk_group = []
        for chunk in sorted_chunks:
            if not merged_chunk_group or chunk['meta']['start_seconds'] > merged_chunk_group[-1]['meta']['end_seconds'] + MERGE_THRESHOLD_SECONDS:
                if merged_chunk_group:
                    start_meta = merged_chunk_group[0]['meta']
                    end_meta = merged_chunk_group[-1]['meta']
                    combined_text = " ".join([c['doc'] for c in merged_chunk_group])
                    merged_sources.append({
                        "source": source_file, "start": start_meta['start_time'], "end": end_meta['end_time'],
                        "start_seconds_raw": start_meta['start_seconds'], "summary": combined_text
                    })
                merged_chunk_group = [chunk]
            else:
                merged_chunk_group.append(chunk)
        if merged_chunk_group:
            start_meta = merged_chunk_group[0]['meta']
            end_meta = merged_chunk_group[-1]['meta']
            combined_text = " ".join([c['doc'] for c in merged_chunk_group])
            merged_sources.append({
                "source": source_file, "start": start_meta['start_time'], "end": end_meta['end_time'],
                "start_seconds_raw": start_meta['start_seconds'], "summary": combined_text
            })
    return merged_sources

def generate_response_stream(prompt, sources_list):
    # (This function remains the same as before)
    try:
        r = requests.post("http://localhost:11434/api/generate", json={"model": LLM_MODEL, "prompt": prompt, "stream": True}, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                try:
                    data = json.loads(chunk.decode('utf-8'))
                    token = data.get("response", "")
                    if token: yield json.dumps({"type": "token", "content": token}) + '\n'
                except json.JSONDecodeError: continue
        if sources_list:
            yield json.dumps({"type": "sources", "content": sources_list}) + '\n'
    except requests.exceptions.RequestException as e:
        print(f"Error calling the generation API: {e}")
        yield json.dumps({"type": "error", "content": "Error: Could not connect to the language model."}) + '\n'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR_ABSOLUTE, filename)

@app.route('/ask', methods=['POST'])
def ask():
    if not collection or not bm25: return Response("Error: Search index not available.", status=500)

    data = request.get_json()
    query = data.get('question')
    history = data.get('history', [])
    if not query: return Response("Error: No question.", status=400)

    try:
        # 1. Semantic Search (ChromaDB)
        semantic_results = collection.query(query_texts=[query], n_results=TOP_N_RESULTS)
        semantic_ids = semantic_results.get('ids', [[]])[0]

        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_results = sorted(zip(corpus_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:TOP_N_RESULTS]
        bm25_ids = [item[0] for item in bm25_results]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        all_doc_ids = set(semantic_ids + bm25_ids)

        for doc_id in all_doc_ids:
            rrf_scores[doc_id] = 0
            if doc_id in semantic_ids:
                rank = semantic_ids.index(doc_id) + 1
                rrf_scores[doc_id] += 1 / (RRF_K + rank)
            if doc_id in bm25_ids:
                rank = bm25_ids.index(doc_id) + 1
                rrf_scores[doc_id] += 1 / (RRF_K + rank)

        sorted_fused_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:TOP_N_RESULTS]

        # 4. Prepare documents and metadatas for context and sources
        final_docs = [corpus_docs[doc_id] for doc_id in sorted_fused_ids]
        final_metadatas = [corpus_metadatas[doc_id] for doc_id in sorted_fused_ids]
        
        context = "\n\n---\n\n".join([f"Source: {m.get('source')}\nContent: {d}" for d, m in zip(final_docs, final_metadatas)])
        merged_sources = merge_source_chunks(final_docs, final_metadatas)

    except Exception as e:
        print(f"Error during hybrid search: {e}")
        return Response("Error retrieving context.", status=500)

    history_str = ""
    if history:
        history_items = [f"{turn['role']}: {turn['content']}" for turn in history]
        history_str = "**Conversation History:**\n" + "\n".join(history_items)

    prompt = PROMPT_TEMPLATE.format(history=history_str, context=context, question=query)
    return Response(generate_response_stream(prompt, merged_sources), mimetype='application/x-ndjson')

if __name__ == '__main__':
    initialize_hybrid_search()
    app.run(debug=True, threaded=True)