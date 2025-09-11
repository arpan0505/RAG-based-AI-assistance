import os
import json
import ffmpeg
import whisper
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
VIDEO_DIR = "videos"
AUDIO_DIR = "audios"
JSON_DIR = "jsons"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "video_transcripts"
CHUNK_SIZE = 500  # The target size of each text chunk in characters
CHUNK_OVERLAP = 100  # The number of characters to overlap between chunks

# --- Ensure Directories Exist ---
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# --- Initialize ChromaDB ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"} 
)

# --- Load Whisper Model ---
# Using the 'base' model which is small and efficient.
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# --- Core Processing Functions ---

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file and saves it as MP3."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='mp3', audio_bitrate='192k', ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {video_path}: {e.stderr.decode()}")
        return False

def create_text_chunks(text):
    """Splits a long text into smaller, overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def process_video(video_path):
    """
    Full processing pipeline for a single video.
    1. Extracts audio.
    2. Transcribes audio to text using Whisper.
    3. Chunks the text.
    4. Generates embeddings and stores them in ChromaDB.
    """
    video_filename = os.path.basename(video_path)
    base_name = os.path.splitext(video_filename)[0]
    audio_filename = f"{base_name}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    json_filename = f"{base_name}.json"
    json_path = os.path.join(JSON_DIR, json_filename)

    print(f"Processing video: {video_filename}")

    # 1. Extract Audio
    if not extract_audio(video_path, audio_path):
        return

    # 2. Transcribe Audio with Whisper
    print(f"  Transcribing audio with Whisper...")
    try:
        # Set fp16=False if you are running on CPU
        result = whisper_model.transcribe(audio_path, fp16=False)
        transcript = result['text']
    except Exception as e:
        print(f"  Error during transcription: {e}")
        return
    
    # Save transcript to a JSON file
    with open(json_path, 'w') as f:
        json.dump({'transcript': transcript}, f, indent=4)
    
    print(f"  Transcript saved to {json_path}")

    # 3. Create Text Chunks
    chunks = create_text_chunks(transcript)
    print(f"  Created {len(chunks)} text chunks.")

    # 4. Generate Embeddings and Store in ChromaDB
    if chunks:
        # Check if documents already exist to avoid duplicates
        existing_ids = collection.get(where={"source": video_filename}).get('ids', [])
        if existing_ids:
            print(f"  Documents for {video_filename} already exist in ChromaDB. Skipping addition.")
            return

        chunk_ids = [f"{base_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": video_filename, "chunk_index": i} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        print(f"  Added {len(chunks)} chunks to ChromaDB.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting the ingestion process...")
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        process_video(video_path)
        
    print("\nIngestion process finished.")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")