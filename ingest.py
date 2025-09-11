import os
import json
import ffmpeg
import whisper
import chromadb
from chromadb.utils import embedding_functions
import datetime

# --- Configuration ---
VIDEO_DIR = "videos"
AUDIO_DIR = "audios"
JSON_DIR = "jsons"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "video_transcripts"

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

def format_timestamp(seconds):
    """Formats time in seconds to H:M:S format."""
    return str(datetime.timedelta(seconds=int(seconds)))

def create_chunks_from_transcript(transcript_result, video_filename):
    """Creates chunks from the transcript result and includes metadata."""
    chunks = []
    metadatas = []
    for segment in transcript_result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        metadata = {
            "source": video_filename,
            "start_time": format_timestamp(start_time),
            "end_time": format_timestamp(end_time),
            "start_seconds": round(start_time, 2),
            "end_seconds": round(end_time, 2)
        }
        chunks.append(text)
        metadatas.append(metadata)
    return chunks, metadatas

def process_video(video_path):
    """
    Full processing pipeline for a single video.
    1. Extracts audio.
    2. Transcribes audio to text using Whisper.
    3. Chunks the text based on segments.
    4. Generates embeddings and stores them in ChromaDB.
    """
    video_filename = os.path.basename(video_path)
    base_name = os.path.splitext(video_filename)[0]
    audio_filename = f"{base_name}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    json_filename = f"{base_name}.json"
    json_path = os.path.join(JSON_DIR, json_filename)

    # Skip if JSON already exists, assuming it's been processed
    if os.path.exists(json_path):
        print(f"Skipping already processed video: {video_filename}")
        return None, None, None

    print(f"Processing video: {video_filename}")

    # 1. Extract Audio
    if not extract_audio(video_path, audio_path):
        return None, None, None

    # 2. Transcribe Audio with Whisper
    print(f"  Transcribing audio with Whisper...")
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
    except Exception as e:
        print(f"  Error during transcription: {e}")
        return None, None, None

    # Save full transcript result to a JSON file
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"  Transcript saved to {json_path}")

    # 3. Create Chunks from Transcript
    chunks, metadatas = create_chunks_from_transcript(result, video_filename)
    print(f"  Created {len(chunks)} text chunks from segments.")

    if not chunks:
        return None, None, None

    chunk_ids = [f"{base_name}_{i}" for i in range(len(chunks))]

    return chunks, metadatas, chunk_ids


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting the ingestion process...")

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))]

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        chunks, metadatas, ids = process_video(video_path)
        if chunks:
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
            all_ids.extend(ids)

    if all_chunks:
        # Check for existing IDs to avoid duplicates before adding
        existing_ids = collection.get(ids=all_ids).get('ids', [])
        if existing_ids:
            print(f"\nFound {len(existing_ids)} existing documents in ChromaDB. Filtering them out.")
            # Filter out chunks that are already in the database
            new_chunks = []
            new_metadatas = []
            new_ids = []
            for i, chunk_id in enumerate(all_ids):
                if chunk_id not in existing_ids:
                    new_chunks.append(all_chunks[i])
                    new_metadatas.append(all_metadatas[i])
                    new_ids.append(chunk_id)
            
            all_chunks = new_chunks
            all_metadatas = new_metadatas
            all_ids = new_ids

    if all_chunks:
        print(f"\nAdding {len(all_chunks)} new chunks to ChromaDB in a single batch...")
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print("Batch addition complete.")
    else:
        print("\nNo new chunks to add to ChromaDB.")

    print("\nIngestion process finished.")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")
