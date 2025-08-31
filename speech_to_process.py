import whisper
import os
import json
import time

# Audio file path
audio_file = "audios/0_Introduction_to_Structured_Query_Language_All_Points_regarding_its_Features_and_Syllabus.mp3"

if os.path.exists(audio_file):
    print(f"Processing: {audio_file}")
    
    start_time = time.time()
    
    # Use original Whisper with GPU
    print("Loading Whisper model on GPU...")
    model = whisper.load_model("large", device="cuda")
    
    print("Starting transcription...")
    result = model.transcribe(audio_file, language="hi", task="translate", word_timestamps=False)
    
    print("Full transcription:")
    print(result["text"])
    
    # Process segments
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "start": segment["start"],
            "end": segment["end"], 
            "text": segment["text"]
        })
        print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n✅ Processing completed in {processing_time/60:.1f} minutes")
    
    # Save outputs
    with open("output.json", "w", encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    with open("full_transcript.txt", "w", encoding='utf-8') as f:
        f.write(result["text"])
    
    print("Files saved: output.json and full_transcript.txt")
    
else:
    print(f"File not found: {audio_file}")
