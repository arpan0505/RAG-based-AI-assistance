import whisper
import json
import os

print("Loading Whisper model on GPU...")
model = whisper.load_model("large", device="cuda")

# Check if audios directory exists
if not os.path.exists("audios"):
    print("Error: 'audios' directory not found!")
    exit()

audios = os.listdir("audios")

# Create jsons directory if it doesn't exist
if not os.path.exists("jsons"):
    os.makedirs("jsons")

for audio in audios: 
    if audio.endswith('.mp3') and "_" in audio:
        try:
            # Since all files now follow the format: "number_title.mp3"
            parts = audio.split("_", 1)  # Split only on first underscore
            number = parts[0]  # First part is the number (0, 1, 2, etc.)
            title = parts[1].replace(".mp3", "")  # Everything after first underscore, remove extension
            
            print(f"Processing: {audio}")
            print(f"Number: {number}, Title: {title}")
            
            result = model.transcribe(audio = f"audios/{audio}", 
                                  language="hi",
                                  task="translate",
                                  word_timestamps=False)
            
            chunks = []
            for segment in result["segments"]:
                chunks.append({
                    "number": number, 
                    "title": title, 
                    "start": segment["start"], 
                    "end": segment["end"], 
                    "text": segment["text"]
                })
            
            chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

            # Create filename in format: number_title.json
            output_filename = f"jsons/{number}_{title}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
                
            print(f"Saved: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {audio}: {str(e)}")
    else:
        print(f"Skipping {audio} - not a valid audio file or missing underscore")
