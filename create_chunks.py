import whisper
import json
import os

print("Loading Whisper model on GPU...")
model = whisper.load_model("large-v2", device="cuda")

# Check if audios directory exists
if not os.path.exists("audios"):
    print("Error: 'audios' directory not found!")
    exit()

audios = os.listdir("audios")

# Create jsons directory if it doesn't exist
if not os.path.exists("jsons"):
    os.makedirs("jsons")

for audio in audios: 
    if("_" in audio and "Video " in audio):
        try:
            # Extract number and title
            number = audio.split('Video ')[1].split()[0]
            title = audio.split("_")[1][:-4]
            
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

            # Save with .json extension (remove original extension first)
            output_filename = f"jsons/{os.path.splitext(audio)[0]}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
                
            print(f"Saved: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {audio}: {str(e)}")
    else:
        print(f"Skipping {audio} - doesn't match expected format")
