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
            # Extract number and title based on different formats
            if "Video " in audio:
                # Format: "1_Introduction_to_LangChain__LangChain_for_Beginners__Video_1__CampusX.mp4.mp3"
                number = audio.split('Video ')[1].split('_')[0]
                title = audio.split("_")[1]
            else:
                # Format: "10_Document_Loaders_in_LangChain.mp3"
                parts = audio.split("_")
                number = parts[0]  # First part is the number
                title = "_".join(parts[1:]).replace(".mp3", "")  # Join remaining parts and remove extension
            
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
