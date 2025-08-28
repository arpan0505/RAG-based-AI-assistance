import whisper
import os

# Since you've already extracted audios, let's process them
audio_file = "audios/Tujhe Kitna Chahein Aur.mp3"

# Check if file exists
if os.path.exists(audio_file):
    print(f"Processing: {audio_file}")
    
    model = whisper.load_model("large")
    result = model.transcribe(audio_file, language="hi", task="translate")
    print(result["text"])
else:
    print(f"File not found: {audio_file}")
