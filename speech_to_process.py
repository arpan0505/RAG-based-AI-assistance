import whisper

model = whisper.load_model("large")
result = model.transcribe("audios/10_Document Loaders in LangChain.mp3", language="hi", task="translate")
print(result["text"])
