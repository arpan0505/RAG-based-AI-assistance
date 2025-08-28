import os
import subprocess

files = os.listdir("videos")

for file in files:
    if 'Video' in file:  # Only process files with "Video" tag
        try:
            tutorial_number = file.split('Video ')[1].split()[0]
            
            # Extract everything before "Generative AI using LangChain"
            clean_title = file.split('Generative AI using LangChain')[0].strip()
            
            # Create formatted output filename
            output_filename = f"{tutorial_number}_{clean_title.replace(' ', '_')}.mp3"
            
            print(f"{tutorial_number}: {clean_title}")
            
            # Move subprocess call inside the try block with proper indentation
            subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{output_filename}"])
            
        except IndexError:
            continue
