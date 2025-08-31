import os
import subprocess

# Create audios directory if it doesn't exist
if not os.path.exists("audios"):
    os.makedirs("audios")

files = os.listdir("videos")

# Separate files with numbers and without numbers
numbered_files = []
unnumbered_files = []

for file in files:
    if file.endswith('.mp4'):
        # Check if file has a lecture number (Lec-XX format)
        if 'Lec-' in file:
            try:
                # Extract number from Lec-XX format
                lec_part = file.split('Lec-')[1]
                number = int(lec_part.split()[0])
                numbered_files.append((number, file))
            except (IndexError, ValueError):
                unnumbered_files.append(file)
        else:
            unnumbered_files.append(file)

# Sort numbered files by their original number
numbered_files.sort(key=lambda x: x[0])

# Process numbered files first (renumber starting from 0)
current_number = 0
for original_number, file in numbered_files:
    try:
        # Extract title - remove lecture number and clean up
        if 'Lec-' in file:
            # Remove the Lec-XX part and everything after .mp4
            title_part = file.split('Lec-')[1]
            # Remove number and clean title
            title_words = title_part.split()[1:]  # Skip the number
            clean_title = ' '.join(title_words).replace('.mp4', '')
        else:
            clean_title = file.replace('.mp4', '')
        
        # Create formatted output filename
        output_filename = f"{current_number}_{clean_title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '')}.mp3"
        
        print(f"{current_number}: {clean_title}")
        
        # Convert video to audio
        subprocess.run([
            "ffmpeg", "-i", f"videos/{file}", 
            "-vn", "-acodec", "mp3", "-ab", "128k", 
            f"audios/{output_filename}"
        ])
        
        current_number += 1
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

# Process unnumbered files after numbered ones
for file in unnumbered_files:
    try:
        # Clean up the title
        clean_title = file.replace('.mp4', '')
        
        # Create formatted output filename
        output_filename = f"{current_number}_{clean_title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '')}.mp3"
        
        print(f"{current_number}: {clean_title}")
        
        # Convert video to audio
        subprocess.run([
            "ffmpeg", "-i", f"videos/{file}", 
            "-vn", "-acodec", "mp3", "-ab", "128k", 
            f"audios/{output_filename}"
        ])
        
        current_number += 1
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print(f"\nProcessing complete! Total files processed: {current_number}")
