"""
Merge multiple MP3 files into one MP4 video with a static banner image.
The script will:
1. Combine MP3s in random order up to specified duration
2. Create a video using a static banner image
3. Output a single MP4 file with the combined audio
"""

import os
import random
from pydub import AudioSegment
import moviepy

# Configuration
root_dir = "/home/jxu/Downloads/lyric/"
max_duration = 1 * 60 * 60 * 1000  # one hour in milliseconds
banner_image = "/home/jxu/Downloads/DALL·E 2025-02-09 19.41.02 - A peaceful YouTube thumbnail designed for Japanese study music with an anime aesthetic, featuring a serene anime-style girl with long, flowing hair, w.webp"

# Get all MP3 files
mp3_files = [f for f in os.listdir(root_dir) if f.endswith('.mp3')]
combined = AudioSegment.empty()
current_duration = 0

# Merge audio files
while current_duration <= max_duration:
    # Reshuffle playlist if needed
    if not mp3_files:
        mp3_files = [f for f in os.listdir(root_dir) if f.endswith('.mp3')]
        random.shuffle(mp3_files)
    
    # Add next track
    mp3_file = mp3_files.pop()
    audio = AudioSegment.from_mp3(os.path.join(root_dir, mp3_file))
    combined += audio
    current_duration += len(audio)

# Export merged audio
output_path = os.path.join(root_dir, "merged.mp3")
combined.export(output_path, format="mp3")
print(f"Successfully merged tracks into {output_path} (duration: {current_duration/1000:.2f} seconds)")

# Create video with static image
audio_clip = moviepy.AudioFileClip(output_path)
video_clip = moviepy.ImageClip(banner_image, duration=audio_clip.duration)
final_clip = moviepy.CompositeVideoClip([video_clip]).with_audio(audio_clip)

# Export optimized MP4
video_output_path = os.path.join(root_dir, "merged.mp4")
final_clip.write_videofile(
    video_output_path,
    fps=1,
    codec='libx264',
    audio_codec='aac',
    preset='ultrafast',
    threads=4,
    bitrate='2000k'
)

# Cleanup
audio_clip.close()
video_clip.close()
final_clip.close()

print(f"Successfully created video at {video_output_path} (duration: {current_duration/1000:.2f} seconds)")