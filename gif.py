from pathlib import Path
from PIL import Image
from multiprocessing import Pool
from functools import partial
import subprocess

# (video_path, start_time_in_seconds, end_time_in_seconds)
# read from new.txt
# for each two line, first line is the video path, second line is the start and end time, seperated by space. Both can be float
txt = Path("/home/jxu/Documents/daily_utils/new.txt").read_text()
lines = txt.splitlines()
videos = []
for i in range(0, len(lines), 2):  # Step by 2 to process pairs of lines
    if i + 1 >= len(lines):  # Check if we have both lines
        break
    video_path = lines[i].strip()
    start_time, end_time = map(float, lines[i + 1].split())
    videos.append((video_path, start_time, end_time))

output_dir = Path("/home/jxu/Documents/daily_utils/new")
output_dir.mkdir(parents=True, exist_ok=True)

def get_video_fps(video_path):
    """Get the FPS of the video using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        num, den = map(int, output.split('/'))
        return num / den
    except subprocess.CalledProcessError as e:
        print(f"Error getting FPS: {e}")
        return None

def filter_video(video_path, start_time_in_seconds, end_time_in_seconds):
    """
    Get the video from start_time_in_seconds to end_time_in_seconds, and save it to output_dir
    The result is another video (mp4)
    """
    # output_path = output_dir / (Path(video_path).name)
    output_path = output_dir / (Path(video_path).name)
    
    # Get actual FPS from video
    fps = get_video_fps(video_path)
    if fps is None:
        print("Could not determine video FPS. Aborting.")
        return
    
    # Calculate duration from start and end times
    duration = end_time_in_seconds - start_time_in_seconds
    cmd = [
        'ffmpeg',
        '-ss', str(start_time_in_seconds),
        '-t', str(duration),
        '-i', video_path,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # ensure even dimensions
        '-y',  # overwrite output if exists
        str(output_path)
    ]
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully created trimmed video: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        print(f"FFmpeg error output: {e.stderr.decode()}")

for vp, st, et in videos:
    filter_video(vp, st, et)
