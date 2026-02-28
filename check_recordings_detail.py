import soundfile as sf
import cv2
import os
import glob
import numpy as np

def check_wav_files(dir_path):
    print(f"Checking WAV files in: {dir_path}")
    wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
    if not wav_files:
        print("No .wav files found.")
        return

    for wav_file in wav_files:
        try:
            data, samplerate = sf.read(wav_file)
            channels = data.shape[1] if len(data.shape) > 1 else 1
            print(f"File: {os.path.basename(wav_file)}")
            print(f"  Channels: {channels}")
            print(f"  Samplerate: {samplerate}")
            print(f"  Duration: {len(data)/samplerate:.2f} seconds")
            
            # Check if channels are active (not all zero)
            if channels > 1:
                active_channels = []
                for i in range(channels):
                    if not np.allclose(data[:, i], 0):
                        active_channels.append(i)
                print(f"  Active channels (non-zero): {len(active_channels)}/{channels}")
                if len(active_channels) < channels:
                    print(f"  Inactive channel indices: {[i for i in range(channels) if i not in active_channels]}")
            
            print("-" * 20)
        except Exception as e:
            print(f"Error checking {wav_file}: {e}")

def check_video_files(dir_path):
    print(f"Checking AVI files in: {dir_path}")
    avi_files = glob.glob(os.path.join(dir_path, "*.avi"))
    if not avi_files:
        print("No .avi files found.")
        return

    for avi_file in avi_files:
        try:
            cap = cv2.VideoCapture(avi_file)
            if not cap.isOpened():
                print(f"Error opening {avi_file}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"File: {os.path.basename(avi_file)}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Frames: {frame_count}")
            print(f"  Duration: {duration:.2f} seconds")
            cap.release()
            print("-" * 20)
        except Exception as e:
            print(f"Error checking {avi_file}: {e}")

if __name__ == "__main__":
    for folder in ["recordings/desktop", "recordings/outside"]:
        check_wav_files(folder)
        check_video_files(folder)
