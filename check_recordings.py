import soundfile as sf
import os
import glob

def check_wav_files(dir_path):
    print(f"Checking directory: {dir_path}")
    wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
    if not wav_files:
        print("No .wav files found.")
        return

    for wav_file in wav_files:
        try:
            info = sf.info(wav_file)
            print(f"File: {os.path.basename(wav_file)}")
            print(f"  Channels: {info.channels}")
            print(f"  Samplerate: {info.samplerate}")
            print(f"  Duration: {info.duration:.2f} seconds")
            print(f"  Format: {info.format} ({info.format_info})")
            print("-" * 20)
        except Exception as e:
            print(f"Error checking {wav_file}: {e}")

if __name__ == "__main__":
    check_wav_files("recordings/desktop")
    check_wav_files("recordings/outside")
