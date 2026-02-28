import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal

def plot_spectrogram(data, fs, title, save_path):
    f, t, Sxx = scipy.signal.spectrogram(data, fs)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.savefig(save_path)
    plt.close()

def analyze_beamformed():
    bf_files = glob.glob("results/beamformed/beamformed_*.wav")
    if not bf_files:
        print("No beamformed files found.")
        return
        
    for bf_file in sorted(bf_files):
        print(f"Analyzing beamformed: {os.path.basename(bf_file)}")
        bf_data, fs = sf.read(bf_file)
        
        # Get original reference channel for comparison
        orig_file = os.path.join("recordings/outside", os.path.basename(bf_file).replace("beamformed_", ""))
        if os.path.exists(orig_file):
            orig_data, _ = sf.read(orig_file)
            ref_channel = orig_data[:, 0]
            
            # Save spectrogram comparison
            session_dir = "results/plots"
            os.makedirs(session_dir, exist_ok=True)
            
            base_name = os.path.basename(bf_file).replace(".wav", "")
            plot_spectrogram(ref_channel, fs, f"Original (Ch 0) - {base_name}", os.path.join(session_dir, f"{base_name}_orig.png"))
            plot_spectrogram(bf_data, fs, f"Beamformed - {base_name}", os.path.join(session_dir, f"{base_name}_bf.png"))
            
            # Compute improvement in RMS? (Not very meaningful without ground truth)
            # But let's check peak signal level
            print(f"  Reference Peak: {np.max(np.abs(ref_channel)):.4f}, Beamformed Peak: {np.max(np.abs(bf_data)):.4f}")
        else:
            print(f"  Original file {orig_file} not found for comparison.")

if __name__ == "__main__":
    analyze_beamformed()
