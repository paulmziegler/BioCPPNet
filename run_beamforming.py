import os
import glob
import numpy as np
import soundfile as sf
import scipy.signal

# We will temporarily modify the config or directly inject the 16-channel geometry.
from src.spatial.beamforming import Beamformer

def process_file(file_path, output_dir="results/beamformed"):
    print(f"
Processing {os.path.basename(file_path)}...")
    data, samplerate = sf.read(file_path)
    
    # data is (Time, Channels). We need (Channels, Time) for estimators.
    if data.ndim == 1:
        print("Signal is 1D. Skipping.")
        return
        
    signal = data.T
    n_channels, n_samples = signal.shape
    print(f"Loaded {n_channels} channels, {n_samples} samples at {samplerate} Hz.")
    
    # Initialize beamformer and inject 16-channel assumed geometry
    bf = Beamformer(sample_rate=samplerate)
    # Assume Uniform Linear Array (ULA) on the X-axis with 4cm spacing
    mic_spacing = 0.04
    bf.mic_positions = np.zeros((n_channels, 3))
    for i in range(n_channels):
        bf.mic_positions[i, 0] = i * mic_spacing
    bf.n_channels = n_channels
    
    # To handle non-stationary signals like bird calls, let's process in chunks
    chunk_duration = 1.0  # seconds
    chunk_size = int(chunk_duration * samplerate)
    
    beamformed_signal = np.zeros(n_samples)
    
    # Let's use the first chunk to estimate the overall DoA for simplicity,
    # or process chunk by chunk. Let's do chunk by chunk.
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = signal[:, start_idx:end_idx]
        
        if chunk.shape[1] < 1024:
            continue
            
        # Estimate DoA
        try:
            doa_azimuth = bf.estimate_doa(chunk)
            print(f"  Chunk {start_idx/samplerate:.1f}s - {end_idx/samplerate:.1f}s: Estimated DoA = {doa_azimuth:.1f} degrees")
            
            # Beamform towards estimated DoA
            bf_chunk = bf.delay_and_sum(chunk, azimuth_deg=doa_azimuth)
            beamformed_signal[start_idx:end_idx] = bf_chunk
        except Exception as e:
            print(f"  Error in chunk {start_idx/samplerate:.1f}s: {e}")
            # Fallback to reference channel
            beamformed_signal[start_idx:end_idx] = chunk[0, :]
            
    # Save the result
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"beamformed_{os.path.basename(file_path)}")
    sf.write(out_file, beamformed_signal, samplerate)
    print(f"Saved beamformed signal to {out_file}")

if __name__ == "__main__":
    files = glob.glob("recordings/outside/*.wav")
    for f in sorted(files):
        process_file(f)
