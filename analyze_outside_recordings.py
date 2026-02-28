import os
import glob
import numpy as np
import soundfile as sf
from src.spatial.beamforming import Beamformer

def find_best_direction(signal, fs, mic_positions):
    search_len = min(fs, signal.shape[1])
    search_chunk = signal[:, :search_len]
    
    # Coarse 2D grid search: 10-degree steps
    azimuths = np.linspace(-180, 180, 37)
    elevations = np.linspace(-90, 90, 19)
    
    max_energy = -1.0
    best_az = 0.0
    best_el = 0.0
    
    bf = Beamformer(sample_rate=fs)
    # Inject positions if not using default config
    bf.mic_positions = mic_positions
    bf.n_channels = len(mic_positions)
    
    for az in azimuths:
        for el in elevations:
            bf_signal = bf.delay_and_sum(search_chunk, azimuth_deg=az, elevation_deg=el)
            energy = np.var(bf_signal)
            if energy > max_energy:
                max_energy = energy
                best_az = az
                best_el = el
                
    return best_az, best_el

def process_file(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    data, fs = sf.read(file_path)
    signal = data.T
    
    # 1. Initialize Beamformer (loads geometry from project_config.yaml)
    bf = Beamformer(sample_rate=fs)
    mic_positions = bf.mic_positions
    print(f"  Using {bf.n_channels}-channel array configuration.")
    
    # 2. Find best direction
    best_az, best_el = find_best_direction(signal, fs, mic_positions)
    print(f"  Selected Direction: Azimuth {best_az:.1f}°, Elevation {best_el:.1f}°")
    
    # 3. Beamform
    beamformed = bf.delay_and_sum(signal, azimuth_deg=best_az, elevation_deg=best_el)
    
    # 4. Normalize for better audibility
    max_val = np.max(np.abs(beamformed))
    if max_val > 0:
        beamformed = beamformed / max_val * 0.9 # Normalize to -1dB peak
        print(f"  Normalized signal (gain increased by {20*np.log10(0.9/max_val):.1f} dB)")
    
    # 5. Save
    output_dir = "results/beamformed"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"normalized_bf_{os.path.basename(file_path)}")
    sf.write(out_path, beamformed, fs)
    print(f"  Final beamformed output saved to: {out_path}")

if __name__ == "__main__":
    files = glob.glob("recordings/outside/*.wav")
    for f in sorted(files):
        process_file(f)
