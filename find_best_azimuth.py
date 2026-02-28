import os
import glob
import numpy as np
import soundfile as sf
from src.spatial.beamforming import Beamformer

def get_uma16_geometry():
    n_mics = 16
    radius = 0.043
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
    geometry = np.zeros((n_mics, 3))
    geometry[:, 0] = radius * np.cos(angles)
    geometry[:, 1] = radius * np.sin(angles)
    return geometry

def find_best_azimuth(file_path):
    print(f"Grid Search: {os.path.basename(file_path)}")
    data, fs = sf.read(file_path)
    signal = data.T
    n_channels = signal.shape[0]
    mic_positions = get_uma16_geometry()
    
    # Use a chunk to save time
    search_len = min(fs, signal.shape[1])
    search_chunk = signal[:, :search_len]
    
    # Grid search -180 to 180 in 5 deg steps
    azimuths = np.linspace(-180, 180, 73)
    max_energy = -1.0
    best_az = 0.0
    
    # Initialize beamformer
    bf = Beamformer(sample_rate=fs)
    bf.mic_positions = mic_positions
    bf.n_channels = 16
    
    for az in azimuths:
        # Beamform and compute variance (energy)
        bf_signal = bf.delay_and_sum(search_chunk, az)
        energy = np.var(bf_signal)
        if energy > max_energy:
            max_energy = energy
            best_az = az
            
    print(f"  Best Azimuth: {best_az:.1f} degrees (Var: {max_energy:.2e})")
    return best_az

if __name__ == "__main__":
    files = glob.glob("recordings/outside/*.wav")
    for f in sorted(files):
        find_best_azimuth(f)
