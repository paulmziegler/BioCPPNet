import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.spatial.beamforming import Beamformer
from src.spatial.physics import (
    azimuth_elevation_to_vector,
    calculate_steering_vector,
    apply_subsample_shifts
)
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_beamforming")

def generate_signal(freq, fs, duration, pulsed=False):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    
    if pulsed:
        # Create a Gaussian pulse envelope centered in the middle
        center = duration / 2
        width = duration / 8
        envelope = np.exp(-((t - center)**2) / (2 * width**2))
        signal = signal * envelope
        
    return t, signal

def plot_comparison(t, original, beamformed, fs, title, filename_suffix):
    """
    Generates comparison plots for Time Domain and PSD.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time Domain
    ax1.plot(t * 1000, original, label='Original (Source)', alpha=0.7)
    ax1.plot(t * 1000, beamformed, label='Beamformed (Recovered)', linestyle='--', alpha=0.7)
    ax1.set_title(f"Time Domain: {title}")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Frequency Domain (PSD)
    f_orig, Pxx_orig = welch(original, fs, nperseg=min(len(original), 1024))
    f_beam, Pxx_beam = welch(beamformed, fs, nperseg=min(len(beamformed), 1024))
    
    ax2.semilogy(f_orig, Pxx_orig, label='Original')
    ax2.semilogy(f_beam, Pxx_beam, label='Beamformed', linestyle='--')
    ax2.set_title(f"Power Spectral Density: {title}")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (V**2/Hz)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Save
    save_path = get_plot_path(f"beamforming_{filename_suffix}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")

@pytest.mark.parametrize("freq", [20, 1000, 15000, 50000])
@pytest.mark.parametrize("pulsed", [False, True])
def test_comprehensive_beamforming(freq, pulsed):
    """
    Tests beamforming reconstruction for various frequencies and signal types.
    Generates plots for verification.
    """
    fs = 250000
    duration = 0.1 # Increased to 100ms to capture 20Hz cycles (T=50ms)
    azimuth_target = 30.0
    
    # Setup Beamformer (Simple linear array)
    bf = Beamformer(sample_rate=fs)
    bf.mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0], # 5cm spacing
        [0.10, 0.0, 0.0],
        [0.15, 0.0, 0.0]
    ])
    
    # 1. Generate Signal
    t, source_signal = generate_signal(freq, fs, duration, pulsed)
    
    # 2. Simulate Propagation (Virtual Array)
    source_vec = azimuth_elevation_to_vector(azimuth_target)
    dist_diffs = calculate_steering_vector(bf.mic_positions, source_vec)
    delays = -dist_diffs / bf.speed_of_sound
    
    multichannel_signal = apply_subsample_shifts(source_signal, delays, fs)
    
    # 3. Beamform (Delay-and-Sum)
    recovered_signal = bf.delay_and_sum(multichannel_signal, azimuth_target)
    
    # 4. Verify Accuracy (MSE)
    # Align comparison: The recovered signal should match the source signal
    # Note: D&S might have slight amplitude scaling or numerical noise, 
    # but with perfect alignment it should be exact.
    mse = np.mean((source_signal - recovered_signal) ** 2)
    
    # Tolerance: Relaxed slightly for low-freq edge effects
    assert mse < 1e-8, f"MSE {mse} too high for {freq}Hz (Pulsed={pulsed})"
    
    # 5. Plot
    type_str = "Pulsed" if pulsed else "CW"
    title = f"{type_str} {freq}Hz at {azimuth_target} deg"
    filename = f"{freq}Hz_{type_str.lower()}"
    
    plot_comparison(t, source_signal, recovered_signal, fs, title, filename)
