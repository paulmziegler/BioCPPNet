import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.noise_models import WhiteNoise, ColoredNoise, RainNoise
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_noise")

def plot_psd(noise_signals, fs, title, filename_suffix):
    """
    Plots the Power Spectral Density of generated noise.
    """
    plt.figure(figsize=(10, 6))
    
    for label, sig in noise_signals.items():
        # Compute PSD (Average over channels)
        f, Pxx = welch(sig, fs, nperseg=1024, axis=-1)
        mean_Pxx = np.mean(Pxx, axis=0) # Avg across channels
        
        plt.loglog(f, mean_Pxx, label=label)
        
    plt.title(f"PSD Comparison: {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V**2/Hz)")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    save_path = get_plot_path(f"noise_psd_{filename_suffix}")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")

def test_white_noise_psd():
    """Verify White Noise has flat PSD."""
    fs = 250000
    dur = 1.0
    gen = WhiteNoise(sample_rate=fs)
    noise = gen.generate(dur, num_channels=4, std=1.0)
    
    # Check shape
    assert noise.shape == (4, int(dur * fs))
    
    # Check PSD flatness
    # PSD should be roughly constant across freq (except DC/Nyquist edge)
    f, Pxx = welch(noise, fs, nperseg=1024, axis=-1)
    mean_Pxx = np.mean(Pxx, axis=0)
    
    # Slope of log-log PSD should be ~0
    # Fit line: log(P) = m * log(f) + c
    # Ignore low freqs < 100Hz due to resolution
    mask = (f > 100) & (f < fs/2 - 1000)
    slope, intercept = np.polyfit(np.log10(f[mask]), np.log10(mean_Pxx[mask]), 1)
    
    logger.info(f"White Noise PSD Slope: {slope:.4f} (Expected ~0.0)")
    assert abs(slope) < 0.2, f"White noise slope {slope} is not flat enough"
    
    plot_psd({"White Noise": noise}, fs, "White Noise", "white")

def test_pink_noise_psd():
    """Verify Pink Noise has 1/f PSD (-10dB/decade or -3dB/octave? No, 1/f power -> -1 slope in log-log)."""
    # Power P ~ 1/f^alpha.
    # log(P) ~ -alpha * log(f).
    # Pink: alpha=1 -> slope -1.
    # Brown: alpha=2 -> slope -2.
    
    fs = 250000
    dur = 1.0
    gen = ColoredNoise(sample_rate=fs, color='pink')
    noise = gen.generate(dur, num_channels=4, std=1.0)
    
    f, Pxx = welch(noise, fs, nperseg=1024, axis=-1)
    mean_Pxx = np.mean(Pxx, axis=0)
    
    mask = (f > 100) & (f < fs/2 - 1000)
    slope, intercept = np.polyfit(np.log10(f[mask]), np.log10(mean_Pxx[mask]), 1)
    
    logger.info(f"Pink Noise PSD Slope: {slope:.4f} (Expected ~ -1.0)")
    assert abs(slope + 1.0) < 0.2, f"Pink noise slope {slope} is incorrect"
    
    plot_psd({"Pink Noise": noise}, fs, "Pink Noise", "pink")

def test_brown_noise_psd():
    """Verify Brown Noise has 1/f^2 PSD (slope -2)."""
    fs = 250000
    dur = 1.0
    gen = ColoredNoise(sample_rate=fs, color='brown')
    noise = gen.generate(dur, num_channels=4, std=1.0)
    
    f, Pxx = welch(noise, fs, nperseg=1024, axis=-1)
    mean_Pxx = np.mean(Pxx, axis=0)
    
    mask = (f > 100) & (f < fs/2 - 1000)
    slope, intercept = np.polyfit(np.log10(f[mask]), np.log10(mean_Pxx[mask]), 1)
    
    logger.info(f"Brown Noise PSD Slope: {slope:.4f} (Expected ~ -2.0)")
    assert abs(slope + 2.0) < 0.2, f"Brown noise slope {slope} is incorrect"
    
    plot_psd({"Brown Noise": noise}, fs, "Brown Noise", "brown")

def test_rain_noise_impulses():
    """Verify Rain Noise generates sparse impulses."""
    fs = 250000
    dur = 0.1
    rate = 100.0 # 100 drops/sec -> 10 drops in 0.1s
    amp = 1.0
    
    gen = RainNoise(sample_rate=fs)
    noise = gen.generate(dur, num_channels=1, rate_hz=rate, amplitude=amp)
    
    # Count non-zero samples
    non_zeros = np.sum(np.abs(noise) > 0)
    expected = int(rate * dur)
    
    # Statistical check (might fail if random falls on same sample, unlikely with 25k samples)
    # Allow small variance
    assert 5 <= non_zeros <= 20, f"Expected around 10 drops, got {non_zeros}"
    
    # Check amplitude
    assert np.max(np.abs(noise)) <= amp
    
    # Plot time series
    plt.figure()
    plt.plot(np.linspace(0, dur, len(noise[0])), noise[0])
    plt.title("Rain Impulse Noise (Time Domain)")
    plt.xlabel("Time (s)")
    plt.savefig(get_plot_path("noise_rain_time"))
    plt.close()
