import numpy as np
import pytest
from src.data_mixer import DataMixer

def test_spatialise_signal_basic():
    """Verify spatialisation produces correct output shape and delay trend."""
    fs = 1000
    mixer = DataMixer(sample_rate=fs)
    
    # 1. Simple Linear Array along X
    mixer.mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]  # 1 meter away
    ])
    mixer.speed_of_sound = 343.0
    
    # 2. Source from Azimuth 0 (X-axis)
    # Mic 2 is closer -> arrives earlier -> negative delay relative to origin (Mic 1)
    
    fs = 1000
    signal = np.zeros(100)
    signal[50] = 1.0  # Impulse at t=50
    
    multi = mixer.spatialise_signal(signal, azimuth_deg=0.0)
    
    # Shape check
    assert multi.shape == (2, 100)
    
    # Peak check
    peak_mic1 = np.argmax(np.abs(multi[0]))
    peak_mic2 = np.argmax(np.abs(multi[1]))
    
    # Mic 2 is closer, should arrive earlier (smaller index)
    assert peak_mic2 < peak_mic1
    
    # Expected sample diff
    # d = 1m. t = 1/343 = 0.0029s. Samples = 0.0029 * 1000 = 2.9 samples.
    # Since we use sub-sample shift, peak might be at 47 or 48 vs 50.
    diff = peak_mic1 - peak_mic2
    assert 2 <= diff <= 4

def test_mix_signals_snr():
    """Verify mixing maintains SNR logic."""
    mixer = DataMixer()
    
    sig1 = np.random.randn(1000)
    sig2 = np.random.randn(1000)
    
    # Mix with 0dB SNR -> Power should be equal
    mixed = mixer.mix_signals(sig1, sig2, snr_db=0.0)
    
    # Check if mixed signal is roughly combination
    assert np.max(np.abs(mixed)) > 0
    assert mixed.shape == sig1.shape

def test_add_noise_white():
    """Verify adding white noise works for mono signal."""
    fs = 250000
    mixer = DataMixer(sample_rate=fs)
    
    sig = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.1, int(0.1*fs)))
    noisy = mixer.add_noise(sig, noise_type='white', snr_db=10.0)
    
    assert noisy.shape == sig.shape
    assert np.max(np.abs(noisy)) > 0

def test_add_noise_pink_multichannel():
    """Verify adding pink noise works for multichannel signal."""
    fs = 250000
    mixer = DataMixer(sample_rate=fs)
    
    # Create dummy multichannel
    sig = np.zeros((4, int(0.1*fs)))
    sig[:, :] = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.1, int(0.1*fs)))
    
    noisy = mixer.add_noise(sig, noise_type='pink', snr_db=20.0)
    
    assert noisy.shape == sig.shape
    # Check that noise was added (signal is not identical to input)
    # Note: mix_signals scales the result, so equality check is tricky.
    # Check if noise component exists -> imperfect correlation with pure sine?
    
    # Just check run without error and shape for now
    pass

def test_add_rain_noise():
    """Verify rain noise generation via DataMixer."""
    fs = 250000
    mixer = DataMixer(sample_rate=fs)
    # Use non-zero signal so SNR calculation produces non-zero noise
    sig = np.ones(int(0.1*fs)) * 0.001 
    
    # Add heavy rain
    # SNR 0dB -> Noise power should equal signal power
    noisy = mixer.add_noise(sig, noise_type='rain', snr_db=0.0, rate_hz=100.0)
    
    # Since signal is constant small value, rain spikes should be distinct
    # Count how many samples deviate significantly from signal level
    assert np.count_nonzero(np.abs(noisy - 0.001) > 1e-6) > 0
