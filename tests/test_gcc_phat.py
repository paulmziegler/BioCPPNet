import numpy as np
import pytest
from src.spatial.estimators import GCCPHAT
from src.spatial.physics import apply_subsample_shifts

def test_gcc_phat_delay_estimation():
    """Verify GCC-PHAT recovers known fractional delays."""
    fs = 100000
    n_samples = 2048
    n_channels = 2
    
    # 1. Generate Noise Signal (Broadband is best for GCC)
    # White noise
    signal_mono = np.random.randn(n_samples)
    
    # 2. Apply Delays
    # Ref channel 0: delay 0
    # Channel 1: delay 5.5 samples (55 microseconds)
    true_delays = np.array([0.0, 5.5 / fs])
    
    # Use physics to apply precise fractional delay
    # Note: apply_subsample_shifts expects (N_samples,) or (N_ch, N_samp) if we map properly
    # We want to broadcast mono to multi with delays.
    # physics.apply_subsample_shifts handles this?
    # No, my implementation:
    # "If signal is 1D: sig_fft is (N_freqs,). Broadcasts to (N_channels, N_freqs)."
    # Yes, it works.
    
    multichannel = apply_subsample_shifts(signal_mono, true_delays, fs)
    
    # 3. Estimate
    # Mic positions don't matter for pure TDOA (just needed for init)
    dummy_pos = np.zeros((2, 3)) 
    estimator = GCCPHAT(fs, dummy_pos)
    
    est_delays = estimator.estimate(multichannel)
    
    # 4. Verify
    # Error should be small (< 0.1 sample) thanks to parabolic interp
    est_samples = est_delays * fs
    true_samples = true_delays * fs
    
    error = np.abs(est_samples - true_samples)
    print(f"True Samples: {true_samples}")
    print(f"Est  Samples: {est_samples}")
    print(f"Max Error: {np.max(error)}")
    
    assert np.allclose(est_samples, true_samples, atol=0.1)

def test_gcc_phat_noise_robustness():
    """Verify GCC-PHAT works in noise (PHAT weighting benefit)."""
    fs = 100000
    signal_mono = np.random.randn(2048)
    true_delays = np.array([0.0, -3.2 / fs])
    
    multichannel = apply_subsample_shifts(signal_mono, true_delays, fs)
    
    # Add noise (SNR 0dB)
    noise = np.random.randn(*multichannel.shape)
    noisy_input = multichannel + noise
    
    estimator = GCCPHAT(fs, np.zeros((2,3)))
    est_delays = estimator.estimate(noisy_input)
    
    est_samples = est_delays * fs
    true_samples = true_delays * fs
    
    # Tolerance relaxed for high noise
    assert np.allclose(est_samples, true_samples, atol=0.5)
