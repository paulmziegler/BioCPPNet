import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.spatial.estimators import GCCPHAT
from src.spatial.physics import apply_subsample_shifts
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_gcc_phat")

def test_gcc_phat_delay_estimation():
    """Verify GCC-PHAT recovers known fractional delays."""
    fs = 100000
    n_samples = 2048
    
    # 1. Generate Noise Signal (Broadband is best for GCC)
    # White noise
    signal_mono = np.random.randn(n_samples)
    
    # 2. Apply Delays
    # Ref channel 0: delay 0
    # Channel 1: delay 5.5 samples (55 microseconds)
    true_delays = np.array([0.0, 5.5 / fs])
    
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
    logger.info(f"True Samples: {true_samples}")
    logger.info(f"Est  Samples: {est_samples}")
    logger.info(f"Max Error: {np.max(error)}")
    
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

def test_gcc_phat_visualization():
    """Generate plots of GCC-PHAT Cross-Correlation."""
    fs = 100000
    signal_mono = np.random.randn(2048)
    # Channel 1 delayed by 10 samples relative to Ch 0
    true_delays = np.array([0.0, 10.0 / fs])
    
    multichannel = apply_subsample_shifts(signal_mono, true_delays, fs)
    
    estimator = GCCPHAT(fs, np.zeros((2,3)))
    est_delays, diag = estimator.estimate(multichannel, return_diagnostics=True)
    
    cc = diag['cc'] # (N_channels, N_samples)
    
    # Plot CC for Channel 1 (relative to Ref Ch 0)
    plt.figure(figsize=(10, 6))
    lags = np.arange(cc.shape[1]) - cc.shape[1] // 2
    
    plt.plot(lags, cc[1], label='GCC-PHAT (Ch 1 vs Ref)')
    plt.axvline(x=10.0, color='r', linestyle='--', label='True Delay (+10)')
    plt.axvline(x=est_delays[1]*fs, color='g', linestyle=':', label='Est Delay')
    
    plt.title("GCC-PHAT Cross Correlation")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.xlim(-50, 50) # Zoom in near zero
    
    save_path = get_plot_path("gcc_phat_cc")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved GCC-PHAT plot to {save_path}")
