import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.spatial.beamforming import Beamformer
from src.data_mixer import DataMixer
from src.spatial.physics import azimuth_elevation_to_vector, calculate_steering_vector, apply_subsample_shifts
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_noise_robustness")

def calculate_snr(signal, noise):
    """Calculates SNR in dB."""
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0: return np.inf
    return 10 * np.log10(p_signal / p_noise)

def generate_target_signal(freq, fs, duration):
    """Generates a pulsed sine wave target."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Gaussian pulse
    center = duration / 2
    width = duration / 6
    envelope = np.exp(-((t - center)**2) / (2 * width**2))
    return t, np.sin(2 * np.pi * freq * t) * envelope

def plot_noise_robustness(t, noisy_input, beamformed_output, clean_target, fs, title, filename_suffix):
    """
    Plots Noisy Input vs Beamformed Output vs Clean Target.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time Domain
    ax1.plot(t * 1000, noisy_input, label='Noisy Input (Ch 0)', color='lightgray', alpha=0.8)
    ax1.plot(t * 1000, clean_target, label='Clean Target', color='green', linestyle='--', alpha=0.6)
    ax1.plot(t * 1000, beamformed_output, label='Beamformed Output', color='blue', alpha=0.8)
    
    ax1.set_title(f"Time Domain: {title}")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # PSD
    f, Pxx_noisy = welch(noisy_input, fs, nperseg=1024)
    f, Pxx_beam = welch(beamformed_output, fs, nperseg=1024)
    f, Pxx_clean = welch(clean_target, fs, nperseg=1024)
    
    ax2.semilogy(f, Pxx_noisy, label='Noisy Input', color='gray', alpha=0.5)
    ax2.semilogy(f, Pxx_clean, label='Clean Target', color='green', linestyle='--')
    ax2.semilogy(f, Pxx_beam, label='Beamformed Output', color='blue')
    
    ax2.set_title(f"PSD: {title}")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = get_plot_path(f"robustness_{filename_suffix}")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")

@pytest.mark.parametrize("noise_type", ["white", "pink", "rain"])
def test_array_gain(noise_type):
    """
    Verifies that beamforming improves SNR in the presence of various noise types.
    Theoretical gain for N=4 uncorrelated noise is ~6dB.
    """
    fs = 250000
    duration = 0.05
    freq = 5000 # 5kHz target
    target_azimuth = 30.0
    input_snr_db = 0.0 # High noise scenario
    
    # 1. Setup Logic
    mixer = DataMixer(sample_rate=fs)
    bf = Beamformer(sample_rate=fs)
    # 4-mic linear array
    bf.mic_positions = np.array([[0,0,0], [0.04,0,0], [0.08,0,0], [0.12,0,0]])
    mixer.mic_positions = bf.mic_positions # Match geometry
    
    # 2. Generate Clean Target
    t, clean_mono = generate_target_signal(freq, fs, duration)
    
    # 3. Spatialise Target
    clean_multichannel = mixer.spatialise_signal(clean_mono, target_azimuth)
    
    # 4. Generate Noise
    # mixer.add_noise uses mix_signals logic.
    # We want strictly additive noise to calculate SNR easily.
    # So let's generate noise separately.
    from src.noise_models import WhiteNoise, ColoredNoise, RainNoise
    
    if noise_type == 'white':
        gen = WhiteNoise(fs)
        noise = gen.generate(duration, num_channels=4, std=1.0)
    elif noise_type == 'pink':
        gen = ColoredNoise(fs, color='pink')
        noise = gen.generate(duration, num_channels=4, std=1.0)
    elif noise_type == 'rain':
        gen = RainNoise(fs)
        noise = gen.generate(duration, num_channels=4, rate_hz=200, amplitude=2.0)
    
    # Scale noise to achieve specific SNR (based on average channel power)
    p_signal = np.mean(clean_multichannel ** 2)
    p_noise_raw = np.mean(noise ** 2)
    
    if p_noise_raw > 0:
        target_p_noise = p_signal / (10**(input_snr_db/10))
        scale = np.sqrt(target_p_noise / p_noise_raw)
        noise = noise * scale
    
    # 5. Create Noisy Mix
    noisy_input = clean_multichannel + noise
    
    # 6. Beamform
    beamformed = bf.delay_and_sum(noisy_input, target_azimuth)
    
    # 7. Calculate Input vs Output SNR
    # Input SNR (Ch 0)
    snr_in = calculate_snr(clean_multichannel[0], noise[0])
    
    # Output SNR
    # Beamforming aligns signal -> Coherent sum -> Power * N^2
    # Uncorrelated noise -> Incoherent sum -> Power * N
    # SNR gain -> N (or 10logN dB)
    
    # Signal component in output: Since D&S averages, signal amplitude is preserved (Mean).
    # Noise component in output: Mean of N uncorrelated noise sources -> Variance reduces by N.
    
    # Extract residual noise in beamformed signal
    # Since beamforming perfectly reconstructs target (in test ideal case),
    # residual = beamformed - clean_mono (shifted to align? No, D&S aligns TO reference)
    # The clean_mono was the source.
    # Does D&S align to t=0 or to the array center?
    # Our D&S implementation aligns signals *relative to each other* then sums.
    # Wait, our D&S implementation in beamforming.py:
    # `correction_delays = distance_diffs / c`
    # It aligns everything to match the reference mic (usually 0,0,0 if that's the origin).
    # Since Mic 0 is at 0,0,0, its distance diff is 0. Its delay is 0.
    # So the output should align with Channel 0's signal content.
    
    # So we compare beamformed output vs clean_multichannel[0].
    residual_noise = beamformed - clean_multichannel[0]
    snr_out = calculate_snr(clean_multichannel[0], residual_noise)
    
    gain = snr_out - snr_in
    logger.info(f"Noise: {noise_type}, SNR In: {snr_in:.2f}dB, SNR Out: {snr_out:.2f}dB, Gain: {gain:.2f}dB")
    
    # 8. Assertions
    # For Rain (sparse), SNR definition is tricky, but gain should still be positive
    # For White/Pink (uncorrelated), Gain should be ~10log10(4) = 6dB.
    # Allow some margin for randomness and Pink correlation (if any).
    if noise_type in ['white', 'pink']:
        assert gain > 3.0, f"Array gain {gain:.2f}dB too low for {noise_type} noise (Expected > 3dB)"
    else:
        # Rain is sparse, gain might vary depending on overlap, but should be positive
        assert gain > 0.0, f"Array gain {gain:.2f}dB should be positive for rain"

    # 9. Plot
    plot_noise_robustness(
        t, 
        noisy_input[0], # Show Ch0
        beamformed, 
        clean_multichannel[0], 
        fs, 
        f"Robustness to {noise_type.title()} Noise (SNR {input_snr_db}dB)", 
        f"noise_{noise_type}"
    )
