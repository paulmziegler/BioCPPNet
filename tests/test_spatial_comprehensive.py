import numpy as np
import pytest

from src.spatial.beamforming import Beamformer
from src.spatial.physics import (
    azimuth_elevation_to_vector,
    calculate_steering_vector,
    apply_subsample_shifts,
)

# Test cases: (azimuth, elevation)
SPATIAL_CASES = [
    (0.0, 0.0),      # Front
    (45.0, 0.0),     # Front-Right
    (90.0, 0.0),     # Right
    (135.0, 0.0),    # Back-Right
    (180.0, 0.0),    # Back
    (-45.0, 0.0),    # Front-Left
    (-90.0, 0.0),    # Left
    (0.0, 45.0),     # Front-Up
    (90.0, 30.0),    # Right-Up
    (-135.0, -45.0), # Back-Left-Down
    (0.0, 90.0),     # Zenith (straight up)
    (0.0, -90.0),    # Nadir (straight down)
]

@pytest.mark.parametrize("az,el", SPATIAL_CASES)
def test_azimuth_elevation_to_vector_norm(az, el):
    """Verify that all generated direction vectors have a unit norm of 1.0."""
    vec = azimuth_elevation_to_vector(az, el)
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0), f"Vector norm for {az}, {el} is {norm}, expected 1.0"

@pytest.mark.parametrize("az,el", SPATIAL_CASES)
def test_spatial_reconstruction_via_beamforming(az, el):
    """
    Verify that if a signal is spatialised to a specific azimuth and elevation,
    beamforming towards that exact direction recovers the original signal accurately,
    and beamforming towards an orthogonal direction results in lower energy.
    """
    from src.utils import CONFIG
    sample_rate = CONFIG.get("audio", {}).get("sample_rate", 48000)
    speed_of_sound = 343.0
    
    # 4-mic 3D array (tetrahedron-like or 3D cube) to ensure sensitivity in all dimensions (X, Y, Z)
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05],
    ])
    
    # Create a pure tone
    f0 = 10000.0 # 10kHz
    duration = 0.01 # 10ms
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    original_signal = np.sin(2 * np.pi * f0 * t)
    
    # 1. Spatialise the signal
    source_vec = azimuth_elevation_to_vector(az, el)
    distances = calculate_steering_vector(mic_positions, source_vec)
    delays = -distances / speed_of_sound
    
    multichannel_signal = apply_subsample_shifts(original_signal, delays, sample_rate)
    
    # 2. Reconstruct via Beamformer
    beamformer = Beamformer(sample_rate=sample_rate)
    # Override defaults for testing
    beamformer.mic_positions = mic_positions
    beamformer.speed_of_sound = speed_of_sound
    beamformer.n_channels = len(mic_positions)
    
    reconstructed = beamformer.delay_and_sum(multichannel_signal, azimuth_deg=az, elevation_deg=el)
    
    # 3. Validation
    # The reconstructed signal should highly correlate with the original
    # We ignore the very edges due to FFT wrapping effects
    corr = np.corrcoef(original_signal[50:-50], reconstructed[50:-50])[0, 1]
    assert corr > 0.99, f"Reconstruction correlation failed for az={az}, el={el}. Corr={corr}"
    
    # Verify the amplitude is maintained
    orig_energy = np.mean(original_signal**2)
    recon_energy = np.mean(reconstructed**2)
    assert np.isclose(orig_energy, recon_energy, rtol=0.05), f"Energy mismatch for az={az}, el={el}"

    # 4. Negative Test: Beamform to wrong direction
    # Try an orthogonal/opposite direction
    wrong_az = az + 90.0
    wrong_el = el + 45.0
    wrong_reconstructed = beamformer.delay_and_sum(multichannel_signal, azimuth_deg=wrong_az, elevation_deg=wrong_el)
    
    wrong_energy = np.mean(wrong_reconstructed**2)
    # The wrong direction should have noticeably less energy than the matched direction
    assert wrong_energy < recon_energy, f"Beamformer did not attenuate incorrect direction az={wrong_az}, el={wrong_el}"
