import numpy as np
import pytest
from src.spatial.beamforming import Beamformer
from src.spatial.physics import (
    azimuth_elevation_to_vector,
    calculate_steering_vector,
    apply_subsample_shifts
)

def test_delay_and_sum_reversibility():
    """Verify that a signal generated at angle A is recovered by beamforming at angle A."""
    fs = 250000
    bf = Beamformer(sample_rate=fs)
    
    # Simple linear array
    bf.mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    bf.speed_of_sound = 343.0
    
    # 1. Generate Signal (Sine Wave)
    t = np.linspace(0, 0.01, int(fs * 0.01), endpoint=False)
    signal = np.sin(2 * np.pi * 1000 * t)
    
    # 2. Simulate arrival from 45 degrees
    azimuth = 45.0
    source_vec = azimuth_elevation_to_vector(azimuth)
    dist_diffs = calculate_steering_vector(bf.mic_positions, source_vec)
    # arrival delay = -d/c
    delays = -dist_diffs / bf.speed_of_sound
    
    multichannel = apply_subsample_shifts(signal, delays, fs)
    
    # 3. Beamform back towards 45 degrees
    recovered = bf.delay_and_sum(multichannel, azimuth_deg=45.0)
    
    # 4. Check similarity
    # Since D&S aligns perfectly, recovered should == original signal (ignoring numerical noise)
    # Mean Square Error
    mse = np.mean((recovered - signal) ** 2)
    assert mse < 1e-10, f"MSE {mse} is too high for perfect alignment"

def test_beamforming_suppression():
    """Verify that beamforming suppresses signals from off-axis directions."""
    fs = 250000
    bf = Beamformer(sample_rate=fs)
    bf.mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],  # 10cm spacing
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0]
    ])
    
    t = np.linspace(0, 0.01, int(fs * 0.01), endpoint=False)
    signal = np.sin(2 * np.pi * 5000 * t)  # 5kHz tone
    
    # Signal from 90 deg (side)
    azimuth_source = 90.0
    source_vec = azimuth_elevation_to_vector(azimuth_source)
    dist_diffs = calculate_steering_vector(bf.mic_positions, source_vec)
    delays = -dist_diffs / bf.speed_of_sound
    multichannel = apply_subsample_shifts(signal, delays, fs)
    
    # Beamform towards 0 deg (front) -> Should be suppressed
    beamformed = bf.delay_and_sum(multichannel, azimuth_deg=0.0)
    
    # Calculate power reduction
    p_in = np.mean(multichannel[0] ** 2)
    p_out = np.mean(beamformed ** 2)
    
    # Should reduce power significantly (destructive interference)
    # Not perfect zero due to sidelobes, but significant
    ratio = p_out / p_in
    assert ratio < 0.5, f"Power ratio {ratio} is not sufficiently suppressed (should be < 0.5)"
