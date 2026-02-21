import numpy as np
import pytest

from src.spatial.estimators import MUSIC
from src.spatial.physics import (
    azimuth_elevation_to_vector,
    calculate_steering_vector,
    apply_subsample_shifts,
)


def test_music_estimate():
    """Verify MUSIC can estimate DoA of a single tone."""
    sample_rate = 250000
    speed_of_sound = 343.0
    
    # 4-mic square array
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.0, 0.04, 0.0],
        [0.04, 0.04, 0.0],
    ])
    
    # True azimuth 45 degrees
    true_az = 45.0
    
    # 4 kHz sine wave to avoid spatial aliasing with 4cm spacing (f_nyq = c/(2d) = 4287.5 Hz)
    f0 = 4000.0
    duration = 0.01 # 10 ms
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    mono_sig = np.sin(2 * np.pi * f0 * t)
    
    # Spatialise
    source_vec = azimuth_elevation_to_vector(true_az, 0.0)
    distances = calculate_steering_vector(mic_positions, source_vec)
    delays = -distances / speed_of_sound
    
    multi_sig = apply_subsample_shifts(mono_sig, delays, sample_rate)
    
    # Estimate with MUSIC
    music = MUSIC(sample_rate, mic_positions, speed_of_sound=speed_of_sound, num_sources=1)
    
    est_az = music.estimate(multi_sig, search_resolution=1.0)
    print("Estimated azimuth:", est_az, "True azimuth:", true_az)
    
    # Allow some tolerance since search_resolution is 1 deg
    assert abs(est_az - true_az) <= 2.0, f"MUSIC estimated {est_az}, expected {true_az}"

