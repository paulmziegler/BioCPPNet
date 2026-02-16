import numpy as np
import pytest
from src.spatial.physics import (
    calculate_steering_vector, 
    azimuth_elevation_to_vector,
    apply_subsample_shifts
)

def test_azimuth_to_vector():
    """Verify coordinate conversion."""
    # Source at 0 degrees (forward)
    v = azimuth_elevation_to_vector(0.0)
    assert np.allclose(v, [1.0, 0.0, 0.0])
    
    # Source at 90 degrees (left)
    v = azimuth_elevation_to_vector(90.0)
    assert np.allclose(v, [0.0, 1.0, 0.0])

def test_steering_vector_simple_linear():
    """Verify delays for a simple linear array along X."""
    mic_pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    # Source at 0 deg (X-axis) -> Should hit mic 2 first if source is at positive X? 
    # Let's clarify convention. "Source Direction" is usually the vector pointing to source.
    # If source is at (100, 0, 0), direction is (1, 0, 0).
    # Mic 1 is at 0, Mic 2 is at 1. Mic 2 is closer to source.
    # Distance proj: dot([1,0,0], [1,0,0]) = 1.
    # Distance proj: dot([0,0,0], [1,0,0]) = 0.
    # So Mic 2 has a 'larger' projection -> is closer -> signal arrives earlier.
    
    source_dir = np.array([1.0, 0.0, 0.0])
    diffs = calculate_steering_vector(mic_pos, source_dir)
    
    # Mic 2 (idx 1) is 1.0 closer than Mic 1 (idx 0)
    assert np.allclose(diffs, [0.0, 1.0])

def test_subsample_shift_sine_wave():
    """Verify phase shift on a pure sine wave."""
    fs = 1000.0
    freq = 10.0
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Signal: sin(2*pi*f*t)
    signal = np.sin(2 * np.pi * freq * t)
    
    # Shift by T/4 (90 degrees phase)
    # Period T = 1/10 = 0.1s. T/4 = 0.025s.
    delay = 0.025
    
    # Function applies delay: f(t - delay)
    # sin(2*pi*f*(t - delay)) = sin(2*pi*f*t - 2*pi*f*delay)
    # delay = T/4 = 1/(4f) -> phase = -2*pi*f*(1/4f) = -pi/2
    # sin(x - pi/2) = -cos(x)
    
    shifted = apply_subsample_shifts(signal, np.array([delay]), fs)[0]
    
    expected = -np.cos(2 * np.pi * freq * t)
    
    # Check correlation or max error
    # Ignore edges due to FFT assumption of periodicity (though integer cycles helps)
    error = np.max(np.abs(shifted - expected))
    assert error < 1e-5, f"Max error {error} is too high for analytical sine shift"
