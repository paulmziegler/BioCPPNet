import numpy as np
import pytest

from src.metrics.sisdr import calculate_sisdr

def test_sisdr_identical():
    """Identical signals should have infinite SI-SDR."""
    ref = np.random.randn(1000)
    est = ref.copy()
    sisdr = calculate_sisdr(ref, est)
    assert np.isinf(sisdr)
    
def test_sisdr_scaled():
    """Scaled signals should have high SI-SDR because it's scale-invariant."""
    ref = np.random.randn(1000)
    est = 5.0 * ref
    sisdr = calculate_sisdr(ref, est)
    # Floating point precision may result in a finite but very large number
    assert sisdr > 100.0
    
def test_sisdr_with_noise():
    """Adding noise should decrease SI-SDR."""
    ref = np.random.randn(1000)
    noise1 = 0.1 * np.random.randn(1000)
    noise2 = 0.5 * np.random.randn(1000)
    
    est1 = ref + noise1
    est2 = ref + noise2
    
    sisdr1 = calculate_sisdr(ref, est1)
    sisdr2 = calculate_sisdr(ref, est2)
    
    # Lower noise -> higher SI-SDR
    assert sisdr1 > sisdr2
    
def test_sisdr_mismatched_lengths():
    """Mismatched lengths should raise ValueError."""
    ref = np.random.randn(1000)
    est = np.random.randn(900)
    with pytest.raises(ValueError):
        calculate_sisdr(ref, est)
