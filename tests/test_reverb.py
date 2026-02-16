import numpy as np
import pytest
from src.spatial.reverb import ReverbGenerator
from src.data_mixer import DataMixer

def test_rir_decay():
    """Verify generated RIR has exponential decay."""
    fs = 10000
    gen = ReverbGenerator(fs)
    rt60 = 0.5
    
    rir = gen.generate_stochastic_rir(num_channels=1, rt60=rt60, direct_ratio=0.0)
    rir = rir[0]
    
    # Check length
    assert len(rir) >= int(rt60 * fs)
    
    # Check energy decay
    # Split into chunks and check decreasing energy
    chunk_size = int(0.1 * fs)
    energies = []
    for i in range(0, len(rir), chunk_size):
        chunk = rir[i:i+chunk_size]
        if len(chunk) == chunk_size:
            energies.append(np.sum(chunk**2))
            
    # Ignore the very first chunk (predelay/attack) and tail noise floor
    # Middle chunks should decrease
    valid_energies = energies[1:5] 
    assert all(valid_energies[i] > valid_energies[i+1] for i in range(len(valid_energies)-1)), "RIR energy does not decay monotonically"

def test_spatialise_with_reverb():
    """Verify spatialisation adds length when reverb is on."""
    fs = 10000
    mixer = DataMixer(fs)
    mixer.mic_positions = np.array([[0,0,0], [1,0,0]]) # 2 mics
    
    signal = np.zeros(1000) # 0.1s
    signal[500] = 1.0 # Impulse
    
    # 1. Without Reverb
    out_clean = mixer.spatialise_signal(signal, 0.0, add_reverb=False)
    assert out_clean.shape == (2, 1000)
    
    # 2. With Reverb
    # Use high direct_ratio to ensure the peak remains at the direct path (approx sample 500)
    out_reverb = mixer.spatialise_signal(signal, 0.0, add_reverb=True, rt60=0.2, direct_ratio=0.8)
    
    # Should be longer (signal + reverb tail)
    expected_min_len = 1000 + int(0.2 * fs)
    assert out_reverb.shape[1] >= expected_min_len
    
    # Check direct path is still dominant or present
    # Reverb usually adds 'tail', so peak should still be around sample 500 (+ delay)
    peak_idx = np.argmax(np.abs(out_reverb[0]))
    # Allow some shift, but shouldn't be at end
    # Direct path is at 500. Predelay is 50. Reverb peak might be around 550.
    assert 490 < peak_idx < 600

def test_mix_multiple():
    """Verify mixing multiple sources."""
    mixer = DataMixer(1000)
    s1 = np.ones((2, 100))
    s2 = np.ones((2, 200)) # Different length
    
    mixed = mixer.mix_multiple([s1, s2])
    
    assert mixed.shape == (2, 200)
    # First 100 samples should be s1 + s2 = 2
    # Normalization might scale this down, but let's check relative
    # If max is 2, it normalizes to 1.
    # So first half = 1.0, second half = 0.5
    
    # Check rough logic
    assert np.abs(mixed[0, 50]) > np.abs(mixed[0, 150])
