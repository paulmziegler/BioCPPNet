import numpy as np

from src.data_mixer import DataMixer
from src.utils import CONFIG


def test_mixer_multiple_sources():
    sample_rate = 48000
    mixer = DataMixer(sample_rate=sample_rate)
    
    n_samples = int(sample_rate * 0.1) # 100ms
    source1 = np.ones(n_samples)
    source2 = np.ones(n_samples) * -0.5
    
    # spatialise
    s1 = mixer.spatialise_signal(source1, azimuth_deg=45.0)
    s2 = mixer.spatialise_signal(source2, azimuth_deg=-30.0)
    
    mixed = mixer.mix_multiple([s1, s2])
    
    # Expected output: It should have the same number of channels and samples
    assert mixed.shape == s1.shape
    assert mixed.shape[1] == n_samples
    
    # Ensure it's not all zeros
    assert np.any(mixed)
    
    # Check max length padding if signals have different lengths
    source3 = np.ones(n_samples // 2)
    s3 = mixer.spatialise_signal(source3, azimuth_deg=0.0)
    mixed_diff_len = mixer.mix_multiple([s1, s3])
    
    assert mixed_diff_len.shape == s1.shape
