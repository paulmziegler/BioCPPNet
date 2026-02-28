import numpy as np
import torch
from src.pipeline import BioCPPNetPipeline

def test_pipeline_unet_active():
    pipeline = BioCPPNetPipeline(device="cpu")
    # For testing, we don't care about the quality, just that the shapes match and U-Net is executed
    
    sample_rate = 48000
    n_samples = int(sample_rate * 0.1) # 100ms
    n_channels = pipeline.beamformer.n_channels
    
    # Fake multichannel input
    multichannel_audio = np.random.randn(n_channels, n_samples)
    
    # Process a single source
    out_single = pipeline.process(multichannel_audio, num_sources=1)
    
    assert isinstance(out_single, np.ndarray)
    assert out_single.ndim == 1
    # Check length roughly matches (STFT/ISTFT might change it slightly)
    assert abs(len(out_single) - n_samples) < 2048
    
    # Process multiple sources
    out_multi = pipeline.process(multichannel_audio, num_sources=2)
    
    assert isinstance(out_multi, list)
    assert len(out_multi) == 2
    assert out_multi[0].ndim == 1
    assert out_multi[1].ndim == 1
