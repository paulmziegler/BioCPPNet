import pytest
import torch
import numpy as np
from src.data_loader import BioAcousticDataset

def test_dataset_generation():
    """Verify that the dataset yields tensors of correct shape."""
    fs = 10000 # Lower FS for speed
    duration = 0.1
    ds = BioAcousticDataset(clean_files=[], sample_rate=fs, duration=duration)
    
    # Get one batch
    iterator = iter(ds)
    noisy, clean = next(iterator)
    
    # Check types
    assert isinstance(noisy, torch.Tensor)
    assert isinstance(clean, torch.Tensor)
    assert noisy.dtype == torch.float32
    
    # Check shapes
    # Default mixer has 2 channels (from DataMixer fallback) or loaded config
    # We should check the channel dimension match
    n_samples = int(fs * duration)
    
    assert noisy.shape[-1] == n_samples
    assert clean.shape[-1] == n_samples
    assert noisy.shape[0] == clean.shape[0] # Channel count match
    
    # Check content (SNR)
    # Noisy should differ from clean
    assert not torch.allclose(noisy, clean)

def test_dataloader_integration():
    """Verify integration with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    fs = 10000
    ds = BioAcousticDataset(clean_files=[], sample_rate=fs, duration=0.1)
    loader = DataLoader(ds, batch_size=4)
    
    batch = next(iter(loader))
    noisy_batch, clean_batch = batch
    
    # Check batch dimensions
    assert noisy_batch.shape[0] == 4
    assert clean_batch.shape[0] == 4
    assert noisy_batch.ndim == 3 # (Batch, Channels, Time)
