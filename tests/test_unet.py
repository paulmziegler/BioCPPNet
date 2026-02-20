import torch
import pytest
from src.models.unet import BioCPPNet
from src.models.losses import BioAcousticLoss

def test_unet_shapes():
    """Verify U-Net input/output shapes match."""
    model = BioCPPNet(in_channels=1, out_channels=1)
    # Power of 2
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out.shape == x.shape

def test_unet_odd_shapes():
    """Verify U-Net handles odd spatial dimensions via interpolation."""
    model = BioCPPNet(in_channels=1, out_channels=1)
    # Odd dimensions
    x = torch.randn(2, 1, 257, 101)
    out = model(x)
    assert out.shape == x.shape

def test_loss_function():
    """Verify BioAcousticLoss computation."""
    loss_fn = BioAcousticLoss(alpha_time=1.0, alpha_mag=1.0, alpha_sc=1.0)
    
    # Dummy waveforms
    est = torch.randn(2, 1000)
    target = torch.randn(2, 1000)
    
    loss = loss_fn(est, target)
    
    assert loss > 0
    assert not torch.isnan(loss)
    
    # Zero loss for identical
    loss_zero = loss_fn(target, target)
    assert loss_zero < 1e-6

def test_unet_learning():
    """Verify U-Net can learn (simple overfit)."""
    model = BioCPPNet(in_channels=1, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss() # Simple MSE on spectrogram for this test
    
    # Spectrogram target
    target = torch.randn(1, 1, 64, 64)
    input_sig = torch.randn(1, 1, 64, 64)
    
    model.train()
    initial_loss = loss_fn(model(input_sig), target).item()
    
    for _ in range(10):
        optimizer.zero_grad()
        out = model(input_sig)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
    final_loss = loss_fn(model(input_sig), target).item()
    
    assert final_loss < initial_loss
