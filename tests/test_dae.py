import torch
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.dae import SpectrogramDAE
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_dae")

def test_dae_shapes():
    """Verify input/output shapes match."""
    model = SpectrogramDAE()
    # Dummy spectrogram: (Batch, 1, Freq, Time)
    # Freq = n_fft/2 + 1 = 513
    # Time = e.g., 100
    x = torch.randn(2, 1, 513, 128) 
    
    # Ensure positive input (like magnitude)
    x = torch.abs(x)
    
    out = model(x)
    
    assert out.shape == x.shape
    assert torch.all(out >= 0) # ReLU output

def test_dae_learning_capability():
    """
    Verify the model can learn to reconstruct a simple pattern (overfit test).
    This proves the architecture allows gradient flow.
    """
    model = SpectrogramDAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    # Target: A simple horizontal line (constant frequency tone)
    target = torch.zeros(1, 1, 513, 64)
    target[:, :, 100:110, :] = 1.0 # Band at bin 100
    
    # Input: Target + Noise
    noise = torch.randn_like(target) * 0.1
    input_sig = torch.clamp(target + noise, min=0.0)
    
    # Train for a few steps
    initial_loss = 0.0
    final_loss = 0.0
    
    model.train()
    for i in range(50):
        optimizer.zero_grad()
        output = model(input_sig)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    logger.info(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")
    
    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, "Model failed to learn simple reconstruction"
    assert final_loss < 0.05, f"Final loss {final_loss} is too high for simple overfit"

    # Visualization
    model.eval()
    with torch.no_grad():
        final_out = model(input_sig)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_sig[0, 0].numpy(), origin='lower', aspect='auto')
    axes[0].set_title("Noisy Input")
    
    axes[1].imshow(target[0, 0].numpy(), origin='lower', aspect='auto')
    axes[1].set_title("Clean Target")
    
    axes[2].imshow(final_out[0, 0].numpy(), origin='lower', aspect='auto')
    axes[2].set_title("Denoised Output")
    
    plt.tight_layout()
    save_path = get_plot_path("dae_learning_viz")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved DAE plot to {save_path}")

def test_stft_helper_shapes():
    """Verify wav_to_spectrogram and inverse."""
    model = SpectrogramDAE(n_fft=512, hop_length=256)
    wav = torch.randn(2, 10000) # (Batch, Samples)
    
    # 1. To Spectrogram
    spec = model.wav_to_spectrogram(wav)
    # Shape check: (B, 1, n_fft/2+1, Time)
    assert spec.ndim == 4
    assert spec.shape[2] == 257 # 512/2 + 1
    
    # 2. Get Phase (need actual STFT for this test usually, but helper assumes internal consistency)
    # We need phase to invert. Helper doesn't return phase.
    # Let's verify we can get phase manually if needed.
    window = torch.hann_window(512)
    stft = torch.stft(wav, n_fft=512, hop_length=256, window=window, return_complex=True)
    
    # 3. Invert using helper
    rec_wav = model.spectrogram_to_wav(spec, stft) # Pass complex STFT as 'phase' source
    
    # Length might differ slightly due to padding/windowing
    # STFT/ISTFT usually pads.
    assert abs(rec_wav.shape[1] - wav.shape[1]) < 512
