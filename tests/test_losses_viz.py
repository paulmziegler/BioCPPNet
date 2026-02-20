import torch
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.losses import BioAcousticLoss
from src.utils import get_plot_path, setup_logger

logger = setup_logger("test_losses_viz")

def test_visualize_composite_loss():
    """
    Visualizes what the loss function 'sees' by plotting waveforms and spectrograms.
    """
    fs = 250000
    n_fft = 1024
    hop = 512
    
    # 1. Generate Signals
    t = torch.linspace(0, 0.05, int(0.05 * fs)) # 50ms
    # Target: 50kHz sine
    target_wav = torch.sin(2 * np.pi * 50000 * t).unsqueeze(0) # (1, T)
    
    # Estimate: Noisy + Phase Shifted
    # Phase shift by 10 samples (significant error)
    shift = 10
    est_wav = torch.roll(target_wav, shifts=shift, dims=-1)
    est_wav += torch.randn_like(est_wav) * 0.2 # Add noise
    
    # 2. Compute Loss
    loss_fn = BioAcousticLoss(n_fft=n_fft, hop_length=hop)
    total_loss = loss_fn(est_wav, target_wav)
    
    # Manually compute components for plotting
    loss_time = torch.mean(torch.abs(est_wav - target_wav)).item()
    
    window = torch.hann_window(n_fft)
    est_stft = torch.stft(est_wav, n_fft, hop, window=window, return_complex=True)
    target_stft = torch.stft(target_wav, n_fft, hop, window=window, return_complex=True)
    
    est_mag = torch.abs(est_stft)
    target_mag = torch.abs(target_stft)
    
    loss_mag = torch.mean(torch.abs(est_mag - target_mag)).item()
    
    # 3. Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Waveform
    t_ms = t.numpy() * 1000
    axes[0].plot(t_ms[:200], target_wav[0, :200].numpy(), label='Target', color='green')
    axes[0].plot(t_ms[:200], est_wav[0, :200].numpy(), label='Estimate (Shift+Noise)', color='red', alpha=0.7)
    axes[0].set_title(f"Time Domain (L1 Loss: {loss_time:.4f})")
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectrograms (Diff)
    # Plot abs difference
    diff_spec = torch.abs(target_mag - est_mag)[0].numpy()
    im = axes[1].imshow(diff_spec, origin='lower', aspect='auto', cmap='hot')
    axes[1].set_title(f"Spectrogram Error Magnitude (Mag Loss: {loss_mag:.4f})")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im, ax=axes[1])
    
    # Bar Chart of Components
    # Assuming weights 1.0, 1.0, 0.1
    components = ['Time', 'Mag', 'SpectralConv']
    values = [loss_time, loss_mag, total_loss.item() - loss_time - loss_mag] # Rough approximation
    
    axes[2].bar(components, values, color=['blue', 'orange', 'green'])
    axes[2].set_title(f"Loss Components (Total: {total_loss.item():.4f})")
    
    plt.tight_layout()
    save_path = get_plot_path("loss_visualization")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved Loss plot to {save_path}")
    
    assert total_loss > 0
