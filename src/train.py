import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import BioAcousticDataset
from src.models.dae import SpectrogramDAE
from src.models.unet import BioCPPNet
from src.models.losses import BioAcousticLoss
from src.spatial.beamforming import Beamformer
from src.utils import CONFIG, get_plot_path, setup_logger

logger = setup_logger("training")

def train():
    # 1. Load Config
    audio_cfg = CONFIG.get("audio", {})
    model_cfg = CONFIG.get("model", {})
    train_cfg = CONFIG.get("training", {})

    device = torch.device(
        train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 2. Setup Dataset & Loader
    sample_rate = audio_cfg.get("sample_rate", 250000)
    dataset = BioAcousticDataset(
        clean_files=[], # Empty list -> uses synthetic generation
        sample_rate=sample_rate,
        duration=audio_cfg.get("duration", 1.0),
        n_channels=4, # From array config ideally
        num_interferers=1
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=model_cfg.get("batch_size", 16),
        num_workers=0 # Windows usually needs 0 for simple debugging
    )
    
    n_fft = audio_cfg.get("n_fft", 1024)
    hop_length = audio_cfg.get("hop_length", 512)
    
    # 3. Setup Models
    # Load and freeze DAE
    dae = SpectrogramDAE(n_fft=n_fft, hop_length=hop_length).to(device)
    dae_path = model_cfg.get("dae_weights_path", None)
    if dae_path and os.path.exists(dae_path):
        dae.load_state_dict(torch.load(dae_path, map_location=device))
        logger.info(f"Loaded DAE weights from {dae_path}")
    dae.eval()
    for p in dae.parameters():
        p.requires_grad = False
        
    # Setup U-Net
    unet = BioCPPNet().to(device)
    unet.train()

    optimizer = torch.optim.Adam(
        unet.parameters(), lr=model_cfg.get("learning_rate", 0.001)
    )
    loss_fn = BioAcousticLoss(n_fft=n_fft, hop_length=hop_length)
    
    beamformer = Beamformer(sample_rate=sample_rate)
    window = torch.hann_window(n_fft, device=device)
    
    # 4. Training Loop
    epochs = model_cfg.get("epochs", 10)
    steps_per_epoch = train_cfg.get("steps_per_epoch", 100)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "results/checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting U-Net training...")
    history = []
    
    for epoch in range(1, epochs + 1):
        unet.train()
        epoch_loss = 0.0
        
        # Create iterator for this epoch
        iterator = iter(loader)
        
        # Progress bar for steps
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}", unit="step")
        
        for step in pbar:
            try:
                # Get batch: (B, C, T)
                noisy_wav, clean_wav, azimuths = next(iterator)
            except StopIteration:
                break
                
            batch_size = noisy_wav.shape[0]
            
            # Move to device
            noisy_wav = noisy_wav.to(device)
            clean_wav = clean_wav.to(device)
            
            # Beamforming has to happen per-sample because azimuths differ
            beamformed_signals = []
            noisy_wav_cpu = noisy_wav.cpu().numpy()
            
            for b in range(batch_size):
                b_sig = beamformer.delay_and_sum(noisy_wav_cpu[b], azimuth_deg=azimuths[b].item())
                beamformed_signals.append(b_sig)
                
            beamformed_tensor = torch.from_numpy(np.array(beamformed_signals)).float().to(device)
            # Add channel dimension: (B, 1, T)
            beamformed_tensor = beamformed_tensor.unsqueeze(1)
            
            # Compute STFT
            # Reshape for STFT: (B, T)
            bf_flat = beamformed_tensor.view(batch_size, -1)
            
            stft = torch.stft(
                bf_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            
            mag = torch.abs(stft)
            log_mag = torch.log1p(mag).unsqueeze(1)  # (B, 1, F, T)
            
            # Denoise (Frozen DAE)
            denoised_log_mag = dae(log_mag)
            
            # Forward U-Net
            mask_logits = unet(denoised_log_mag)
            mask = torch.sigmoid(mask_logits)
            
            # Apply Mask
            target_log_mag = denoised_log_mag * mask
            
            # Reconstruct ISTFT
            # Provide phase from original STFT
            # dae.spectrogram_to_wav expects (B*C, 1, F, T) and phase (B*C, F, T)
            output_wav = dae.spectrogram_to_wav(target_log_mag, stft)
            
            # Reshape to (B, 1, T) to match target_clean
            output_wav = output_wav.view(batch_size, 1, -1)
            
            # Ensure sequence lengths match due to ISTFT padding
            min_len = min(output_wav.shape[-1], clean_wav.shape[-1])
            output_wav = output_wav[..., :min_len]
            clean_wav = clean_wav[..., :min_len]
            
            # Loss
            loss = loss_fn(output_wav, clean_wav)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_loss = epoch_loss / steps_per_epoch
        history.append(avg_loss)
        logger.info(f"Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if epoch % train_cfg.get("save_interval", 5) == 0:
            path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pt")
            torch.save(unet.state_dict(), path)
            logger.info(f"Saved checkpoint to {path}")

    # 5. Plot Training Progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history, marker='o', linestyle='-')
    plt.title("U-Net Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.grid(True)
    
    # Save plot to results folder
    plot_path = get_plot_path("training_loss_unet")
    plt.savefig(plot_path)
    logger.info(f"Training completed. Loss plot saved to {plot_path}")

if __name__ == "__main__":
    train()
