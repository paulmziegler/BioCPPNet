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
from src.metrics.sisdr import calculate_sisdr
from src.utils import CONFIG, get_plot_path, setup_logger

logger = setup_logger("training")

def save_debug_spectrograms(epoch, mixture, clean, predicted, mask, out_dir):
    """Saves a plot of the spectrograms for visual debugging."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mixture (Beamformed + Denoised)
    im1 = axes[0, 0].imshow(np.log1p(mixture), aspect='auto', origin='lower', cmap='magma')
    axes[0, 0].set_title("DAE Output (Input to U-Net)")
    fig.colorbar(im1, ax=axes[0, 0])
    
    # 2. Clean Target
    im2 = axes[0, 1].imshow(np.log1p(clean), aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title("Ground Truth Target")
    fig.colorbar(im2, ax=axes[0, 1])
    
    # 3. Predicted Target
    im3 = axes[1, 0].imshow(np.log1p(predicted), aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title("U-Net Predicted Output")
    fig.colorbar(im3, ax=axes[1, 0])
    
    # 4. Raw Mask
    im4 = axes[1, 1].imshow(mask, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("U-Net Predicted Mask")
    fig.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_spectrograms.png"))
    plt.close(fig)

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
    # Ignore time-domain loss (phase errors) to allow magnitude mask to learn
    loss_fn = BioAcousticLoss(n_fft=n_fft, hop_length=hop_length, alpha_time=0.0)
    
    beamformer = Beamformer(sample_rate=sample_rate)
    window = torch.hann_window(n_fft, device=device)
    
    # 4. Generate Fixed Validation Set
    val_size = 128
    logger.info(f"Generating fixed validation set of {val_size} samples...")
    val_iterator = iter(dataset)
    val_data = []
    for _ in tqdm(range(val_size), desc="Gen Val Data"):
        val_data.append(next(val_iterator))
    
    # 5. Training Loop
    epochs = model_cfg.get("epochs", 10)
    steps_per_epoch = train_cfg.get("steps_per_epoch", 100)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "results/checkpoints")
    debug_dir = "results/debug"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting U-Net training...")
    history_loss = []
    history_val_sisdr = []
    
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
            
            # Compute STFT of clean target for direct magnitude loss
            clean_flat = clean_wav.view(batch_size, -1)
            target_stft = torch.stft(
                clean_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            clean_mag = torch.abs(target_stft) # (B, F, T)
            
            # Denoise (Frozen DAE)
            denoised_log_mag = dae(log_mag)
            
            # Forward U-Net
            mask_logits = unet(denoised_log_mag)
            mask = torch.sigmoid(mask_logits)
            
            # Apply Mask in Linear Domain
            # denoised_log_mag = log(1 + mag) -> mag = exp(denoised_log_mag) - 1
            denoised_linear_mag = torch.expm1(denoised_log_mag)
            target_linear_mag = denoised_linear_mag * mask
            
            # Reshape to match clean_mag (B, F, T)
            target_linear_mag = target_linear_mag.squeeze(1)
            
            # Ensure sequence lengths match (STFT frames might differ by 1)
            min_frames = min(target_linear_mag.shape[-1], clean_mag.shape[-1])
            target_linear_mag = target_linear_mag[..., :min_frames]
            clean_mag = clean_mag[..., :min_frames]
            
            # Direct Magnitude Loss
            # Bypass the loss_fn waveform wrapper to avoid ISTFT phase destruction
            loss = torch.mean(torch.abs(target_linear_mag - clean_mag))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_loss = epoch_loss / steps_per_epoch
        history_loss.append(avg_loss)
        
        # --- Validation Step ---
        unet.eval()
        val_sisdr_scores = []
        with torch.no_grad():
            for v_noisy, v_clean, v_az in val_data:
                # Need to add batch dimension
                v_noisy = v_noisy.unsqueeze(0).to(device)
                v_clean = v_clean.numpy() # keep clean as numpy for sisdr
                
                v_bf = beamformer.delay_and_sum(v_noisy[0].cpu().numpy(), azimuth_deg=v_az)
                v_bf_tensor = torch.from_numpy(v_bf).float().unsqueeze(0).unsqueeze(1).to(device)
                
                v_stft = torch.stft(v_bf_tensor.view(1, -1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
                v_mag = torch.abs(v_stft)
                v_log_mag = torch.log1p(v_mag).unsqueeze(1)
                
                v_denoised_log_mag = dae(v_log_mag)
                v_mask = torch.sigmoid(unet(v_denoised_log_mag))
                
                v_target_linear_mag = torch.expm1(v_denoised_log_mag) * v_mask
                v_target_log_mag = torch.log1p(v_target_linear_mag)
                
                v_out_wav = dae.spectrogram_to_wav(v_target_log_mag, v_stft)
                v_out_wav_np = v_out_wav.squeeze().cpu().numpy()
                
                min_len = min(len(v_out_wav_np), v_clean.shape[-1])
                score = calculate_sisdr(v_clean[0, :min_len], v_out_wav_np[:min_len])
                if not np.isinf(score) and not np.isnan(score):
                    val_sisdr_scores.append(score)
                    
            avg_val_sisdr = np.mean(val_sisdr_scores) if val_sisdr_scores else -100.0
            history_val_sisdr.append(avg_val_sisdr)
            
        logger.info(f"Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f} | Val SI-SDR: {avg_val_sisdr:.2f} dB")
        
        # Save Debug Spectrograms for the last sample of the validation set
        save_debug_spectrograms(
            epoch, 
            torch.expm1(v_denoised_log_mag).squeeze().cpu().numpy(),
            # Compute clean target mag for plotting
            torch.abs(torch.stft(torch.from_numpy(v_clean).float().to(device).view(1, -1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)).squeeze().cpu().numpy(),
            v_target_linear_mag.squeeze().cpu().numpy(),
            v_mask.squeeze().cpu().numpy(),
            debug_dir
        )
        
        # Save Checkpoint
        if epoch % train_cfg.get("save_interval", 5) == 0:
            path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pt")
            torch.save(unet.state_dict(), path)
            logger.info(f"Saved checkpoint to {path}")

    # 6. Plot Training Progress
    # Will be handled by plot_training.py which needs to be updated
    
if __name__ == "__main__":
    train()
