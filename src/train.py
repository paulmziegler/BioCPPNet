import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import BioAcousticDataset
from src.models.dae import SpectrogramDAE
from src.utils import CONFIG, setup_logger, get_plot_path

logger = setup_logger("training")

def train():
    # 1. Load Config
    audio_cfg = CONFIG.get("audio", {})
    model_cfg = CONFIG.get("model", {})
    train_cfg = CONFIG.get("training", {})
    
    device = torch.device(train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Setup Dataset & Loader
    dataset = BioAcousticDataset(
        clean_files=[], # Empty list -> uses synthetic generation
        sample_rate=audio_cfg.get("sample_rate", 250000),
        duration=audio_cfg.get("duration", 1.0),
        n_channels=4 # From array config ideally
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=model_cfg.get("batch_size", 16),
        num_workers=0 # Windows usually needs 0 for simple debugging
    )
    
    # 3. Setup Model
    model = SpectrogramDAE(
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_length=audio_cfg.get("hop_length", 512)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.get("learning_rate", 0.001))
    loss_fn = nn.MSELoss()
    
    # 4. Training Loop
    epochs = model_cfg.get("epochs", 10)
    steps_per_epoch = train_cfg.get("steps_per_epoch", 100)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "results/checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Create iterator for this epoch
        iterator = iter(loader)
        
        # Progress bar for steps
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}", unit="step")
        
        for step in pbar:
            try:
                # Get batch: (B, C, T)
                noisy_wav, clean_wav = next(iterator)
            except StopIteration:
                break
                
            # Move to device
            noisy_wav = noisy_wav.to(device)
            clean_wav = clean_wav.to(device)
            
            # Reshape for DAE: Flatten channels into batch
            # (B, C, T) -> (B*C, T)
            b, c, t = noisy_wav.shape
            noisy_flat = noisy_wav.view(-1, t)
            clean_flat = clean_wav.view(-1, t)
            
            # Convert to Spectrogram: (B*C, 1, F, T_spec)
            noisy_spec = model.wav_to_spectrogram(noisy_flat)
            clean_spec = model.wav_to_spectrogram(clean_flat)
            
            # Forward
            output_spec = model(noisy_spec)
            
            # Loss
            loss = loss_fn(output_spec, clean_spec)
            
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
            path = os.path.join(checkpoint_dir, f"dae_epoch_{epoch}.pt")
            torch.save(model.state_dict(), path)
            logger.info(f"Saved checkpoint to {path}")

    # 5. Plot Training Progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history, marker='o', linestyle='-')
    plt.title("DAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Log-Spectrogram)")
    plt.grid(True)
    
    # Save plot to results folder
    plot_path = get_plot_path("training_loss_dae")
    plt.savefig(plot_path)
    logger.info(f"Training completed. Loss plot saved to {plot_path}")

if __name__ == "__main__":
    train()
