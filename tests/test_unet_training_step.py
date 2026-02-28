import torch
import torch.nn as nn
from src.models.unet import BioCPPNet
from src.models.dae import SpectrogramDAE
from src.models.losses import BioAcousticLoss
from src.spatial.beamforming import Beamformer

def test_unet_training_step():
    device = torch.device("cpu")
    n_fft = 1024
    hop_length = 512
    n_samples = 48000
    
    dae = SpectrogramDAE(n_fft=n_fft, hop_length=hop_length).to(device)
    dae.eval()
    for p in dae.parameters():
        p.requires_grad = False
        
    unet = BioCPPNet().to(device)
    unet.train()
    
    loss_fn = BioAcousticLoss(n_fft=n_fft, hop_length=hop_length)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    
    beamformer = Beamformer(sample_rate=48000)
    
    n_channels = beamformer.n_channels
    # Fake batch: (B, C, T) -> (1, n_channels, n_samples)
    noisy_mix = torch.randn(1, n_channels, n_samples).to(device)
    target_clean = torch.randn(1, 1, n_samples).to(device) # reference mic
    target_azimuth = 45.0
    
    # Simulate beamforming on CPU tensor using numpy
    beamformed = beamformer.delay_and_sum(noisy_mix[0].numpy(), azimuth_deg=target_azimuth)
    beamformed_tensor = torch.from_numpy(beamformed).float().unsqueeze(0).to(device)
    
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(
        beamformed_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    
    mag = torch.abs(stft)
    log_mag = torch.log1p(mag).unsqueeze(1) # (B, 1, F, T)
    
    denoised_log_mag = dae(log_mag)
    
    mask_logits = unet(denoised_log_mag)
    mask = torch.sigmoid(mask_logits)
    
    target_log_mag = denoised_log_mag * mask
    
    output_wav = dae.spectrogram_to_wav(target_log_mag, stft) # phase from STFT
    
    # We need to reshape output_wav to (B, C, T) to match target
    output_wav = output_wav.view(1, 1, -1)
    # Ensure same length
    min_len = min(output_wav.shape[-1], target_clean.shape[-1])
    output_wav = output_wav[..., :min_len]
    target_clean = target_clean[..., :min_len]
    
    loss = loss_fn(output_wav, target_clean)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that DAE weights didn't change (no grads)
    for p in dae.parameters():
        assert p.grad is None
        
    # Check UNet weights got grads
    got_grad = False
    for p in unet.parameters():
        if p.grad is not None:
            got_grad = True
            break
            
    assert got_grad
