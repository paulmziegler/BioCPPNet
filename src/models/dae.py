import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramDAE(nn.Module):
    """
    Convolutional Denoising Autoencoder for Spectrograms.
    Input: (Batch, 1, FreqBins, TimeFrames)
    Output: (Batch, 1, FreqBins, TimeFrames)
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # Downsample
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck (Latent)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Output Head
        self.dec1 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # No activation if target is log-magnitude (can be negative? No, log(1+x) >= 0).
        # Actually standard log-spec is usually normalized.
        # Let's use ReLU to ensure non-negativity if we assume input is > 0.
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input size to handle padding for odd dimensions
        # Autoencoders with strided convs can change dimensions slightly.
        # Ideally input dimensions are powers of 2.
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        
        # Crop or Pad to match input size exactly
        if out.shape != x.shape:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return F.relu(out) # Ensure non-negative output

    def wav_to_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Converts waveform (B, T) to Log-Magnitude Spectrogram (B, 1, F, T).
        """
        # Ensure wav is torch tensor
        window = torch.hann_window(self.n_fft, device=wav.device)
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        log_mag = torch.log1p(mag)
        return log_mag.unsqueeze(1) # Add channel dim

    def spectrogram_to_wav(self, log_mag: torch.Tensor, original_phase: torch.Tensor) -> torch.Tensor:
        """
        Converts Log-Magnitude Spectrogram back to waveform using original phase.
        """
        # Remove channel dim
        if log_mag.ndim == 4:
            log_mag = log_mag.squeeze(1)
            
        mag = torch.expm1(log_mag)
        stft_recon = mag * torch.exp(1j * torch.angle(original_phase))
        
        window = torch.hann_window(self.n_fft, device=log_mag.device)
        wav = torch.istft(stft_recon, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        return wav
