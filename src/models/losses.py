import torch
import torch.nn as nn


class BioAcousticLoss(nn.Module):
    """
    Composite loss function for Bioacoustic Source Separation.
    Combines:
    1. L1 Waveform Loss (Time Domain)
    2. STFT Magnitude Loss (Frequency Domain)
    3. Spectral Convergence Loss
    """
    def __init__(self, 
                 alpha_time: float = 1.0, 
                 alpha_mag: float = 1.0, 
                 alpha_sc: float = 0.1,
                 n_fft: int = 1024,
                 hop_length: int = 512):
        super().__init__()
        self.alpha_time = alpha_time
        self.alpha_mag = alpha_mag
        self.alpha_sc = alpha_sc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def forward(self, est_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est_wav: (B, T) estimated waveform.
            target_wav: (B, T) target waveform.
        """
        # 1. Time Domain Loss (L1)
        loss_time = torch.mean(torch.abs(est_wav - target_wav))
        
        if self.alpha_mag == 0 and self.alpha_sc == 0:
            return loss_time * self.alpha_time
            
        # 2. Compute STFT
        # Ensure window is on correct device
        if self.window.device != est_wav.device:
            self.window = self.window.to(est_wav.device)
            
        est_stft = torch.stft(
            est_wav,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_wav,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )

        est_mag = torch.abs(est_stft)
        target_mag = torch.abs(target_stft)

        # 3. Magnitude Loss (L1)
        loss_mag = torch.mean(torch.abs(est_mag - target_mag))

        # 4. Spectral Convergence Loss
        # Frobenius norm = sqrt(sum(x^2)).
        # sc_loss = || |target| - |est| ||_F / || |target| ||_F
        # Use torch.linalg.norm or flattened L2

        diff_norm = torch.linalg.norm(
            target_mag - est_mag, ord="fro", dim=(1, 2)
        )  # Norm over F and T
        target_norm = torch.linalg.norm(target_mag, ord="fro", dim=(1, 2))
        
        # Avoid div by zero
        loss_sc = torch.mean(diff_norm / (target_norm + 1e-7))
        
        total_loss = (
            (self.alpha_time * loss_time)
            + (self.alpha_mag * loss_mag)
            + (self.alpha_sc * loss_sc)
        )

        return total_loss
