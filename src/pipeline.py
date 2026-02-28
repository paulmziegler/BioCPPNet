import numpy as np
import torch

from src.models.dae import SpectrogramDAE
from src.models.unet import BioCPPNet
from src.spatial.beamforming import Beamformer
from src.utils import CONFIG


class BioCPPNetPipeline:
    """
    End-to-End pipeline for BioCPPNet.
    Integrates Beamforming -> Denoising Autoencoder (DAE) -> U-Net Source Separation.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        audio_config = CONFIG.get("audio", {})
        self.sample_rate = audio_config.get("sample_rate", 250000)
        self.n_fft = audio_config.get("n_fft", 1024)
        self.hop_length = audio_config.get("hop_length", 512)
        
        # Initialize components
        self.beamformer = Beamformer(sample_rate=self.sample_rate)
        
        self.dae = SpectrogramDAE(n_fft=self.n_fft, hop_length=self.hop_length).to(self.device)
        self.dae.eval() # Set to evaluation mode
        
        # BioCPPNet uses default in_channels=1, out_channels=1
        self.unet = BioCPPNet().to(self.device)
        self.unet.eval() # Set to evaluation mode

    def load_weights(self, dae_path: str = None, unet_path: str = None):
        """Loads pre-trained weights for the models."""
        if dae_path:
            self.dae.load_state_dict(torch.load(dae_path, map_location=self.device))
        if unet_path:
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))

    @torch.no_grad()
    def process(self, multichannel_audio: np.ndarray, azimuth_deg: float | list[float] = None, elevation_deg: float = 0.0, num_sources: int = 1) -> np.ndarray | list[np.ndarray]:
        """
        Process multichannel audio through the end-to-end pipeline.
        
        Args:
            multichannel_audio: Numpy array of shape (Time, Channels) or (Channels, Time).
            azimuth_deg: Target azimuth in degrees, or list of azimuths. If None, it will be estimated.
            elevation_deg: Target elevation in degrees.
            num_sources: The number of sources to separate if azimuth_deg is None.
            
        Returns:
            Numpy array of shape (Time,) containing the separated target source, 
            or a list of such arrays if multiple sources are requested.
        """
        # 1. Input Normalization
        if multichannel_audio.ndim == 2:
            # If (Time, Channels), transpose to (Channels, Time)
            if multichannel_audio.shape[1] < multichannel_audio.shape[0]:
                multichannel_audio = multichannel_audio.T
                
        # 2. Spatial Filtering (Beamforming)
        if azimuth_deg is None:
            azimuth_deg = self.beamformer.estimate_doa(multichannel_audio, num_sources=num_sources)
            
        azimuths = azimuth_deg if isinstance(azimuth_deg, list) else [azimuth_deg]
        
        output_signals = []
        
        for az in azimuths:
            beamformed_signal = self.beamformer.delay_and_sum(
                multichannel_audio, 
                azimuth_deg=az, 
                elevation_deg=elevation_deg
            )
            
            # Convert to PyTorch tensor and add batch dimension: (1, Time)
            wav_tensor = torch.from_numpy(beamformed_signal).float().unsqueeze(0).to(self.device)
            
            # 3. Time-Frequency Transform (STFT)
            window = torch.hann_window(self.n_fft, device=self.device)
            stft = torch.stft(
                wav_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            
            # Extract magnitude and phase
            mag = torch.abs(stft)
            log_mag = torch.log1p(mag).unsqueeze(1)  # (Batch=1, Channels=1, FreqBins, TimeFrames)
            phase = stft # original complex stft contains the phase
            
            # 4. Denoising (DAE)
            denoised_log_mag = self.dae(log_mag)
            
            # 5. Source Separation (U-Net)
            mask_logits = self.unet(denoised_log_mag)
            mask = torch.sigmoid(mask_logits)
            
            # Apply Mask in Linear Domain
            denoised_linear_mag = torch.expm1(denoised_log_mag)
            target_linear_mag = denoised_linear_mag * mask
            
            # Convert back to log-magnitude for ISTFT
            target_log_mag = torch.log1p(target_linear_mag)
            
            # 6. Signal Reconstruction (ISTFT)
            output_wav = self.dae.spectrogram_to_wav(target_log_mag, phase)
            output_signals.append(output_wav.squeeze().cpu().numpy())
            
        if isinstance(azimuth_deg, list) or num_sources > 1:
            return output_signals
        return output_signals[0]
