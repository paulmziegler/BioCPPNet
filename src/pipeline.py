import numpy as np
import torch
import torch.nn.functional as F

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
    def process(self, multichannel_audio: np.ndarray, azimuth_deg: float = None, elevation_deg: float = 0.0) -> np.ndarray:
        """
        Process multichannel audio through the end-to-end pipeline.
        
        Args:
            multichannel_audio: Numpy array of shape (Time, Channels) or (Channels, Time).
            azimuth_deg: Target azimuth in degrees. If None, it will be estimated.
            elevation_deg: Target elevation in degrees.
            
        Returns:
            Numpy array of shape (Time,) containing the separated target source.
        """
        # 1. Input Normalization
        if multichannel_audio.ndim == 2:
            # If (Time, Channels), transpose to (Channels, Time)
            if multichannel_audio.shape[1] < multichannel_audio.shape[0]:
                multichannel_audio = multichannel_audio.T
                
        # 2. Spatial Filtering (Beamforming)
        if azimuth_deg is None:
            azimuth_deg = self.beamformer.estimate_doa(multichannel_audio)
            
        beamformed_signal = self.beamformer.delay_and_sum(
            multichannel_audio, 
            azimuth_deg=azimuth_deg, 
            elevation_deg=elevation_deg
        )
        
        # Convert to PyTorch tensor and add batch dimension: (1, Time)
        wav_tensor = torch.from_numpy(beamformed_signal).float().unsqueeze(0).to(self.device)
        
        # 3. Time-Frequency Transform (STFT)
        # Using the DAE's internal wav_to_spectrogram would give us log_mag, 
        # but we need the phase for reconstruction. So we compute STFT directly.
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
        # U-Net outputs a mask (logits), we apply sigmoid to get values in [0, 1]
        mask_logits = self.unet(denoised_log_mag)
        mask = torch.sigmoid(mask_logits)
        
        # Apply mask to the denoised magnitude
        target_log_mag = denoised_log_mag * mask
        
        # 6. Signal Reconstruction (ISTFT)
        # DAE.spectrogram_to_wav handles the ISTFT with original phase
        output_wav = self.dae.spectrogram_to_wav(target_log_mag, phase)
        
        # Convert back to numpy array (Time,)
        return output_wav.squeeze().cpu().numpy()
