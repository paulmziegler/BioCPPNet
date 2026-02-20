from typing import Iterator, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from src.data_mixer import DataMixer


class BioAcousticDataset(IterableDataset):
    """
    A PyTorch IterableDataset that generates bioacoustic training data on-the-fly.
    
    It combines:
    1. Clean source loading (from disk or synthetic generation).
    2. Spatial mixing (using DataMixer).
    3. Noise injection.
    4. Reverberation.
    
    Yields:
        (noisy_mixture, clean_target) tuples of torch.Tensors.
    """
    def __init__(
        self, 
        clean_files: List[str] = [],
        sample_rate: int = 250000,
        duration: float = 1.0,
        n_channels: int = 4,
        return_geometry: bool = False
    ):
        super().__init__()
        self.clean_files = clean_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_channels = n_channels
        self.return_geometry = return_geometry
        
        # Initialize DataMixer
        self.mixer = DataMixer(sample_rate=sample_rate)
        
                # Update mixer geometry if needed
        # (assuming linear array for now or loaded from config)
        # Ideally, DataMixer loads config automatically.

    def _generate_synthetic_source(self) -> np.ndarray:
        """Generates a random chirp or pulse if no files are provided."""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

        # Random parameters
        freq_start = np.random.uniform(20000, 40000)
        freq_end = np.random.uniform(40000, 80000)

        # Linear Chirp
        signal = scipy.signal.chirp(
            t, f0=freq_start, f1=freq_end, t1=self.duration, method="linear"
        )

        # Apply envelope
        envelope = np.hanning(self.n_samples)
        return signal * envelope

    def _load_source(self) -> np.ndarray:
        """Loads a random file from the list or generates synthetic."""
        if not self.clean_files:
            return self._generate_synthetic_source()
            
        # Pick random file
        filepath = np.random.choice(self.clean_files)
        
        try:
            # Load with soundfile
            # Note: Librosa is slower, soundfile is faster for reading
            # We assume mono files.
            data, sr = sf.read(filepath)
            
            # Resample if needed (skip for now to keep simple, assuming pre-processed)
            if sr != self.sample_rate:
                # Placeholder for resampling logic
                pass
                
            # Random crop or pad
            if len(data) > self.n_samples:
                start = np.random.randint(0, len(data) - self.n_samples)
                data = data[start:start+self.n_samples]
            else:
                data = np.pad(data, (0, self.n_samples - len(data)))
                
            return data
            
        except Exception as e:
            # Fallback
            print(f"Error loading {filepath}: {e}")
            return self._generate_synthetic_source()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Infinite iterator for DataLoader.
        """
        
        while True:
            # 1. Get Source (Target)
            if self.clean_files:
                clean_mono = self._load_source()
            else:
                clean_mono = self._generate_synthetic_source()

            # 2. Pick Random Location
            azimuth = np.random.uniform(0, 180) # Frontal semi-circle
            
            # 3. Spatialise Target (Clean Reference)
            # We want the clean target as it appears at the array (or at reference mic)
            clean_multichannel = self.mixer.spatialise_signal(
                clean_mono, azimuth, add_reverb=False
            )
            
            # 4. Generate & Add Interference (Noise)
            # Scenario: Target + Pink Noise + Reverb
            
            # Apply reverb to the "clean" path for the mixture? 
            # Usually: 
            # Mixture = Source * RIR + Noise
            # Target = Source * Direct_Path (or Reference Mic)
            
            # Let's add reverb to the mixture component
            reverberant_source = self.mixer.spatialise_signal(
                clean_mono, azimuth, add_reverb=True, rt60=np.random.uniform(0.1, 0.5)
            )
            
            # Crop reverb tail to maintain fixed length for batching
            if reverberant_source.shape[1] > self.n_samples:
                reverberant_source = reverberant_source[:, : self.n_samples]
            elif reverberant_source.shape[1] < self.n_samples:
                reverberant_source = np.pad(
                    reverberant_source,
                    ((0, 0), (0, self.n_samples - reverberant_source.shape[1])),
                )

            # Add Environmental Noise (Wind/Thermal)
            # Random SNR between 0 and 20 dB
            snr = np.random.uniform(0, 20)
            noisy_mixture = self.mixer.add_noise(
                reverberant_source, noise_type='pink', snr_db=snr
            )
            
            # 5. Convert to Tensor
            # Shape: (Channels, Time)
            input_tensor = torch.from_numpy(noisy_mixture).float()
            
            # Target: Usually we want to predict the clean direct path
            # at the reference mic (Ch0)
            # Shape: (1, Time) or (Channels, Time) depending on task
            # (mask estimation vs beamforming)
            # Let's return the full clean spatialised signal (Direct path only)

            # Ensure target is also n_samples (DataMixer delays might shift it?)
            # spatialise_signal (no reverb) keeps length roughly same but let's be safe
            if clean_multichannel.shape[1] > self.n_samples:
                clean_multichannel = clean_multichannel[:, : self.n_samples]
            elif clean_multichannel.shape[1] < self.n_samples:
                clean_multichannel = np.pad(
                    clean_multichannel,
                    ((0, 0), (0, self.n_samples - clean_multichannel.shape[1])),
                )

            target_tensor = torch.from_numpy(clean_multichannel).float()
            yield input_tensor, target_tensor
