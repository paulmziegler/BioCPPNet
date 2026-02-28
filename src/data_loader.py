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
        return_geometry: bool = False,
        num_interferers: int = 1
    ):
        super().__init__()
        self.clean_files = clean_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_channels = n_channels
        self.return_geometry = return_geometry
        self.num_interferers = num_interferers
        
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Infinite iterator for DataLoader.
        """
        
        while True:
            # 1. Get Source (Target)
            clean_mono_target = self._load_source()

            # 2. Pick Random Location for target
            target_azimuth = np.random.uniform(0, 180) # Frontal semi-circle
            
            # 3. Spatialise Target
            # We want the clean target as it appears at the reference mic (mono)
            target_multichannel = self.mixer.spatialise_signal(
                clean_mono_target, target_azimuth, add_reverb=False
            )
            # Reference mic is channel 0
            clean_reference_target = target_multichannel[0:1, :]
            
            # 4. Generate Target with Reverb
            reverberant_target = self.mixer.spatialise_signal(
                clean_mono_target, target_azimuth, add_reverb=True, rt60=np.random.uniform(0.1, 0.5)
            )
            
            sources_to_mix = [reverberant_target]
            
            # 5. Generate Interferers
            for _ in range(self.num_interferers):
                interferer_mono = self._load_source()
                # Ensure interferer is somewhat separated spatially
                interferer_azimuth = np.random.uniform(0, 180)
                reverberant_interferer = self.mixer.spatialise_signal(
                    interferer_mono, interferer_azimuth, add_reverb=True, rt60=np.random.uniform(0.1, 0.5)
                )
                
                # Scale interferer for SNR
                # Target is reference. Let's make interferer SNR randomly between 0 to 10 dB lower or higher
                snr_db = np.random.uniform(-5, 5)
                p_target = np.mean(reverberant_target**2)
                p_interferer = np.mean(reverberant_interferer**2)
                if p_interferer > 0:
                    scale = np.sqrt(p_target / (p_interferer * (10**(snr_db/10))))
                    reverberant_interferer *= scale
                    
                sources_to_mix.append(reverberant_interferer)
                
            # 6. Mix all sources
            mixture = self.mixer.mix_multiple(sources_to_mix)

            # Crop or pad mixture and target to maintain fixed length for batching
            if mixture.shape[1] > self.n_samples:
                mixture = mixture[:, : self.n_samples]
            elif mixture.shape[1] < self.n_samples:
                mixture = np.pad(mixture, ((0, 0), (0, self.n_samples - mixture.shape[1])))
                
            if clean_reference_target.shape[1] > self.n_samples:
                clean_reference_target = clean_reference_target[:, : self.n_samples]
            elif clean_reference_target.shape[1] < self.n_samples:
                clean_reference_target = np.pad(clean_reference_target, ((0, 0), (0, self.n_samples - clean_reference_target.shape[1])))

            # Add Environmental Noise (Wind/Thermal)
            snr_noise = np.random.uniform(10, 20)
            noisy_mixture = self.mixer.add_noise(
                mixture, noise_type='pink', snr_db=snr_noise
            )
            
            # 5. Convert to Tensor
            input_tensor = torch.from_numpy(noisy_mixture).float()
            target_tensor = torch.from_numpy(clean_reference_target).float()
            
            yield input_tensor, target_tensor, target_azimuth
