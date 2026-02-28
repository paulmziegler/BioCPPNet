import torch
from src.data_loader import BioAcousticDataset

def test_online_dataset_multisource():
    dataset = BioAcousticDataset(
        clean_files=[], 
        sample_rate=48000, 
        duration=0.1, 
        n_channels=4,
        num_interferers=2
    )
    
    iterator = iter(dataset)
    noisy_mix, target_clean, azimuth = next(iterator)
    
    # Check shapes
    n_samples = int(48000 * 0.1)
    
    assert noisy_mix.shape[0] == dataset.mixer.mic_positions.shape[0]  # n_channels
    assert noisy_mix.shape[1] == n_samples
    
    assert target_clean.shape[0] == 1  # reference channel or full channels? 
    # Usually we want the target to be a mono signal (e.g. at reference mic) for U-Net
    
    assert isinstance(azimuth, float)
    assert 0.0 <= azimuth <= 360.0 or -180.0 <= azimuth <= 180.0
