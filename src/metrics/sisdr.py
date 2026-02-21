import numpy as np


def calculate_sisdr(ref: np.ndarray, est: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        ref: Ground truth reference signal (1D Numpy array).
        est: Estimated signal (1D Numpy array).
        
    Returns:
        SI-SDR value in decibels (dB).
    """
    # Ensure inputs are 1D
    ref = np.squeeze(ref)
    est = np.squeeze(est)
    
    # Check lengths
    if ref.shape != est.shape:
        raise ValueError("Reference and estimated signals must have the same length.")
        
    # Remove DC offset
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    
    # Compute power of ref
    ref_energy = np.sum(ref**2)
    if ref_energy == 0:
        return np.inf if np.all(est == 0) else -np.inf
        
    # Projection of estimated signal onto reference signal
    alpha = np.sum(est * ref) / ref_energy
    
    # Target signal
    target = alpha * ref
    
    # Noise/distortion signal
    e_noise = est - target
    
    # Compute energies
    target_energy = np.sum(target**2)
    noise_energy = np.sum(e_noise**2)
    
    if noise_energy == 0:
        return np.inf
        
    # SI-SDR in dB
    sisdr = 10 * np.log10(target_energy / noise_energy)
    
    return float(sisdr)
