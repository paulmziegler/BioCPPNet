import numpy as np


def calculate_sisdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Clean target source signal.
        estimate: Estimated source signal.
        
    Returns:
        SI-SDR in dB.
    """
    eps = np.finfo(estimate.dtype).eps
    reference = reference.reshape(-1, 1)
    estimate = estimate.reshape(-1, 1)

    R_ss = np.dot(reference.T, reference)
    alpha = np.dot(estimate.T, reference) / (R_ss + eps)
    
    e_target = alpha * reference
    e_res = estimate - e_target
    
    S_target = np.sum(e_target ** 2)
    S_res = np.sum(e_res ** 2)
    
    return 10 * np.log10((S_target + eps) / (S_res + eps))
