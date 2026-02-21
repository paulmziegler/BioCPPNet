import numpy as np
from scipy.fft import irfft, rfft, rfftfreq


def calculate_steering_vector(
    mic_positions: np.ndarray, source_direction: np.ndarray
) -> np.ndarray:
    """
    Calculates the relative path length differences for a plane wave
    arriving from a specific direction.

    Args:
        mic_positions: (N_mics, 3) array of microphone coordinates in meters.
        source_direction: (3,) unit vector pointing *towards* the source.

    Returns:
        (N_mics,) array of distance differences relative to the reference.
        Positive distance means the wavefront arrives *sooner*.
    """
    # Project mic positions onto the source vector
    return np.dot(mic_positions, source_direction)

def azimuth_elevation_to_vector(
    azimuth_deg: float, elevation_deg: float = 0.0
) -> np.ndarray:
    """
    Converts spherical coordinates to a unit vector.
    Coordinate system: X is forward (0 deg), Y is left (90 deg), Z is up.

    Args:
        azimuth_deg: Angle in degrees in the XY plane.
        elevation_deg: Angle in degrees from the XY plane towards Z.

    Returns:
        (3,) unit vector.
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    
    x = np.cos(az_rad) * np.cos(el_rad)
    y = np.sin(az_rad) * np.cos(el_rad)
    z = np.sin(el_rad)
    
    return np.array([x, y, z])

def apply_dynamic_subsample_shifts(
    signal: np.ndarray, delays_seconds: np.ndarray, sample_rate: int
) -> np.ndarray:
    """
    Applies precise time-varying sub-sample time delays to a signal using cubic interpolation.

    Args:
        signal: (N_samples,) 1D array representing the mono source signal.
        delays_seconds: (N_channels, N_samples) array of time-varying delays to apply.
        sample_rate: Sampling frequency in Hz.

    Returns:
        (N_channels, N_samples) array of shifted signals.
    """
    from scipy.interpolate import interp1d

    n_samples = signal.shape[-1]
    n_channels = delays_seconds.shape[0]

    # Original time axis
    t = np.arange(n_samples) / sample_rate

    # Create interpolator for the original signal
    interpolator = interp1d(t, signal, kind='cubic', bounds_error=False, fill_value=0.0)

    shifted_signals = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # We sample the original signal at t - delay.
        t_sample = t - delays_seconds[i]
        shifted_signals[i] = interpolator(t_sample)

    return shifted_signals

def apply_subsample_shifts(
    signal: np.ndarray, delays_seconds: np.ndarray, sample_rate: int
) -> np.ndarray:
    """
    Applies precise sub-sample time delays to a signal using FFT phase shifting.

    Args:
        signal: (N_samples,) or (N_channels, N_samples) array.
        delays_seconds: (N_channels,) array of delays to apply.
        sample_rate: Sampling frequency in Hz.

    Returns:
        (N_channels, N_samples) array of shifted signals.
    """
    n_samples = signal.shape[-1]

    # 1. FFT
    # rfft over last axis
    sig_fft = rfft(signal, axis=-1)
    freqs = rfftfreq(n_samples, d=1 / sample_rate)

    # 2. Compute Phase Shifts
    # (N_channels, N_freqs)
    phase_shifts = np.exp(-1j * 2 * np.pi * freqs[None, :] * delays_seconds[:, None])

    # 3. Apply Shift
    # If signal is 1D: sig_fft is (N_freqs,). Broadcasts to (N_channels, N_freqs).
    # If signal is 2D: sig_fft is (N_channels, N_freqs). Element-wise mult.
    shifted_fft = sig_fft * phase_shifts

    # 4. IFFT
    shifted_signals = irfft(shifted_fft, n=n_samples, axis=-1)

    return shifted_signals
