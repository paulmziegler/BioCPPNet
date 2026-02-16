import numpy as np
import scipy.signal


class ReverbGenerator:
    """
    Generates synthetic Room Impulse Responses (RIRs) and applies convolution.
    Approximates reverb using a stochastic model with exponential decay.
    """
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate

    def generate_rt60_envelope(self, duration: float, rt60: float) -> np.ndarray:
        """
        Generates an exponential decay envelope.
        RT60 is the time for amplitude to decay by 60dB.
        """
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        # 60dB decay means amplitude drops by factor of 1000 (10^(60/20)).
        # exp(-alpha * rt60) = 0.001
        # alpha = -ln(0.001) / rt60 = 6.908 / rt60
        alpha = 6.908 / rt60
        return np.exp(-alpha * t)

    def generate_stochastic_rir(
        self, num_channels: int, rt60: float = 0.5, direct_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Generates a multichannel RIR with diffuse late reverberation.

        Args:
            num_channels: Number of microphones in the array.
            rt60: Reverberation time in seconds.
            direct_ratio: Ratio of direct sound to reverb energy.

        Returns:
            (N_channels, N_samples) RIR.
        """
        # Duration of RIR usually slightly longer than RT60
        duration = rt60 * 1.2
        n_samples = int(duration * self.sample_rate)

        # 1. Generate Noise (Diffuse field assumption -> Independent gaussian)
        # For better realism, could impose spatial coherence,
        # but independent is a hard case for beamformers.
        noise = np.random.randn(num_channels, n_samples)

        # 2. Apply Decay Envelope
        envelope = self.generate_rt60_envelope(duration, rt60)
        late_reverb = noise * envelope

        # 3. Add Direct Path (Impulse at t=0)
        # Note: In a full pipeline, 'direct path' is handled by steering vectors.
        # This RIR is intended to be convolved *after* or *in addition to*
        # the direct path?
        # Standard approach: The RIR *contains* the direct path.
        # But our `DataMixer` already calculates precise sub-sample delays
        # for the direct path.
        # If we convolve with this RIR, we might double-up or smear the direct path.
        # STRATEGY: This generator will produce ONLY the "Tail" (Reverb).
        # The DataMixer will mix Direct + Reverb.
        
        # Normalize energy
        reverb_energy = np.mean(late_reverb ** 2)
        if reverb_energy > 0:
            late_reverb /= np.sqrt(reverb_energy)
            
        # Scale relative to direct (which is assumed unit energy 1.0)
        # direct_ratio 0.5 means reverb is 50% as loud as direct? Or DRR?
        # Let's assume amplitude scaling.
        late_reverb *= (1.0 - direct_ratio)
        
        # Apply a pre-delay to separation direct from late reverb
        # e.g., 5-10ms gap
        predelay_samples = int(0.005 * self.sample_rate)
        late_reverb[:, :predelay_samples] = 0
        
        return late_reverb

    def apply_reverb(self, signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """
        Convolves signal with RIR.
        
        Args:
            signal: (N_channels, N_samples) or (N_samples,)
            rir: (N_channels, N_rir_len)
            
        Returns:
            (N_channels, N_out)
        """
        # Handle 1D signal input (broadcast to channels?)
        # Typically signal is already spatialised (N_channels, N_samples).
        # We convolve each channel with its corresponding RIR channel.
        
        if signal.ndim == 1:
            # If mono input, we can produce multi-channel reverb output
            # (N_ch, N_rir) * (N_sig) -> (N_ch, N_out)
            output = []
            for ch in range(rir.shape[0]):
                # Using fftconvolve for speed
                out_ch = scipy.signal.fftconvolve(signal, rir[ch], mode='full')
                output.append(out_ch)
            return np.array(output)
            
        elif signal.ndim == 2:
            if signal.shape[0] != rir.shape[0]:
                raise ValueError("Signal and RIR channel counts must match")
            
            output = []
            for ch in range(signal.shape[0]):
                out_ch = scipy.signal.fftconvolve(signal[ch], rir[ch], mode='full')
                output.append(out_ch)
            return np.array(output)
            
        return signal
