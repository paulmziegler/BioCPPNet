import os
import gradio as gr
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import torch

from src.pipeline import BioCPPNetPipeline
from src.utils import CONFIG

# Initialize the pipeline
print("Initializing BioCPPNet Pipeline for Gradio...")
pipeline = BioCPPNetPipeline()

# Try to load pre-trained weights if they exist
checkpoint_path = "results/checkpoints/dae_epoch_50.pt"
if os.path.exists(checkpoint_path):
    print(f"Loading trained weights from {checkpoint_path}...")
    pipeline.load_weights(dae_path=checkpoint_path)
else:
    print("Warning: No trained weights found. Using random initialization.")

def plot_spectrogram(signal, sample_rate, title="Spectrogram"):
    """Helper function to create a spectrogram image."""
    plt.figure(figsize=(10, 4))
    
    # Calculate STFT
    n_fft = pipeline.n_fft
    hop_length = pipeline.hop_length
    
    # We use PyTorch to get the spectrogram just like the model does
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        torch.tensor(signal).unsqueeze(0).float(),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    mag = torch.abs(stft).squeeze().numpy()
    log_mag = np.log1p(mag)
    
    plt.imshow(log_mag, aspect='auto', origin='lower', cmap='magma', 
               extent=[0, len(signal)/sample_rate, 0, sample_rate/2])
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def process_audio(audio_file, target_azimuth):
    """Gradio interface function."""
    if audio_file is None:
        return None, None, None, "Please upload a valid multichannel audio file."
        
    try:
        # Load audio using soundfile (gradio provides filepath)
        # We need multichannel data, so we don't convert to mono
        audio_data, sr = sf.read(audio_file)
        
        if sr != pipeline.sample_rate:
            return None, None, None, f"Error: Audio sample rate ({sr} Hz) does not match pipeline ({pipeline.sample_rate} Hz)."
            
        if audio_data.ndim == 1:
            return None, None, None, "Error: Uploaded file is mono. Please upload a multichannel file."
            
        # Run pipeline
        output_signal = pipeline.process(audio_data, azimuth_deg=float(target_azimuth))
        
        # We need to save the output temporarily for Gradio to play it back
        output_path = "temp_output.wav"
        sf.write(output_path, output_signal, pipeline.sample_rate)
        
        # Generate spectrograms
        # 1. Input Spectrogram (Reference Channel 0)
        in_spec_img = plot_spectrogram(audio_data[:, 0] if audio_data.shape[1] < audio_data.shape[0] else audio_data[0, :], pipeline.sample_rate, "Input Spectrogram (Channel 0)")
        
        # 2. Output Spectrogram
        out_spec_img = plot_spectrogram(output_signal, pipeline.sample_rate, "Separated Output Spectrogram")
        
        from PIL import Image
        in_img = Image.open(in_spec_img)
        out_img = Image.open(out_spec_img)
        
        return output_path, in_img, out_img, "Processing complete!"
        
    except Exception as e:
        import traceback
        return None, None, None, f"Error during processing: {str(e)}\n{traceback.format_exc()}"

# Gradio Interface
with gr.Blocks(title="BioCPPNet - Cocktail Party Problem Solver") as demo:
    gr.Markdown("# 🦇 BioCPPNet: Bioacoustic Source Separation")
    gr.Markdown("Upload a high-frequency (250kHz) multichannel recording and isolate a specific source by providing its estimated direction (azimuth).")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Upload Multichannel Audio (.wav)")
            azimuth_slider = gr.Slider(minimum=-180.0, maximum=180.0, value=0.0, step=1.0, label="Target Azimuth (Degrees)")
            process_btn = gr.Button("Process Audio", variant="primary")
            
        with gr.Column():
            status_text = gr.Textbox(label="Status", interactive=False)
            audio_out = gr.Audio(label="Separated Audio Output", interactive=False)
            
    with gr.Row():
        spec_in = gr.Image(label="Input Mixture")
        spec_out = gr.Image(label="Isolated Source")
        
    process_btn.click(
        fn=process_audio,
        inputs=[audio_in, azimuth_slider],
        outputs=[audio_out, spec_in, spec_out, status_text]
    )

if __name__ == "__main__":
    # Ensure checkpoint is loaded before starting
    demo.launch(server_name="0.0.0.0", server_port=8502, share=False)
