import click
import numpy as np
from src.pipeline import BioCPPNetPipeline

@click.command()
@click.option('--duration', default=1.0, help='Duration of mock audio in seconds.')
def main(duration):
    """Run the End-to-End Pipeline on mock data."""
    click.echo("Initializing BioCPPNet Pipeline...")
    pipeline = BioCPPNetPipeline()
    
    # Generate mock multichannel audio (e.g., 4 channels)
    sample_rate = pipeline.sample_rate
    num_samples = int(sample_rate * duration)
    num_channels = pipeline.beamformer.n_channels
    
    click.echo(f"Generating mock data: {num_channels} channels, {num_samples} samples at {sample_rate}Hz.")
    # Provide data as (Time, Channels) or (Channels, Time). Let's do (Channels, Time).
    mock_audio = np.random.randn(num_channels, num_samples).astype(np.float32)
    
    click.echo("Running pipeline...")
    # Run the pipeline
    output_audio = pipeline.process(mock_audio, azimuth_deg=45.0)
    
    click.echo(f"Pipeline completed! Output shape: {output_audio.shape}")
    
if __name__ == "__main__":
    main()
