import os
import subprocess
from pathlib import Path
import click
import yaml

# Load configuration
CONFIG_FILE = "project_config.yaml"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

config = load_config()
DIRS = config.get("directories", {})
TEST_CONFIG = config.get("testing", {})
DATA_CONFIG = config.get("training", {})

@click.group()
def cli():
    """BioCPPNet Project Management CLI."""
    pass

@cli.command()
def test():
    """Run unit tests and generate reports."""
    test_dir = DIRS.get("tests", "tests")
    results_dir = DIRS.get("test_results", "unit test results")
    report_file = TEST_CONFIG.get("report_file", "results.xml")

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    report_path = os.path.join(results_dir, report_file)
    cmd = ["python", "-m", "pytest", test_dir, f"--junitxml={report_path}"]

    click.echo(f"Running tests in {test_dir}...")
    subprocess.run(cmd, check=False)

@cli.command()
def lint():
    """Run linting checks."""
    src_dir = DIRS.get("src", "src")
    cmd = ["ruff", "check", src_dir]
    click.echo(f"Linting {src_dir}...")
    subprocess.run(cmd, check=False)

@cli.command()
def run():
    """Run the application."""
    src_dir = DIRS.get("src", "src")
    main_file = os.path.join(src_dir, "main.py")
    cmd = ["python", "-m", "src.main"]
    click.echo(f"Running {main_file}...")
    subprocess.run(cmd, check=False)

@cli.command()
@click.option('--split', default='test', help='Dataset split to download (e.g. train, test).')
@click.option('--limit', default=10, help='Maximum number of files to download.')
def download_data(split, limit):
    """Download isolated vocalizations from the Earth Species Project."""
    data_dir = DIRS.get("data", "data")
    raw_dir = os.path.join(data_dir, "raw")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Downloading data to {raw_dir}...")
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
        
        click.echo(f"Downloading Earth Species Project BEANS dataset ({split}, limit={limit})...")
        # Load the BEANS dataset in streaming mode to avoid downloading 50GB
        dataset = load_dataset("EarthSpeciesProject/BEANS-Zero", split=split, streaming=True)
        
        count = 0
        for i, item in enumerate(dataset):
            if count >= limit:
                break
            
            # The structure might be different, let's extract the array safely
            if "audio" in item and isinstance(item["audio"], dict) and "array" in item["audio"]:
                audio_data = item["audio"].get("array")
                sr = item["audio"].get("sampling_rate", 16000)
            elif "audio" in item and hasattr(item["audio"], "get"):
                # fallback dict
                audio_data = item["audio"].get("array")
                sr = item["audio"].get("sampling_rate", 16000)
            elif "audio" in item and isinstance(item["audio"], list):
                # audio is just a raw list of floats
                audio_data = np.array(item["audio"])
                sr = 16000 # default
            else:
                click.echo(f"Skipping item: audio field type is {type(item.get('audio'))}")
                continue
                
            # Try to get a meaningful label
            label = str(item.get("dataset_name", item.get("task", "unknown"))).replace("/", "_").replace(" ", "_")
            
            filename = f"beans_{label}_{i}.wav"
            filepath = os.path.join(raw_dir, filename)
            
            # Ensure float32 for pipeline compatibility
            sf.write(filepath, audio_data.astype(np.float32), sr)
            count += 1
            
        click.echo(f"Successfully downloaded and saved {count} mono audio files to {raw_dir}.")
    except ImportError:
        click.echo("Error: 'datasets' or 'soundfile' library not found. Please run 'pip install datasets huggingface_hub soundfile'.")
    except Exception as e:
        click.echo(f"Error downloading data: {e}")

@cli.command()
def mix_data():
    """Generate synthetic mixtures from isolated calls."""
    src_dir = DIRS.get("src", "src")
    click.echo("Starting synthetic data generation...")
    # Import here to avoid top-level errors if dependencies are missing
    try:
        from src.data_mixer import DataMixer
        mixer = DataMixer()
        click.echo(f"Initialized DataMixer with sample rate {mixer.sample_rate}Hz")
        # Logic to iterate over raw files and mix them
    except ImportError as e:
        click.echo(f"Error importing DataMixer: {e}")

@cli.command()
@click.option('--config', default=CONFIG_FILE, help='Path to config file')
def train(config):
    """Train the BioCPPNet model."""
    click.echo(f"Starting training using config: {config}")
    
    # Import and run training loop
    try:
        from src.train import train as run_training
        run_training()
    except Exception as e:
        click.echo(f"Error during training: {e}")
        # Re-raise for debugging if needed
        raise e

@cli.command()
def evaluate():
    """Evaluate model performance using SI-SDR."""
    click.echo("Starting evaluation on synthetic data...")
    try:
        import numpy as np
        from src.metrics.sisdr import calculate_sisdr
        from src.pipeline import BioCPPNetPipeline
        from src.spatial.physics import azimuth_elevation_to_vector, calculate_steering_vector, apply_subsample_shifts
        
        pipeline = BioCPPNetPipeline()
        sample_rate = pipeline.sample_rate
        duration = 1.0
        n_samples = int(duration * sample_rate)
        
        # 1. Generate clean target signal (e.g., 4kHz tone)
        t = np.arange(n_samples) / sample_rate
        clean_signal = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
        
        # 2. Spatialise to 45 degrees
        source_vec = azimuth_elevation_to_vector(45.0, 0.0)
        distances = calculate_steering_vector(pipeline.beamformer.mic_positions, source_vec)
        delays = -distances / pipeline.beamformer.speed_of_sound
        multichannel = apply_subsample_shifts(clean_signal, delays, sample_rate)
        
        # Add a bit of noise
        multichannel += 0.1 * np.random.randn(*multichannel.shape).astype(np.float32)
        
        # 3. Process through pipeline
        output_signal = pipeline.process(multichannel, azimuth_deg=45.0)
        
        # 4. Compute SI-SDR
        # Need to ensure lengths match due to potential STFT padding differences
        min_len = min(len(clean_signal), len(output_signal))
        score = calculate_sisdr(clean_signal[:min_len], output_signal[:min_len])
        
        click.echo(f"Evaluation complete. SI-SDR score: {score:.2f} dB")
    except ImportError as e:
        click.echo(f"Error importing dependencies for evaluation: {e}")

if __name__ == "__main__":
    cli()
