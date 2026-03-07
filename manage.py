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
@click.option('--out-dir', default='D:\\Data\\Common\\BEANS', help='Output directory for the downloaded data.')
def download_data(split, limit, out_dir):
    """Download isolated vocalizations from the Earth Species Project."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Downloading data to {out_dir}...")
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
        
        click.echo(f"Downloading Earth Species Project BEANS dataset (split: {split}, limit={limit})...")
        # Load the BEANS dataset in streaming mode to avoid downloading 50GB
        try:
            dataset = load_dataset("EarthSpeciesProject/BEANS-Zero", split=split, streaming=True)
        except ValueError as e:
            if "Bad split" in str(e):
                click.echo(f"Split '{split}' failed. Attempting fallback to 'test' split...")
                dataset = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test", streaming=True)
            else:
                raise e
        
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
            filepath = os.path.join(out_dir, filename)
            
            # Ensure float32 for pipeline compatibility
            sf.write(filepath, audio_data.astype(np.float32), sr)
            count += 1
            
        click.echo(f"Successfully downloaded and saved {count} mono audio files to {out_dir}.")
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
@click.option('--num-samples', default=5, help='Number of real samples to evaluate.')
def evaluate(num_samples):
    """Evaluate model performance using real samples from the BEANS dataset."""
    click.echo(f"Starting evaluation on {num_samples} real samples...")
    try:
        import numpy as np
        import torch
        from src.metrics.sisdr import calculate_sisdr
        from src.pipeline import BioCPPNetPipeline
        from src.data_mixer import DataMixer
        import soundfile as sf
        
        # Load Pipeline
        pipeline = BioCPPNetPipeline()
        sample_rate = pipeline.sample_rate
        
        # Load trained weights - find the latest one
        checkpoint_dir = "results/checkpoints"
        latest_unet = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("unet_epoch_") and f.endswith(".pt")]
            if checkpoints:
                # Sort by epoch number
                import re
                def get_epoch(f):
                    match = re.search(r"unet_epoch_(\d+)\.pt", f)
                    return int(match.group(1)) if match else 0
                
                checkpoints.sort(key=get_epoch, reverse=True)
                latest_unet = os.path.join(checkpoint_dir, checkpoints[0])

        if latest_unet:
            click.echo(f"Loading latest trained U-Net weights from {latest_unet}...")
            pipeline.load_weights(unet_path=latest_unet)
        else:
            click.echo(f"Warning: No trained U-Net weights found in {checkpoint_dir}. Using random init.")
            
        # Locate real data
        shared_data_dir = r"D:\Data\Common\BEANS"
        docker_data_dir = "/data/beans"
        data_dir = shared_data_dir if os.path.exists(shared_data_dir) else docker_data_dir
        
        if not os.path.exists(data_dir):
            click.echo(f"Error: Data directory {data_dir} not found. Cannot run real evaluation.")
            return

        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
        if len(all_files) < 2:
            click.echo("Error: Not enough files in data directory to create mixtures.")
            return

        mixer = DataMixer(sample_rate=sample_rate)
        scores = []

        for i in range(num_samples):
            # 1. Pick two random files
            target_path = np.random.choice(all_files)
            interferer_path = np.random.choice(all_files)
            
            target_mono, _ = sf.read(target_path)
            interferer_mono, _ = sf.read(interferer_path)
            
            # Truncate to 1 second
            n_samples = sample_rate
            target_mono = target_mono[:n_samples] if len(target_mono) > n_samples else np.pad(target_mono, (0, n_samples - len(target_mono)))
            interferer_mono = interferer_mono[:n_samples] if len(interferer_mono) > n_samples else np.pad(interferer_mono, (0, n_samples - len(interferer_mono)))

            # 2. Spatialize
            target_az = np.random.uniform(0, 180)
            interferer_az = np.random.uniform(0, 180)
            
            target_spatial = mixer.spatialise_signal(target_mono, target_az, add_reverb=True)
            interferer_spatial = mixer.spatialise_signal(interferer_mono, interferer_az, add_reverb=True)
            
            # Mix with random SNR
            snr_db = np.random.uniform(-5, 5)
            mixture = mixer.mix_signals(target_spatial, interferer_spatial, snr_db)
            mixture = mixer.add_noise(mixture, 'pink', snr_db=15)
            
            # 3. Process
            output_signal = pipeline.process(mixture, azimuth_deg=target_az)
            
            # 4. Score against dry reference (Direct Path)
            # Reference mic is channel 0 of direct spatialisation
            reference_dry = mixer.spatialise_signal(target_mono, target_az, add_reverb=False)[0]
            
            min_len = min(len(reference_dry), len(output_signal))
            score = calculate_sisdr(reference_dry[:min_len], output_signal[:min_len])
            
            if not np.isinf(score) and not np.isnan(score):
                scores.append(score)
                click.echo(f" Sample {i+1}/{num_samples}: SI-SDR = {score:.2f} dB")

        if scores:
            avg_score = np.mean(scores)
            click.echo(f"\nEvaluation complete. Average Real-World SI-SDR: {avg_score:.2f} dB")
        else:
            click.echo("Evaluation failed to produce valid scores.")

    except ImportError as e:
        click.echo(f"Error importing dependencies: {e}")
    except Exception as e:
        click.echo(f"Evaluation error: {e}")

@cli.command()
def demo():
    """Run the interactive Gradio demo."""
    click.echo("Starting Gradio demo...")
    try:
        import app
        app.demo.launch(server_name="0.0.0.0", server_port=8502, share=False)
    except ImportError as e:
        click.echo(f"Error importing dependencies for demo. Did you run 'pip install gradio'? {e}")

if __name__ == "__main__":
    cli()
