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
    cmd = ["pytest", test_dir, f"--junitxml={report_path}"]

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
    cmd = ["python", main_file]
    click.echo(f"Running {main_file}...")
    # subprocess.run(cmd, check=False) # main.py might not exist yet
    click.echo("Main entry point not yet implemented.")

@cli.command()
def download_data():
    """Download isolated vocalizations."""
    data_dir = DIRS.get("data", "data")
    raw_dir = os.path.join(data_dir, "raw")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Downloading data to {raw_dir}...")
    click.echo("TODO: Implement integration with Earth Species Project Library.")
    # Example: subprocess.run(["wget", ...])

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
    # Load hyperparams
    params = load_config()
    click.echo(f"Hyperparameters: {params.get('model', {})}")
    
    # Import model
    try:
        from src.models.unet import BioCPPNet
        model = BioCPPNet()
        click.echo("Model initialized.")
        # Training loop
    except ImportError as e:
        click.echo(f"Error importing BioCPPNet: {e}")

@cli.command()
def evaluate():
    """Evaluate model performance using SI-SDR."""
    click.echo("Starting evaluation...")
    try:
        from src.metrics.sisdr import calculate_sisdr
        # Dummy evaluation
        import numpy as np
        ref = np.random.randn(1000)
        est = ref + 0.1 * np.random.randn(1000)
        score = calculate_sisdr(ref, est)
        click.echo(f"Dummy SI-SDR score: {score:.2f} dB")
    except ImportError as e:
        click.echo(f"Error importing metrics: {e}")

if __name__ == "__main__":
    cli()
