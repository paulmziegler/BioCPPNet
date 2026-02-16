import os
import logging
import datetime
from pathlib import Path
import yaml

# Load config once to be shared
def load_config(config_file="project_config.yaml"):
    # Current dir
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}

    # Parent dir
    parent_path = os.path.join("..", config_file)
    if os.path.exists(parent_path):
        with open(parent_path, "r") as f:
            return yaml.safe_load(f) or {}

    # Root via file path
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config_file)
    if os.path.exists(root_path):
        with open(root_path, "r") as f:
            return yaml.safe_load(f) or {}
            
    return {}

CONFIG = load_config()

def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def setup_logger(name):
    """
    Sets up a logger that outputs to console and a timestamped file.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    config_logging = CONFIG.get("logging", {})
    level_str = config_logging.get("level", "INFO")
    logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))
    
    formatter = logging.Formatter(
        config_logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        datefmt=config_logging.get("date_format", "%Y-%m-%d %H:%M:%S")
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_dir_name = CONFIG.get("directories", {}).get("logs", "logs")
    
    # If config not loaded correctly, default to 'logs'
    if not log_dir_name:
        log_dir_name = "logs"
        
    Path(log_dir_name).mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp_str()
    log_file = os.path.join(log_dir_name, f"biocppnet_{timestamp}.log")
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_plot_path(filename_prefix):
    """
    Returns a path for saving a plot with a timestamp in the test results directory.
    """
    results_dir_name = CONFIG.get("directories", {}).get("test_results", "unit test results")
    if not results_dir_name:
        results_dir_name = "unit test results"
        
    timestamp = get_timestamp_str()
    
    session_dir = os.path.join(results_dir_name, f"session_{timestamp}")
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    
    ext = CONFIG.get("testing", {}).get("plot_format", "png")
    filename = f"{filename_prefix}.{ext}"
    return os.path.join(session_dir, filename)
