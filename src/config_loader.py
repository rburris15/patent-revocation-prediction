import yaml
from pathlib import Path


# Get the absolute path to the directory containing config.py
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"

def load_config():
    """Loads the configuration from the YAML file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    

