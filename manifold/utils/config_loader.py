
import yaml
from pathlib import Path 

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "configs"

def load_config(path:str|Path = CONFIG_DIR/"config.yaml")-> dict:

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open (config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return cfg