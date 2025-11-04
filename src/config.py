import yaml

def load_config(path: str = "config/base.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
