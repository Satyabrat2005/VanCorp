import yaml, json

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_json_config(path: str) -> dict:
    with open(path, "r") as file:
        return json.load(file)
