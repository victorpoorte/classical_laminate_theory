import yaml
import os

def load_config(file_path):
    """Load YAML configuration and handle file includes."""

    data = _load_yaml(file_path)

    trigger = "include_"
    include_entries = [
        key.replace(trigger, "")
        for key in data if trigger in key
    ]

    for entry in include_entries:
        data = _load_and_merge_include_entries(file_path, data, entry)

    return data

def _load_yaml(file_path) -> dict:
    """Load YAML file and return its content as a dictionary."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading YAML file '{file_path}': {e}")
        return {}

def _load_and_merge_include_entries(base_file_path: str, data: dict, entry: str) -> dict:
    """Merge included YAML entry into the current data."""
    include_key = f"include_{entry}"

    # Return if key is not in dict
    if include_key not in data:
        return data

    # Load the data which is to be included
    include_data = _load_include_data(base_file_path, data[include_key])

    # Update the original data with the includes, while maintaining case specific info
    data = _merge_included_data(data, include_data)

    # Remove include key, to keep data clean
    data.pop(include_key)

    return data

def _merge_included_data(data: dict, include_data: dict) -> dict:
    for key, value in include_data.items():
        if (
            key in data 
            and isinstance(data[key], dict) 
            and isinstance(value, dict)
        ):
            # Recursively merge nested dicts
            data[key] = _merge_included_data(data[key], value)
        else:
            # If key not in data or not both dicts, overwrite or set default
            data.setdefault(key, value)
    return data

def _load_include_data(base_file_path, include_path):
    full_include_path = os.path.join(os.path.dirname(base_file_path), include_path)
    include_data: dict = _load_yaml(full_include_path)
    
    return include_data

