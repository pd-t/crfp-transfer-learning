
import json

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)