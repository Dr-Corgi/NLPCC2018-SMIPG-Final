import json

def load_json(fpath):
    data = json.load(open(fpath))

    return data