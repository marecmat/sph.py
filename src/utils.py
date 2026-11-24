import numpy as np

def load(data):
    with np.load(data) as f:
        return {key: f[key] for key in f.files}