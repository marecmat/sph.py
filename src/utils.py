import numpy as np
import sys
from pathlib import Path

def save_data(filename, data):
    path = Path(sys.argv[0]).resolve().parent / filename
    np.savez_compressed(path, **data)
    print(f"wrote data to {'/'.join(str(path).split('/')[-4:])}")
    return None 

def load(filename):
    with np.load(Path(sys.argv[0]).resolve().parent / filename) as f:
        return {key: f[key] for key in f.files}