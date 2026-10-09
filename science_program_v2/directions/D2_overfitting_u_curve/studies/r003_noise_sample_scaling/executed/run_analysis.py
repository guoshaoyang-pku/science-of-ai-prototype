import os
from pathlib import Path
import runpy

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np


original_load = np.load


class CachedNPZ:
    def __init__(self, source):
        self.files = source.files
        self.arrays = {name: source[name] for name in self.files}
        source.close()

    def __getitem__(self, name):
        return self.arrays[name]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def cached_load(*args, **kwargs):
    loaded = original_load(*args, **kwargs)
    return CachedNPZ(loaded) if isinstance(loaded, np.lib.npyio.NpzFile) else loaded


if __name__ == '__main__':
    np.load = cached_load
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'analysis.py'), run_name='__main__')
