#!/usr/bin/env python3
"""只将 NumPy scalar 转为 Python scalar，执行未改动的固定分析。"""
import json
from pathlib import Path
import runpy

import numpy as np

original_default = json.JSONEncoder.default


def numpy_scalar_default(self, value):
    if isinstance(value, np.generic):
        return value.item()
    return original_default(self, value)


json.JSONEncoder.default = numpy_scalar_default
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'analysis.py'), run_name='__main__')
