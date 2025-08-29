import pickle
from pathlib import Path
import sys

p = Path(r"artifacts/models/xgb_mfe/EURUSDm/model_components.pkl")
print(f"Loading: {p}")
obj = pickle.load(open(p, 'rb'))
print("Loaded type:", type(obj))

if isinstance(obj, dict):
    print("Top-level keys:", list(obj.keys()))
    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"Nested dict under '{k}': keys={list(v.keys())}")
        else:
            shape = getattr(v, 'shape', None)
            try:
                length = len(v)
            except Exception:
                length = None
            print(f"Key '{k}': type={type(v)}, shape={shape}, len={length}")
else:
    print("Dir attributes:", dir(obj)[:50])

