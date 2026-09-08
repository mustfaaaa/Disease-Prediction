"""Small shared helpers: JSON-safe serialisation, timing, seeding."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def jsonable(obj):
    """Recursively convert numpy / pandas scalars so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timed(label: str, sink: dict | None = None):
    start = time.perf_counter()
    print(f"  -> {label} ...", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"  <- {label} done in {elapsed:.1f}s", flush=True)
        if sink is not None:
            sink[label] = round(elapsed, 2)


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush=True)
