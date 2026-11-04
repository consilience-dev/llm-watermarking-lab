import hashlib
import importlib.metadata
import json
import platform
from math import isfinite
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    packages = ["torch", "transformers", "numpy", "matplotlib", "datasets", "scipy"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def load_config(path):
    config = json.loads(Path(path).read_text())
    for name in ("model_revision", "oracle_revision", "dataset_revision"):
        revision = config[name]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{name} must be an immutable 40-character commit")
    if not 0 < config["prompt_count"] <= 100:
        raise ValueError("prompt_count must be between 1 and 100")
    if not 0 < config["prompt_tokens"] <= 1024 or config["completion_tokens"] != 200:
        raise ValueError("Require 1–1024 prompt tokens and 200 completion tokens")
    if any(n < 2 or n > config["completion_tokens"] for n in config["lengths"]):
        raise ValueError("Detection lengths must lie between 2 and completion_tokens")
    if config["device"] not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if config["dtype"] not in ("float16", "float32"):
        raise ValueError("dtype must be float16 or float32")
    if (
        not config["gammas"]
        or len(set(config["gammas"])) != len(config["gammas"])
        or any(not 0 < g < 1 for g in config["gammas"])
    ):
        raise ValueError("gammas must be distinct values between zero and one")
    if (
        not config["deltas"]
        or len(set(config["deltas"])) != len(config["deltas"])
        or any(not isfinite(d) or d <= 0 for d in config["deltas"])
    ):
        raise ValueError("deltas must be distinct finite positive values")
    if 4 not in config["thresholds"] or 200 not in config["lengths"]:
        raise ValueError("The report requires threshold 4 and observed length 200")
    return config
