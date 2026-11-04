import argparse
import json
import os
from pathlib import Path


def main():
    # Keep downloaded weights and runtime caches within the project workspace.
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    parser = argparse.ArgumentParser(description="Kirchenbauer watermark replication lab")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "prepare",
            "generate",
            "detect",
            "quality",
            "evaluate",
            "plot",
            "inspect",
            "extend",
            "archive",
        ],
    )
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--from-config", default="configs/pilot.json")
    args = parser.parse_args()
    if args.command == "doctor":
        import torch

        from .storage import environment

        print(
            json.dumps(
                {
                    **environment(),
                    "mps_built": torch.backends.mps.is_built(),
                    "mps_available": torch.backends.mps.is_available(),
                },
                indent=2,
            )
        )
        return
    if args.command == "inspect":
        import torch

        from .core import Detector, SoftWatermark, WatermarkConfig

        settings = WatermarkConfig(8, gamma=0.5, delta=2)
        processor = SoftWatermark(settings)
        ids = torch.tensor([[1, 3]])
        logits = torch.tensor([[0.0, 0.5, -0.2, 1.0, 0.3, -0.5, 0.7, 0.1]])
        biased = processor(ids, logits)
        print(
            json.dumps(
                {
                    "input_shape": list(ids.shape),
                    "logits_shape": list(logits.shape),
                    "previous_token": 3,
                    "green_ids": processor.greenlist.ids(3).tolist(),
                    "original_logits": logits.tolist(),
                    "biased_logits": biased.tolist(),
                    "original_probabilities": logits.softmax(-1).tolist(),
                    "biased_probabilities": biased.softmax(-1).tolist(),
                    "example_detection": Detector(settings).score([1, 3, 2, 3, 2]),
                },
                indent=2,
            )
        )
        return
    from .storage import load_config

    config = load_config(args.config)
    if args.command == "extend":
        from .pipeline import extend

        extend(config, load_config(args.from_config))
        return
    if args.command in ("prepare", "generate", "detect", "quality"):
        from . import pipeline

        getattr(pipeline, args.command)(config)
    else:
        from . import analysis

        getattr(analysis, args.command)(config)


if __name__ == "__main__":
    main()
