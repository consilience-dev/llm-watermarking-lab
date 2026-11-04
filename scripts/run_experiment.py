"""Run resumable experiment stages in separate processes to release model memory."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    def run(stage, extra=()):
        status = {"stage": stage, "state": "running", "updated_at": datetime.now(UTC).isoformat()}
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        result = subprocess.run(
            [sys.executable, "-m", "watermarking.cli", stage, "--config", args.config, *extra],
            check=False,
        )
        if result.returncode:
            status.update(state="failed", returncode=result.returncode)
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(result.returncode)

    run("prepare")
    # status.json is not a scientific artifact; remove it while validating an empty target.
    if args.from_config and not (output / "run.json").exists():
        (output / "status.json").unlink()
        command = [
            sys.executable,
            "-m",
            "watermarking.cli",
            "extend",
            "--config",
            args.config,
            "--from-config",
            args.from_config,
        ]
        subprocess.run(command, check=True)
    for stage in ("generate", "detect", "quality", "evaluate", "plot", "archive"):
        run(stage)
    (output / "status.json").write_text(
        json.dumps(
            {
                "stage": "complete",
                "state": "complete",
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
