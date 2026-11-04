# Reproduction

## Environment

Python 3.12 and uv are required. Install the resolved dependencies with:

```sh
uv sync --frozen --extra dev --python 3.12
uv run --frozen watermark-lab doctor
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check src tests scripts
uv run --frozen --extra dev ruff format --check src tests scripts
```

Run commands from the repository root. Models and runtime caches are excluded
from Git. The CLI defaults Hugging Face and Matplotlib caches to `.cache/`.
Generation configurations use Apple MPS; CPU runs require explicit device/dtype
changes and a separate output directory. Model loading and execution need Metal
access; a restricted process may report MPS unavailable.

## Archived results

The archives contain generation IDs and text, frozen prompts, reference
continuations, settings, oracle losses, detector scores, metrics, and figures.
Detection, evaluation, plotting, and the analyses below do not load an LLM.

```sh
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab detect --config reports/main/replay.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab evaluate --config reports/main/replay.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab plot --config reports/main/replay.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab paraphrase-report --config reports/paraphrase-pilot/replay.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab paraphrase-report --config reports/paraphrase-qwen3-pilot/replay.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab paraphrase-compare --config configs/paraphrase_comparison.json
HF_HUB_OFFLINE=1 uv run --frozen watermark-lab quality-uncertainty --config configs/quality_uncertainty.json
```

The two rewriter archives include all attempts, explicit short-output counts,
tokenization controls, and provisional model-assisted review notes. Comparison
requires matching source records, task settings, implementation hashes, and
recorded inference environments. Quality uncertainty uses 20,000 paired prompt
resamples and a fixed NumPy seed. It is post-hoc analysis.

## Regenerate the baseline

```sh
uv run --frozen watermark-lab prepare --config configs/pilot.json
uv run --frozen watermark-lab generate --config configs/pilot.json
uv run --frozen watermark-lab detect --config configs/pilot.json
uv run --frozen watermark-lab quality --config configs/pilot.json
uv run --frozen watermark-lab evaluate --config configs/pilot.json
uv run --frozen watermark-lab plot --config configs/pilot.json
uv run --frozen python scripts/run_experiment.py --config configs/main.json --from-config configs/pilot.json
```

The runner imports the nested pilot and completes the main run in separate
processes, releasing model memory between stages. Alternatively, run each stage
with `configs/main.json`. Generation and quality scoring resume complete records;
inputs, settings, and recorded environments must match. The main run contains
700 completions and 800 oracle scores, including 100 reference continuations.

## Regenerate rewriting attempts

```sh
uv run --frozen watermark-lab paraphrase --config configs/paraphrase_pilot.json
uv run --frozen watermark-lab paraphrase --config configs/paraphrase_qwen3_pilot.json
```

Each configuration generates 30 outputs. The source OPT generations are already
archived; only the OPT tokenizer and the configured rewriter weights are loaded.
Rewriting uses BF16 and allows EOS. The initial FP16 failure is retained as a
failed-run manifest and is not included as a completed rewrite.

## Provenance

Model and dataset revisions are immutable hashes. Run manifests preserve source
records, configurations, environments, and source hashes. Modified or duplicate
rewrite records are rejected; short outputs are not scored as full-length texts.
Partial JSON lines cause a read error. There must be only one writer per output
directory. Changed scientific settings or generation code require a new output
directory. Fixed seeds do not guarantee identical results across hardware or
library versions.

See [archive provenance](../reports/PROVENANCE.md), individual `REPLAY.md` files,
and [verification](../reports/verification.md). Model weights and the complete
C4 shard are not redistributed. Source-document URLs and attribution are retained
in the prompt records and [methods](protocol.md).
