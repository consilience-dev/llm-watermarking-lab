# Replay the paraphrasing pilot

Run from the repository root:

```sh
HF_HUB_OFFLINE=1 .venv/bin/watermark-lab paraphrase-report --config reports/paraphrase-pilot/replay.json
```

This regenerates `evaluation.json`, `metrics.csv`, `report.md`, `examples.md`, and
`scores.png` from the archived records. It verifies record hashes and preserves
the completed review worksheet. No model or tokenizer is loaded. The
interpretation and review judgments are separate from the generated metrics.

`run.json` includes the original selected inputs, source configuration, rewriter
settings, environment, and source hashes. `rewrites.jsonl` includes prompts,
Qwen and OPT token IDs, seeds, scores, stop flags, and timings.
`provenance.json` records the implementation commit used for generation.
`failed_fp16_run.json` retains the initial failed attempt's settings; that attempt
produced no saved rewrites. `artifact_hashes.json` checks the delivered files.

To regenerate text in a fresh checkout, install the locked environment and run:

```sh
.venv/bin/watermark-lab paraphrase --config configs/paraphrase_pilot.json
.venv/bin/watermark-lab paraphrase-report --config configs/paraphrase_pilot.json
```

The first command downloads the pinned rewriter if absent and requires Metal
access. Original OPT generations are already included in `reports/main`; only
OPT's tokenizer is needed, not its model weights. Existing runs resume; changing
settings, source code, inputs, or environment requires a new output directory.
Hardware and library changes can change numerical results despite fixed seeds.
