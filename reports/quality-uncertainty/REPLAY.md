# Rebuild quality uncertainty estimates

From the repository root:

```sh
HF_HUB_OFFLINE=1 .venv/bin/watermark-lab quality-uncertainty --config configs/quality_uncertainty.json
```

This reads saved oracle losses and frozen generation/prompt records, verifies
the quality manifest's source fingerprint, aligns all prompt-condition pairs,
and resamples prompt groups together across conditions. No LLM or tokenizer is
loaded. The configuration fixes 20,000 resamples and the NumPy seed.

`estimates.json` records numerical results, settings, input hashes, the analysis
source hash, the RNG implementation, and runtime environment. `report.md` states
the method and limitations. These are post-hoc nominal percentile intervals,
not preregistered tests or simultaneous coverage guarantees.
