# Replay the larger-rewriter follow-up

From the repository root:

```sh
HF_HUB_OFFLINE=1 .venv/bin/watermark-lab paraphrase-report --config reports/paraphrase-qwen3-pilot/replay.json
HF_HUB_OFFLINE=1 .venv/bin/watermark-lab paraphrase-compare --config configs/paraphrase_comparison.json
```

These commands load saved records, not models or tokenizers. The first rebuilds
metrics, examples, and the plot while preserving existing review notes. The
second verifies matching inputs, settings, code hashes, and runtime environments
before comparing the pilots on a common eligible set.

`run.json` freezes source records, configurations, environment and generation
source hashes. `rewrites.jsonl` saves messages, both token vocabularies, seeds,
stop flags, scores and timings. `review_packet.md` omits scores and condition
labels, but recognizable inputs mean it is not an independently blinded review.
`provenance.json` distinguishes the generation-launch commit from the later
analysis commit. `artifact_hashes.json` checks the delivered files.

To regenerate text, install the locked environment and use:

```sh
.venv/bin/watermark-lab paraphrase --config configs/paraphrase_qwen3_pilot.json
```

This requires Metal access and downloads the pinned Qwen3 checkpoint and OPT
tokenizer if absent. It does not need OPT weights. Existing unchanged runs resume;
changed inputs, generation source, settings, or environment require a new output
directory. Fixed seeds do not guarantee identical numerical results across
hardware or library versions.
