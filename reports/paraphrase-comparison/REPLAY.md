# Rebuild the paired comparison

From the repository root:

```sh
HF_HUB_OFFLINE=1 .venv/bin/watermark-lab paraphrase-compare --config configs/paraphrase_comparison.json
```

Inputs are the two archived `run.json`, `rewrites.jsonl`, and completed
`manual_review.csv` files. No model or tokenizer is loaded. The command rejects
changed task settings, source records, generation/scoring code hashes, or runtime
environments across the pilots. It also rejects incomplete review worksheets and
duplicate or missing tasks.

`comparison.json` records input hashes, the comparison source hash, paired
counts, and qualitative label counts. The numerical comparison uses only tasks
long enough in both pilots at the requested length. Review labels do not filter
that table. Read `report.md` for the limitations of this exploratory comparison.
