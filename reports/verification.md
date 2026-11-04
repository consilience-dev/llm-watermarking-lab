# Verification

The Metal-enabled suite passes 27 tests. CPU-only environments skip the MPS test.
Static and formatting checks cover all 16 source, test, and runner files.

## Test coverage

- Deterministic CPU partitions, separate sampling RNG state, and bounded caching.
- Exact processor/detector comparisons against the pinned reference at both green
  fractions, four bias values, and both repetition-scoring modes.
- Reference-source integrity, hand-calculated z/p scores, zero bias,
  insufficient text, and repeated pairs.
- Tiny OPT end-to-end generation, vocabulary/EOS suppression, detection, quality
  scoring, metrics, plotting, resume, and nested pilot reuse.
- Metal logits and generation with shared CPU-keyed partitions.
- Rewriter/OPT vocabulary separation, completion slicing, short outputs,
  changed provenance, record tampering, and model-free replay.
- Common eligible denominators, strict thresholds, and mismatched protocols in
  paired rewriter comparisons.
- Prompt alignment, missing/duplicate quality records, and preservation of paired
  correlation in bootstrap resampling.

The reference is unmodified at revision
`82922516930c02f8aa322765defdb5863d07a00e`, with SHA-256
`512c40644bc9e9932a8674bbf13046c1a4e92db429cff92afc9e90d2226896fc`.
Comparisons use CPU and disable unrelated text normalizers.

## Real runs and replay

The OPT study contains 700 generations, 8,000 detector records, and 800 oracle
scores. The ten-prompt pilot is nested within the main sample. Both rewriting
runs completed 30 tasks using the same source set.

The initial Qwen2.5 FP16 attempt failed before saving a rewrite; its manifest is
retained. Both completed rewriting runs use BF16. This is separate from the
unperformed 4-bit/8-bit weight-quantization comparison.

Archived numerical reports, plots, paired comparisons, and quality uncertainty
are replayed offline. Publication edits normalize reviewer descriptions and
revise narrative reporting; numerical source records and review labels remain
unchanged. Implementation checks and statistical findings are reported separately.

## Commands

```sh
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check src tests scripts
uv run --frozen --extra dev ruff format --check src tests scripts
```

The GitHub workflow runs CPU checks without downloading models. Metal execution
is verified separately on Apple silicon. See [reproduction](../docs/reproduction.md)
and [archive provenance](PROVENANCE.md).
