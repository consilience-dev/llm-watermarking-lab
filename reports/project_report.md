# Kirchenbauer watermarking replication and paraphrasing study

This project implements the original soft green-list watermark and reproduces
selected detection–quality trends on an Apple-silicon Mac. Its contribution is a
validated, reproducible implementation and a bounded empirical study. It does
not introduce a new watermarking algorithm or establish a new research extension.

## Replication design

The target is [A Watermark for Large Language Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html),
Kirchenbauer et al., ICML 2023. We use OPT-1.3B for generation and separately load
OPT-2.7B to score completions conditioned on their original prompts.

The experiment contains 100 fixed C4 news-like prompts, each with an unwatermarked
completion and six watermark conditions: γ ∈ {0.25, 0.5}, δ ∈ {1, 2, 5}. This
produces 700 completions. Each contains exactly 200 tokens, sampled at temperature
1 with no top-k/top-p truncation. EOS is suppressed, a documented simplification
that can leave incomplete passages. Human continuations provide 100 additional
controls. Models, data, configurations, software versions, seeds, token IDs,
scores, and timings are recorded.

Our custom Transformers logit processor uses previous-token seeding and a
dedicated CPU PyTorch generator for green-list permutations. The model runs on
MPS; detection reconstructs the same CPU partitions without loading an LLM.
The primary detector ignores repeated token pairs and scores standalone
completions, so the first completion token has no scored predecessor. Secondary
results include repeated pairs.

Correctness is checked against the authors' pinned modern minimal implementation
at revision `82922516930c02f8aa322765defdb5863d07a00e`. That is a reference
compatibility check, not a reconstruction of the exact historical ICML runtime.
Tests also cover hand-calculated scores, deterministic partitions, zero bias,
insufficient text, repetition, generation/detection agreement, MPS execution,
resume checks, and the end-to-end pipeline.

## Measured core results

At 200 observed tokens, using repeated-pair deduplication and z > 4:

| Condition | Flagged / 100 | Mean conditional perplexity |
| --- | ---: | ---: |
| Unwatermarked OPT | 0 | 21.35 |
| γ=0.25, δ=1 | 72 | 21.08 |
| γ=0.25, δ=2 | 100 | 27.12 |
| γ=0.25, δ=5 | 100 | 80.96 |
| γ=0.5, δ=1 | 62 | 22.77 |
| γ=0.5, δ=2 | 100 | 27.48 |
| γ=0.5, δ=5 | 100 | 44.72 |
| Human continuation | 0 | 11.60 |

The control counts above apply to each γ separately; they are not independent
new control samples for every watermark condition. Perplexity here is the
arithmetic mean of per-passage values under OPT-2.7B. Pooled token perplexity is
also saved and can rank small differences differently. In particular, δ=1's
slightly lower arithmetic mean at γ=0.25 is not evidence of better writing.
Neither perplexity summary measures factual correctness.

For γ=0.25, δ=2, detection rises from 24/100 at 25 observed tokens to 73/100 at
50, 97/100 at 100, and 100/100 at 200. The results reproduce the expected broad
trade-off: stronger bias and more observed text support detection, while strong
bias can substantially increase perplexity. They are not expected to reproduce
the original paper's exact numerical results.

Sample uncertainty matters: a 100/100 detection count has a 95% Wilson interval
of approximately 96.3–100%; 0/100 flags has an interval of approximately 0–3.7%.
The 72/100 count has an interval of approximately 62.5–79.9%. These intervals do
not account for every source of uncertainty across datasets, models, keys, and
software. The experiment uses one prompt domain and one key, and cannot establish
extremely small empirical false-positive rates.

ROC curves, AUC against separate human and unwatermarked controls, observed-length
curves, quality comparisons, and representative failures are included in
[the main archive](main/report.md). A real weak-bias miss is documented in
[the failure-case analysis](main/failure_cases.md).

A post-hoc paired bootstrap of the saved oracle scores uses 20,000 resamples of
the 100 prompt groups, keeping baseline and watermark conditions together.
For γ=0.25, the mean per-token negative-log-likelihood changes relative to
baseline are 0.017 at δ=1 (nominal 95% interval −0.076 to 0.108), 0.257 at δ=2
(0.160 to 0.355), and 1.239 at δ=5 (1.080 to 1.390). Higher values mean higher
oracle prediction loss. The δ=1 interval includes zero; it does not establish
improvement or equivalence. These exploratory intervals are not adjusted for
multiple comparisons and do not include variation across keys or model families.
All six conditions and arithmetic-perplexity differences are in
[the uncertainty report](quality-uncertainty/report.md).

## Paraphrasing scope

The first pilot used Qwen2.5-1.5B-Instruct to rewrite 15 saved OPT passages under
two instructions, producing 30 outputs. Five prompt groups were reused across
an unwatermarked baseline and γ=0.25 at δ=1 and δ=2. Detection was measured at
fixed OPT-token lengths, with re-tokenization controls and explicit short-output
counts.

At 100 observed tokens, light rewriting reduced δ=2 detection from 4/5 to 2/5.
However, some outputs changed facts, completed missing information, or refused
the task. These are feasibility observations, not established robustness rates
for meaning-preserving paraphrasing. All outputs and provisional qualitative
review notes are retained in [the first pilot archive](paraphrase-pilot/interpretation.md).

The follow-up protocol changes only the rewriter to Qwen3-4B-Instruct-2507,
keeping the same inputs, instructions, seeds, and inference/scoring settings.
[Its scope and review criteria](../docs/paraphrase_followup_protocol.md) were
committed before generation. This small follow-up is exploratory on already
inspected inputs; it is not a confirmatory comparison or independent human study.

The larger rewriter produced all 30 outputs without a refusal, but factual and
interpretive changes remained. The paired 100-token comparison is:

| Source condition | Rewrite | Common eligible passages | Clean hits | Qwen2.5-1.5B hits | Qwen3-4B hits |
| --- | --- | ---: | ---: | ---: | ---: |
| Unwatermarked | Light | 5 | 0 | 0 | 0 |
| Unwatermarked | Strong | 5 | 0 | 0 | 0 |
| γ=0.25, δ=1 | Light | 5 | 2 | 2 | 0 |
| γ=0.25, δ=1 | Strong | 5 | 2 | 1 | 0 |
| γ=0.25, δ=2 | Light | 5 | 4 | 2 | 1 |
| γ=0.25, δ=2 | Strong | 4 | 3 | 1 | 0 |

The final row excludes the smaller rewriter's refusal from both sides of the
paired comparison. Its full cohort has five passages, which are retained in both
archives. All numerical rows include fidelity failures; no qualitative label was
used to select the table. Lower detection counts therefore cannot establish
general rates of successful meaning-preserving attacks.

Provisional model-assisted inspection found no clear change in five outputs for each
rewriter. The remaining labels were changed/omitted, unclear source, or refusal.
Those judgments are not a validated quality metric or independent human study;
the second pilot's text was reviewed without displaying its detector scores,
but inputs were recognizable from the first run. The models also differ in
training and architecture, so this is not an isolated model-size experiment.
All 60 rewriting attempts derive from only 15 source completions and five prompt
groups, not 60 independent prompts.

The [follow-up archive](paraphrase-qwen3-pilot/interpretation.md) contains its
examples and notes; the [paired comparison](paraphrase-comparison/report.md)
includes eligibility and qualitative counts. These pilots show how detector
evidence can change under real rewriting systems and why semantic fidelity must
be checked. They do not validate an unrestricted paraphrase-removal claim.

## Reproducibility and project outcome

The implementation, configurations, tests, frozen inputs, generated outputs,
analysis, and report artifacts are saved in Git. Local caches and downloaded
weights are excluded. Model-based generation is separate from model-free
detection and offline numerical/report replay.

The final test suite has 27 passing tests with Metal access. Static and formatting
checks cover source, tests, and runner scripts. Archived reports are replayed
offline; record hashes and input/source hashes support provenance checks. The
run manifests and per-archive replay instructions retain the details needed to
inspect or reproduce the results under the recorded environment.

The current project scope is complete: a validated smaller replication, two
exploratory rewriting pilots, uncertainty estimates, and a technical report.
It does not claim an original research contribution, exact numerical replication
of the historical paper, general AI-text detection, or negligible false positives.

Quantization remains untested. A literature review identified a possible narrower
question about quantization and rewriting at matched initial watermark strength,
but did not establish its originality. That work is optional and is not needed
to present the completed replication honestly. A future robustness study would
need a declared policy for incoherent/fragmentary sources, a stronger fidelity
evaluation, and fresh confirmation data before drawing broader conclusions.
