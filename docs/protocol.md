# Replication protocol

## Sources

- [Published original paper and appendix](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)
- [Pinned minimal authors' implementation](https://github.com/jwkirchenbauer/lm-watermarking/tree/82922516930c02f8aa322765defdb5863d07a00e)
- [OPT-1.3B model card and license](https://huggingface.co/facebook/opt-1.3b)
- [OPT-2.7B model card and license](https://huggingface.co/facebook/opt-2.7b)
- [C4 dataset card, attribution, and source cautions](https://huggingface.co/datasets/allenai/c4)

The original experiments use OPT-1.3B, C4 news-like text, approximately 500
completions per setting, and OPT-2.7B perplexity. They examine parameter sweeps,
decoding, length, detection errors, span replacement, and factual QA. This replication selects the parameter, length, and detection/quality comparisons.

## Fixed choices

The sample contains the first 100 distinct eligible documents in the pinned realnewslike
validation shard, each with at least 328 tokenizer tokens. The first 128 tokens
are the prompt, the next 200 are the human continuation. The pilot is the first
ten entries of the same manifest; it is not an independent validation dataset.
Selection requires length, not model response quality or detection score.

One seed per prompt is reused across baseline and six watermarked conditions.
This pairs the experimental inputs but does not ensure token-level alignment
after distributions diverge. The dedicated partition RNG never uses that seed.

The detector sees only completion IDs and omits their first token. Measure both
unique-pair and all-pair scores at every configured length. Use nominal gamma
in the z formula for upstream compatibility; record the effective integer-rounded
green fraction alongside it. For OPT the rounding difference is small.

Use MPS float16 inference and CPU PRNG partitioning. Suppress EOS for fixed
200-token outputs and padded non-tokenizer output IDs for all conditions.
No additional repetition penalties, temperature changes, or top-k/p filtering.

## Interpretation

This is a smaller trend replication with a pinned modern minimal reference,
not an exact historical environment. We differ in sample count, document
selection, prompt slicing, uniform EOS suppression, primary deduplication,
hardware, and random-number backend from the original experiments.

Assess whether stronger bias shifts detection and quality, whether additional
observations strengthen detection on average, and whether controls remain
separated from watermarked outputs. These are questions, not assertions tests
must force to pass. Report contrary outcomes and inspect failure cases.

No claim of an empirically verified 1e-5 false-positive rate follows from this
sample. No AI authorship probability follows from a Gaussian p-value. Human
continuations are web text from C4, not independently certified human authorship.

Paraphrasing and novel research hypotheses follow after this baseline; no attack
robustness claim is made by this milestone.
