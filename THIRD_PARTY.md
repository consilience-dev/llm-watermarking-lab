# Third-party sources

The unmodified reference processor in `vendor/lm_watermarking/` comes from
[jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking/tree/82922516930c02f8aa322765defdb5863d07a00e).
Its Apache-2.0 license and source attribution are retained in that directory.

The experiments use pretrained OPT and Qwen checkpoints downloaded from their
publishers. Model weights are not included in this repository. The pinned model
IDs and revisions are recorded in `configs/` and run manifests; model cards linked
from the methods and follow-up protocol describe their licenses.

Prompt and reference-continuation excerpts are derived from the pinned
[AllenAI C4](https://huggingface.co/datasets/allenai/c4) news-like validation shard.
Source-document URLs, dataset revision, and document fingerprints are retained
in the archived prompt records. The complete dataset shard is not included.
