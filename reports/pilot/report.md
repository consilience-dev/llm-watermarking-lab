# OPT-1.3B replication results

Measured run: `outputs/pilot`. 10 fixed C4 news-like prompts; 70 generated completions, 200 tokens each.

This is a smaller replication of experimental trends, not the original numerical results.

## Detection at 200 observed tokens

Primary scoring ignores repeated token pairs, omits the first completion token, and excludes all prompt tokens. Thus 200 observed tokens provide at most 199 scored pairs.

| γ | Condition | n | Mean z | Rate z > 4 (95% Wilson interval) | AUC vs human |
|---|---|---:|---:|---|---:|
| 0.25 | baseline | 10 | 0.43 | 0.0% (0.0%–27.8%) | — |
| 0.25 | g0.25-d1 | 10 | 5.01 | 80.0% (49.0%–94.3%) | 1.000 |
| 0.25 | g0.25-d2 | 10 | 9.43 | 100.0% (72.2%–100.0%) | 1.000 |
| 0.25 | g0.25-d5 | 10 | 20.33 | 100.0% (72.2%–100.0%) | 1.000 |
| 0.25 | human | 10 | 0.10 | 0.0% (0.0%–27.8%) | — |
| 0.5 | baseline | 10 | -0.71 | 0.0% (0.0%–27.8%) | — |
| 0.5 | g0.5-d1 | 10 | 4.43 | 80.0% (49.0%–94.3%) | 1.000 |
| 0.5 | g0.5-d2 | 10 | 8.47 | 100.0% (72.2%–100.0%) | 1.000 |
| 0.5 | g0.5-d5 | 10 | 12.34 | 100.0% (72.2%–100.0%) | 1.000 |
| 0.5 | human | 10 | -0.38 | 0.0% (0.0%–27.8%) | — |

Rates for controls are false-positive rates; rates for watermarked conditions are detection rates. Intervals describe sampling uncertainty; they do not correct domain, key, or model selection effects.

## Generation performance

Mean per-completion throughput: 33.4 tokens/s. Timing includes processor overhead and synchronization, excludes model download/loading. The first completion includes warm-up overhead.
Largest sampled MPS driver allocation: 3.14 GiB. This is a between-completion sample, not total process memory or peak memory.

## Conditional perplexity

OPT-2.7B scores only completion tokens conditioned on the saved prompt. Perplexity is a fluency proxy, not a factuality or human-preference measure.

| Condition | n | Mean per-sample perplexity | Pooled perplexity |
|---|---:|---:|---:|
| baseline | 10 | 22.62 | 20.42 |
| g0.25-d1 | 10 | 20.88 | 19.91 |
| g0.25-d2 | 10 | 31.36 | 27.30 |
| g0.25-d5 | 10 | 75.86 | 69.82 |
| human | 10 | 12.79 | 11.41 |
| g0.5-d1 | 10 | 26.13 | 23.63 |
| g0.5-d2 | 10 | 34.45 | 30.00 |
| g0.5-d5 | 10 | 47.75 | 43.00 |

## Protocol and limits

- Model and dataset revisions, environment, and prompt hash: `run.json`.
- Full generation settings and vocabulary dimensions: `generation_settings.json`.
- All eligible-pair and deduplicated scores: `detections.jsonl`; summaries: `metrics.csv`.
- Fixed 128-token prefixes from the first 100 eligible distinct validation documents; not a random sample of the complete C4 distribution.
- EOS is suppressed for all conditions; this differs from the original multinomial sample-selection protocol. Repetitive tails remain in the analysis.
- CPU PRNG partitions match the pinned reference on CPU, not its GPU PRNG streams.
- OPT has padded output logits beyond the tokenizer vocabulary. These are suppressed in all conditions; the watermark partitions the actual tokenizer vocabulary.
- Detector uses nominal γ for the reference-compatible z-score; effective γ is saved.
- Gaussian p-values are approximate and assume a suitable null; they are not empirically established false-positive rates or authorship probabilities.
- One key and one sampling seed per prompt; no tiny false-positive-rate claim is supported.
- Beam search, attacks, TriviaQA, Qwen, and research extensions are deferred.

## Figures

![Detection strength versus length](figures/length.png)

![Detection ROC curves](figures/roc.png)

![Detection strength versus conditional perplexity](figures/quality.png)
