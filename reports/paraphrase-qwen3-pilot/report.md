# Paraphrasing pilot

15 saved OPT completions, 5 prompts, 30 rewrites. Rewriter: Qwen/Qwen3-4B-Instruct-2507 at `cdbee75f17c01a7cc42f958dc650907174af0554`; watermark disabled.

Exploratory pipeline check, not a quantization result. Five prompt groups cannot establish robustness rates or small false-positive rates. No quality-based filtering is applied. Meaning preservation requires manual review; token-pair retention measures lexical overlap only.

All detection uses OPT token IDs, CPU partitions, and repeated-pair deduplication. A hit means z > 4. Original = saved token IDs; roundtrip = original text re-tokenized without rewriting. Both clean columns use the same eligible subset as the rewrite. Short outputs are counted separately, not silently scored as if they reached the observation length.

| Variant | Rewrite | Observed | Eligible/total | Original hits | Roundtrip hits | Rewritten hits | Mean rewritten z |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | light | 50 | 5/5 | 0 | 0 | 0 | 0.47 |
| baseline | light | 100 | 5/5 | 0 | 0 | 0 | 0.16 |
| baseline | light | 200 | 3/5 | 0 | 0 | 0 | 0.57 |
| baseline | strong | 50 | 5/5 | 0 | 0 | 0 | 0.35 |
| baseline | strong | 100 | 5/5 | 0 | 0 | 0 | 0.50 |
| baseline | strong | 200 | 2/5 | 0 | 0 | 0 | 0.22 |
| g0.25-d1 | light | 50 | 5/5 | 0 | 0 | 0 | 0.98 |
| g0.25-d1 | light | 100 | 5/5 | 2 | 2 | 0 | 1.57 |
| g0.25-d1 | light | 200 | 1/5 | 1 | 1 | 0 | 2.93 |
| g0.25-d1 | strong | 50 | 5/5 | 0 | 0 | 0 | 0.43 |
| g0.25-d1 | strong | 100 | 5/5 | 2 | 2 | 0 | 0.86 |
| g0.25-d1 | strong | 200 | 3/5 | 2 | 2 | 0 | 1.23 |
| g0.25-d2 | light | 50 | 5/5 | 3 | 3 | 0 | 2.29 |
| g0.25-d2 | light | 100 | 5/5 | 4 | 4 | 1 | 2.73 |
| g0.25-d2 | light | 200 | 5/5 | 5 | 5 | 3 | 4.50 |
| g0.25-d2 | strong | 50 | 5/5 | 3 | 3 | 0 | 0.75 |
| g0.25-d2 | strong | 100 | 5/5 | 4 | 4 | 0 | 1.03 |
| g0.25-d2 | strong | 200 | 4/5 | 4 | 4 | 0 | 1.68 |

Original generations suppress EOS and may end mid-sentence. Rewrites allow EOS. Consequently their lengths and endings can differ; inspect examples and truncation flags. The attack instructions request different edit strengths but do not guarantee them. This pilot does not measure oracle perplexity or semantic similarity with a learned metric.
