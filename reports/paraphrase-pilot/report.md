# Paraphrasing pilot

15 saved OPT completions, 5 prompts, 30 rewrites. Rewriter: Qwen/Qwen2.5-1.5B-Instruct at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; watermark disabled.

Exploratory pipeline check, not a quantization result. Five prompt groups cannot establish robustness rates or small false-positive rates. No quality-based filtering is applied. Meaning preservation requires manual review; token-pair retention measures lexical overlap only.

All detection uses OPT token IDs, CPU partitions, and repeated-pair deduplication. A hit means z > 4. Original = saved token IDs; roundtrip = original text re-tokenized without rewriting. Both clean columns use the same eligible subset as the rewrite. Short outputs are counted separately, not silently scored as if they reached the observation length.

| Variant | Rewrite | Observed | Eligible/total | Original hits | Roundtrip hits | Rewritten hits | Mean rewritten z |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | light | 50 | 5/5 | 0 | 0 | 0 | 0.47 |
| baseline | light | 100 | 5/5 | 0 | 0 | 0 | 0.55 |
| baseline | light | 200 | 1/5 | 0 | 0 | 0 | -0.96 |
| baseline | strong | 50 | 5/5 | 0 | 0 | 0 | -0.03 |
| baseline | strong | 100 | 5/5 | 0 | 0 | 0 | 0.66 |
| baseline | strong | 200 | 0/5 | 0 | 0 | 0 | — |
| g0.25-d1 | light | 50 | 5/5 | 0 | 0 | 0 | 1.78 |
| g0.25-d1 | light | 100 | 5/5 | 2 | 2 | 2 | 2.98 |
| g0.25-d1 | light | 200 | 2/5 | 2 | 2 | 1 | 5.08 |
| g0.25-d1 | strong | 50 | 5/5 | 0 | 0 | 0 | 1.27 |
| g0.25-d1 | strong | 100 | 5/5 | 2 | 2 | 1 | 2.13 |
| g0.25-d1 | strong | 200 | 1/5 | 1 | 1 | 1 | 4.33 |
| g0.25-d2 | light | 50 | 5/5 | 3 | 3 | 1 | 1.95 |
| g0.25-d2 | light | 100 | 5/5 | 4 | 4 | 2 | 3.79 |
| g0.25-d2 | light | 200 | 1/5 | 1 | 1 | 1 | 10.56 |
| g0.25-d2 | strong | 50 | 4/5 | 2 | 2 | 0 | 2.18 |
| g0.25-d2 | strong | 100 | 4/5 | 3 | 3 | 1 | 3.57 |
| g0.25-d2 | strong | 200 | 0/5 | 0 | 0 | 0 | — |

Original generations suppress EOS and may end mid-sentence. Rewrites allow EOS. Consequently their lengths and endings can differ; inspect examples and truncation flags. The attack instructions request different edit strengths but do not guarantee them. This pilot does not measure oracle perplexity or semantic similarity with a learned metric.
