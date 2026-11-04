# Two local rewriters: exploratory comparison

First: `Qwen/Qwen2.5-1.5B-Instruct`. Second: `Qwen/Qwen3-4B-Instruct-2507`.

Both pilots use the same 30 tasks, source records, instructions, seeds, generation settings, and detector implementation. This is a comparison of two complete model checkpoints, not an isolated model-size effect.

Detection below uses 100 observed OPT tokens, repeated-pair deduplication, and z > 4. Paired counts include only passages long enough in both pilots. Individual eligibility counts show the excluded short outputs. No qualitative label is used to filter the numerical table.

| Variant | Rewrite | First eligible/total | Second eligible/total | Common passages | Clean hits | First hits | Second hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | light | 5/5 | 5/5 | 5 | 0 | 0 | 0 |
| baseline | strong | 5/5 | 5/5 | 5 | 0 | 0 | 0 |
| g0.25-d1 | light | 5/5 | 5/5 | 5 | 2 | 2 | 0 |
| g0.25-d1 | strong | 5/5 | 5/5 | 5 | 2 | 1 | 0 |
| g0.25-d2 | light | 5/5 | 5/5 | 5 | 4 | 2 | 1 |
| g0.25-d2 | strong | 4/5 | 5/5 | 4 | 3 | 1 | 0 |

## Provisional qualitative inspection

| Label | First rewriter | Second rewriter |
| --- | ---: | ---: |
| changed_or_omitted | 20 | 17 |
| no_clear_change | 5 | 5 |
| refusal | 1 | 0 |
| unclear_source | 4 | 8 |

These are model-assisted screening labels, not an independent human study or a validated semantic metric. No clear change means no obvious issue was found, not certified equivalence. The inputs were already inspected during the first pilot. Review notes and all outputs remain in the two archives. Source factual accuracy was not assessed.

With five prompt groups, different eligible subsets, and fidelity failures, these counts cannot establish general attack success rates, model superiority, or quantization effects. Lower detector scores alone are not evidence of a successful meaning-preserving attack.
