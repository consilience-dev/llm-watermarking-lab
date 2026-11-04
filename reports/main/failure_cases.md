# A real weak-bias false negative

Prompt ID: `c4-validation-109`. Generation: gamma 0.25, delta 1. Threshold: z > 4.

| Measurement | Value |
|---|---:|
| Observed completion tokens | 200 |
| Available adjacent pairs | 199 |
| Unique scored pairs | 160 |
| Green matches | 46 |
| Green fraction | 0.2875 |
| Nominal null expectation | 40 |
| z-score | 1.095 |
| Approximate one-sided p-value | 0.137 |

The standardized excess is `(46 - 40) / sqrt(160 * 0.25 * 0.75) = 1.095`.
This is a watermarked output that the detector misses at the chosen threshold.

Of the 199 available adjacent pairs, 39 are excluded as repetitions. The
completion starts with a discussion of company growth and then drifts into
syndication disclaimers and copyright notices. It repeats the footer phrase
“© IE Online Media Services Pvt Ltd.” These observations do not establish low
model entropy or memorization as the cause of failure; we did not measure spike
entropy.

The exact completion and recorded embedding settings are in
[`generations.jsonl`](generations.jsonl), in the record with prompt ID
`c4-validation-109` and variant `g0.25-d1`. Its prompt and human continuation are
in [`prompts.jsonl`](prompts.jsonl). Re-running detection from the replay
configuration reconstructs this result without loading OPT.

The embedding strength delta is recorded as 1 in the generation metadata.
The detector configuration omits delta because counting green matches does not
require knowing how strongly they were favored during generation.
