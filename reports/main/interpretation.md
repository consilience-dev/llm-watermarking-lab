# What this replication found

This run used 100 fixed C4 news-like prompts, seven conditions, and 200-token
completions from OPT-1.3B. It produced 700 completions and 800 OPT-2.7B quality
scores, including human continuations. The ten pilot prompts are included in
the 100; they are not additional independent samples. Settings were not tuned
after inspecting the pilot.

## Detection improves with bias and observed length

At a fixed z > 4 threshold and gamma 0.25, delta 1 detects 72/100 outputs, while
delta 2 and delta 5 each detect 100/100. At gamma 0.5 those counts are 62/100,
100/100, and 100/100. These results support the expected bias-strength trend
within this experiment.

For gamma 0.25, delta 2, observing more of each completion increases detection:

| Observed tokens | Mean z | Detected at z > 4 |
|---:|---:|---:|
| 25 | 3.14 | 24/100 |
| 50 | 4.74 | 73/100 |
| 100 | 6.85 | 97/100 |
| 200 | 9.76 | 100/100 |

At 200 observed tokens, deduplication leaves an average of 183.2 scored pairs
in that condition. This makes observed length and effective evidence different
quantities. More text strengthens detection on average; it need not increase
every individual sample's score monotonically.

## Stronger detection has a fluency cost

Mean per-sample oracle perplexity is 21.35 without the watermark. At gamma 0.25,
it is 21.08 with delta 1, 27.12 with delta 2, and 80.96 with delta 5. Strong bias
therefore carries a substantial cost under this fluency proxy. At gamma 0.25,
delta 5 also detects 100/100 outputs using only the first 25 observed tokens,
compared with 24/100 at delta 2. The extra strength buys earlier detection in
this sample despite adding no full-length detection-rate gain.

The slight decrease in arithmetic mean perplexity at delta 1 is not evidence
that watermarking improves quality. Pooled perplexity moves from 18.40 to 18.72,
and only one sample per prompt and condition was generated. The two summaries
weight the distribution of sequence losses differently. Neither evaluates
factual correctness or establishes human preference.

## Controls and uncertainty

No human or unwatermarked control crosses z > 4 for either gamma setting. The
95% Wilson interval for each 0/100 rate extends to 3.7%. For each 100/100
detection rate the lower endpoint is 96.3%. These intervals do not account for
the nonrandom document selection, shared source material, fixed key, or domain
shift. This study cannot establish an extremely small population false-positive
rate.

Weak-bias ROC AUCs are high even when fixed-threshold detection is imperfect.
Ranking separation and detection at a preselected threshold answer different
questions; both are reported.

## Cases worth reading

The weakest gamma 0.25, delta 2 example has z = 4.69 and contains news-site
boilerplate and footer text. It still passes the fixed threshold. The highest
score, z = 13.31, accompanies a technically questionable Vista-security
continuation. Both that continuation and its unwatermarked counterpart contain
unsupported assertions. These examples illustrate why detector confidence and
factual quality need separate evaluations; they do not isolate watermark-induced
factual errors. See [the selected outputs](examples.md).

A genuine delta 1 false negative has only 46 green matches among 160 unique
pairs, yielding z = 1.10. It includes repeated footer language. See
[the failure case](failure_cases.md). We did not measure spike entropy, so we
cannot attribute the failure specifically to low model entropy or memorization.

## Replication boundaries

We reproduced central trends with a smaller, explicitly different protocol.
Document selection, prefix slicing, EOS suppression, deduplicated primary
scoring, hardware, software versions, and CPU random-number generation differ
from the historical experiment. The numerical results are not interchangeable
with the original paper's tables. No single difference has been isolated as the
cause of any numerical discrepancy.

The implementation passes the pinned CPU reference comparisons and the MPS
integration check. Saved outputs support offline detection and reanalysis.
Paraphrasing is evaluated in separate exploratory archives. Weight quantization
and generalization to other generation models are not evaluated.
