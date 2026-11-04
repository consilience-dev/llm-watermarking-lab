# Reading the pilot

This pilot is a pipeline check and an exploratory look at ten prompts. Its perfect
ROC AUCs are not evidence of perfect performance on a population.

At gamma 0.25, increasing delta from 1 to 2 to 5 increases mean z from 5.01 to 9.43
to 20.33. Mean per-sample oracle perplexity is 20.88, 31.36, and 75.86, respectively.
The unwatermarked baseline has perplexity 22.62. This illustrates a detection/quality
tradeoff at stronger bias in this sample. The weak-bias condition happens to have
lower mean perplexity than baseline; ten samples and a proxy metric cannot establish
that watermarking improves quality.

At delta 1, all ten watermarked scores rank above all ten control scores, producing
AUC 1.0, but two watermarked samples fall below the preselected z > 4 threshold.
Perfect ranking and 80% fixed-threshold detection therefore do not contradict one
another.

No human or unwatermarked control crosses z > 4. With ten controls, the reported
95% Wilson upper limit is still 27.8%. We cannot turn a zero observed count into
a claim of an extremely small real false-positive rate.

The next experiment expands the fixed prompt set without tuning settings to this
pilot. No paraphrasing or attack robustness has been measured yet.
