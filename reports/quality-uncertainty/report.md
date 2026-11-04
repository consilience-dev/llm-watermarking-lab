# Uncertainty in quality differences

Post-hoc analysis of 100 existing prompt groups, with 20,000 paired resamples. Every resampled prompt keeps its baseline and watermarked losses together. Positive differences mean higher oracle loss/perplexity under watermarking.

| Variant | Mean NLL change [95% interval] | Mean sample perplexity change [95% interval] |
| --- | ---: | ---: |
| g0.25-d1 | 0.017 [-0.076, 0.108] | -0.266 [-2.485, 1.929] |
| g0.25-d2 | 0.257 [0.160, 0.355] | 5.768 [3.529, 8.154] |
| g0.25-d5 | 1.239 [1.080, 1.390] | 59.612 [46.029, 80.533] |
| g0.5-d1 | 0.093 [-0.001, 0.185] | 1.426 [-0.764, 3.590] |
| g0.5-d2 | 0.281 [0.185, 0.375] | 6.134 [3.848, 8.449] |
| g0.5-d5 | 0.805 [0.693, 0.917] | 23.373 [19.753, 27.193] |

Intervals are nominal per-comparison percentile bootstrap intervals, not family-wise guarantees or preregistered hypothesis tests. They describe variation across these prompt groups under one key and one sampling seed per prompt. They do not include variability across model families, domains, keys, generation seeds, or oracle choices. An interval including zero does not establish equivalence. Oracle loss and perplexity do not assess factual accuracy or semantic preservation.
