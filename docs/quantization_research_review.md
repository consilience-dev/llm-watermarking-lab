# Quantization and watermarking: literature review

Review date: November 4, 2026. **No weight-quantization experiment was performed.**
This section documents prior work, a possible follow-up question, and the reasons
it remains untested. It does not claim a new method or an established research gap.

## Scope of the question

The proposed comparison concerns weight precision in a model to which a fresh
inference-time KGW watermark is applied. That differs from quantizing weights
that already encode a learned watermark, and from extracting ownership signatures
directly from weights. Those distinctions matter when interpreting prior results.

The candidate question is whether generator weight precision changes paraphrasing
resistance after accounting for initial watermark strength and quality cost.
Fixed δ alone does not ensure comparable starting signal across model versions.

## Closest work

| Primary source | Relevant evidence | Boundary for this study |
| --- | --- | --- |
| [Towards Watermarking of Open-Source LLMs](https://arxiv.org/html/2502.10525v1), 2025, §§4–5 and Table 1 | Evaluates embedded/distilled watermarks under quantization, pruning, merging, and fine-tuning. KGW-D reaches 0.99 detection at 5% false positives in several 4-bit and 8-bit settings. | Quantization durability is already studied. The watermark is learned before quantization rather than added freshly by the decoding code. |
| [PRO](https://arxiv.org/html/2510.23891v1), 2025 preprint version, §4 and Appendix G | Jointly trains a watermark policy and LLM, with regularization against downstream modifications; evaluation includes quantization. | Concerns watermarks embedded in model weights, not the proposed controlled decode-time comparison. |
| [From Text to Source](https://arxiv.org/html/2309.13322v2), 2024, §§4–5 | Includes 4-bit GPTQ and red/green watermark experiments on LLaMA families, with γ=0.5 and δ=2, in learned detection and attribution tasks. | The reported classifier tasks do not establish a keyed-detector precision-by-paraphrasing comparison at matched initial signal. |
| [MarkMyWords](https://arxiv.org/html/2312.00273v3), 2024 version, §§IV–V | Studies quality, required text length, parameter choices, and tamper resistance, including paraphrasing. | Quality-aware comparison and paraphrasing evaluation are established methodology. |
| [Watermark under Fire / WaterPark](https://arxiv.org/html/2411.13425v4), 2025 version, §4 and limitations | Benchmarks removal attacks, fidelity, detection, and watermark design choices. | The inspected experiments do not report the proposed precision sweep with initial-strength matching. |
| [MorphMark](https://arxiv.org/html/2505.11541v2), 2025, §§3–4 | Adapts watermark strength using green-token probability mass; evaluates GPT-3.5 and DIPPER paraphrasing. | Adaptive strength, probability-mass analysis, and paraphrasing tests are not new contributions here. |
| [Watermarkability](https://arxiv.org/html/2610.22810v1), October 2026 preprint, §§3–6 | Studies model-dependent target-set costs, target-set-conditioned LoRA, and a closed-form bias controller. Evaluates OPT-125M and Qwen3-8B with grouped KGW among its schedules. | Different effective strength under fixed bias and target-mass control are existing work. Grouped KGW is distinguished from native token-level KGW. |

[On the Reliability of Watermarks](https://arxiv.org/html/2306.04634v4) already
evaluates human and machine paraphrasing. [Watermarking LLMs with Weight
Quantization](https://arxiv.org/abs/2310.11237) plants watermarks through the
quantization process, describing a watermark active in FP32 and hidden in INT8.
[EmMark](https://arxiv.org/abs/2402.17938) authenticates ownership through signatures
in quantized weights. The latter two abstracts were screened to distinguish their
carriers and detectors from inference-time KGW.

[DynamicBias](https://openreview.net/pdf?id=Vvks41GeL9) describes sequence-aware
bias calibration and reports paraphrasing robustness in its indexed primary-source
abstract. Full PDF access was unavailable during this review. Its complete
experiments require inspection before any calibration-based originality claim.

## Proposed controls

Three analyses would address different questions:

| Comparison | Calibration | Purpose |
| --- | --- | --- |
| Fixed bias | Same γ, δ, and sampling settings at FP16, 8-bit, and 4-bit | Effect of changing weight precision without retuning watermark settings |
| Matched initial signal | Tune δ on separate calibration prompts to a shared clean-score target | Rewriting resistance at similar initial detection strength |
| Matched additional quality cost | Tune δ to comparable incremental oracle loss relative to each precision's baseline | Resistance at a comparable extra watermarking cost |

Matching signal and quality simultaneously is not guaranteed by a single δ.
Separate comparisons and trade-off curves would therefore be required. A shared
mean z-score would not imply identical score distributions; quantiles, observed
lengths, and unique-pair counts would also need reporting.

The design would require one pinned checkpoint, tokenizer, and inference backend
across precisions, with quantization method, group size, affected layers, and
compute types recorded. Weight quantization would be separated from watermark
bias arithmetic. Rewriter checkpoint, precision, instructions, and sampling would
remain fixed. A PyTorch FP16 versus MLX 4-bit comparison would confound precision
with backend changes.

Independent calibration and test prompts, unwatermarked controls before and after
rewriting, multiple keys/seeds, semantic-fidelity checks, and uncertainty estimates
would be needed. Frozen-prefix probability measurements could help distinguish
immediate distribution changes from autoregressive history divergence. Existing
adaptive controllers would be attributed baselines, not proposed inventions.

## Assessment

No exact match for this combined controlled comparison was located in the reviewed
sources. The search was focused rather than systematic, and did not complete a
forward-citation audit. This absence does not establish originality.

The completed paraphrasing pilots exposed factual drift, refusals, and ambiguous
source fragments. Those limitations prevented a defensible move to a quantization
sweep within this study. BF16 inference was used for numerical stability, not as
an evaluated low-bit weight-quantization extension.

A future result could show a residual precision effect, disappearance of an
apparent effect after calibration, or insufficient evidence. A small nonsignificant
difference would not establish equivalence. No one outcome is required for the
question to be scientifically useful.
