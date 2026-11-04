# Larger-rewriter follow-up

Protocol recorded before the Qwen3 follow-up outputs were generated, November 4,
2026. This is an exploratory comparison on previously inspected inputs, not a
preregistered confirmatory study.

## Question and scope

Does a different, larger local rewriter avoid the obvious fidelity failures seen
in the first pilot? Use Qwen3-4B-Instruct-2507, with a pinned revision and native
BF16 inference on MPS. This checkpoint supports non-thinking responses, so a
short output budget is sufficient for the requested rewriting task. Its model
card is available at
[Qwen's repository](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Generate exactly the same 30 rewriting tasks as the first pilot: the first five
fixed source prompts, baseline and two watermarked variants, light and strong
rewriting instructions. Keep instructions, seeds, sampling settings, stop policy,
output budget, detector, and observation lengths unchanged. Only the rewriter
checkpoint/revision and output location change. Different model tokenizers and
architectures mean that equal seeds do not create identical random draws.

No output is regenerated because of its detection score or qualitative label.
Keep refusals, shortened outputs, factual changes, and incoherent sources in the
archive. Quantization is outside the scope of this follow-up.

## Fidelity review before interpreting scores

Inspect every source/rewrite pair for:

1. Changes to names, numbers, quantities, attribution, negation, or uncertainty.
2. Omitted substantive claims or added facts, including invented completions of
   source fragments.
3. Refusals, commentary, or responses that are not rewrites.
4. Sources too garbled or incomplete to support a reliable equivalence judgment.

Use the same provisional labels as the first pilot: `no_clear_change`,
`changed_or_omitted`, `unclear_source`, and `refusal`. Grammar and wording changes
are acceptable if substantive claims remain unchanged. `no_clear_change` means
no obvious issue found; it does not certify semantic equivalence or factual truth.

Review text without displaying detector scores in the review packet. The reviewer
already knows the first pilot and can recognize inputs, so this is not a blinded
or independent human evaluation. Identify the reviewer explicitly. Do not present
subjective labels as a validated metric or use them to hide unfavorable cases.

## Analysis and stopping point

Report all outputs and their lengths, refusals, stopping flags, pair retention,
and fixed-length detection scores. Use 100 OPT tokens for the principal
descriptive comparison; show 50 and 200 too. Show original and re-tokenization
controls, and denominators for shorter outputs. If fidelity-conditioned counts
are shown, label the exploratory selection, state exclusions, and retain the
unfiltered table.
