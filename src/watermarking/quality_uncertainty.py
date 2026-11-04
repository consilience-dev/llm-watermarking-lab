"""Post-hoc paired bootstrap of existing oracle losses; no model inference."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .pipeline import checked_generations, samples, variants
from .storage import environment, fingerprint, load_config, read_jsonl, write_json


def aligned_losses(records, prompt_ids, names, tokens):
    indexed = {(r["prompt_id"], r["variant"]): r for r in records}
    expected = {(p, v) for p in prompt_ids for v in names}
    if len(indexed) != len(records) or indexed.keys() != expected:
        raise ValueError("Quality records are missing, duplicated, or unexpected")
    if any(
        r["num_tokens"] != tokens or not np.isfinite(r["mean_nll"]) or r["mean_nll"] < 0
        for r in records
    ):
        raise ValueError("Invalid token counts or oracle losses")
    return {name: np.array([indexed[p, name]["mean_nll"] for p in prompt_ids]) for name in names}


def bootstrap_difference(baseline, treatment, indices):
    baseline, treatment = np.asarray(baseline), np.asarray(treatment)
    if baseline.ndim != 1 or baseline.shape != treatment.shape or len(baseline) == 0:
        raise ValueError("Require aligned nonempty one-dimensional measurements")
    if indices.ndim != 2 or indices.shape[1] != len(baseline):
        raise ValueError("Each resample must contain a full prompt sample")
    differences = treatment - baseline
    means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def evaluate(config):
    source = load_config(config["source_config"])
    path = Path(source["output_dir"])
    generations = checked_generations(source)
    prompts = samples(source)
    manifest = json.loads((path / "quality.manifest.json").read_text())
    ignored = {"output_dir", "prompts_path"}
    if {k: v for k, v in manifest["config"].items() if k not in ignored} != {
        k: v for k, v in source.items() if k not in ignored
    }:
        raise ValueError("Quality manifest configuration differs")
    inputs = generations + [
        {"prompt_id": p["id"], "variant": "human", "token_ids": p["human_ids"]} for p in prompts
    ]
    if fingerprint(inputs) != manifest["records_hash"]:
        raise ValueError("Quality manifest source records differ")
    names = [name for name, _ in variants(source)] + ["human"]
    losses = aligned_losses(
        read_jsonl(path / "quality.jsonl"),
        [p["id"] for p in prompts],
        names,
        source["completion_tokens"],
    )
    if not 1000 <= config["resamples"] <= 100000:
        raise ValueError("Use 1000–100000 bootstrap resamples")
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, len(prompts), size=(config["resamples"], len(prompts)))
    estimates = []
    for name in names:
        if name in ("baseline", "human"):
            continue
        estimates.append(
            {
                "variant": name,
                "count": len(prompts),
                "mean_nll_change": bootstrap_difference(losses["baseline"], losses[name], indices),
                "mean_sample_perplexity_change": bootstrap_difference(
                    np.exp(losses["baseline"]), np.exp(losses[name]), indices
                ),
            }
        )
    output = Path(config["output_dir"])
    write_json(
        output / "estimates.json",
        {
            "config": config,
            "environment": environment(),
            "rng": type(rng.bit_generator).__name__,
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_hashes": {
                n: hashlib.sha256((path / n).read_bytes()).hexdigest()
                for n in (
                    "quality.jsonl",
                    "quality.manifest.json",
                    "generations.jsonl",
                    "prompts.jsonl",
                )
            },
            "method": "Paired prompt bootstrap; shared prompt indices across conditions; percentile 95% intervals; post-hoc and not multiplicity-adjusted.",
            "estimates": estimates,
        },
    )
    lines = [
        "# Uncertainty in quality differences",
        "",
        (
            f"Post-hoc analysis of {len(prompts)} existing prompt groups, with "
            f"{config['resamples']:,} paired resamples. "
            "Every resampled prompt keeps its baseline and watermarked losses together. "
            "Positive differences mean higher oracle loss/perplexity under watermarking."
        ),
        "",
        "| Variant | Mean NLL change [95% interval] | Mean sample perplexity change [95% interval] |",
        "| --- | ---: | ---: |",
    ]
    for row in estimates:
        cells = []
        for key in ("mean_nll_change", "mean_sample_perplexity_change"):
            value = row[key]
            cells.append(
                f"{value['mean_difference']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"
            )
        lines.append(f"| {row['variant']} | {' | '.join(cells)} |")
    lines += [
        "",
        (
            "Intervals are nominal per-comparison percentile bootstrap intervals, not family-wise "
            "guarantees or preregistered hypothesis tests. They describe variation across these prompt "
            "groups under one key and one sampling seed per prompt. They do not include variability "
            "across model families, domains, keys, generation seeds, or oracle choices. An interval "
            "including zero does not establish equivalence. Oracle loss and perplexity do not assess "
            "factual accuracy or semantic preservation."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Saved paired quality uncertainty: {output}", flush=True)
