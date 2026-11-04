import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .storage import read_jsonl, write_json


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    midpoint = (proportion + z * z / (2 * total)) / denominator
    width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return [max(0.0, midpoint - width / denominator), min(1.0, midpoint + width / denominator)]


def roc(positives, negatives):
    if not positives or not negatives:
        raise ValueError("ROC requires both classes")
    scores = np.array(list(positives) + list(negatives), dtype=float)
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]] / len(positives)
    fp = np.r_[0, np.cumsum(1 - labels)[ends]] / len(negatives)
    return fp.tolist(), tp.tolist(), float(np.trapezoid(tp, fp))


def evaluate(config):
    output = Path(config["output_dir"])
    detections = read_jsonl(output / "detections.jsonl")
    groups = defaultdict(list)
    for row in detections:
        if row["z_score"] is not None:
            groups[(row["gamma"], row["variant"], row["length"], row["deduplicate"])].append(row)
    quality_path = output / "quality.jsonl"
    quality_rows = read_jsonl(quality_path) if quality_path.exists() else []
    quality_groups = defaultdict(list)
    for row in quality_rows:
        quality_groups[row["variant"]].append(row)
    metrics = []
    curves = []
    for (gamma, variant, length, deduplicate), rows in sorted(groups.items()):
        scores = [r["z_score"] for r in rows]
        metric = {
            "gamma": gamma,
            "variant": variant,
            "length": length,
            "deduplicate": deduplicate,
            "count": len(rows),
            "mean_z": float(np.mean(scores)),
            "mean_tokens_scored": float(np.mean([r["num_tokens_scored"] for r in rows])),
        }
        for threshold in config["thresholds"]:
            hits = sum(z > threshold for z in scores)
            metric[f"rate_at_z{threshold:g}"] = hits / len(scores)
            metric[f"rate_at_z{threshold:g}_ci95"] = wilson(hits, len(scores))
        quality_records = quality_groups[variant]
        if quality_records:
            nlls = [r["mean_nll"] for r in quality_records]
            metric["quality_count"] = len(quality_records)
            metric["mean_sample_perplexity"] = float(np.mean(np.exp(nlls)))
            metric["pooled_perplexity"] = math.exp(
                sum(r["mean_nll"] * r["num_tokens"] for r in quality_records)
                / sum(r["num_tokens"] for r in quality_records)
            )
        if variant not in ("baseline", "human"):
            for control in ("human", "baseline"):
                negatives = [r["z_score"] for r in groups[(gamma, control, length, deduplicate)]]
                fp, tp, auc = roc(scores, negatives)
                metric[f"auc_vs_{control}"] = auc
                curves.append(
                    {
                        "gamma": gamma,
                        "variant": variant,
                        "length": length,
                        "deduplicate": deduplicate,
                        "control": control,
                        "fpr": fp,
                        "tpr": tp,
                        "auc": auc,
                    }
                )
        metrics.append(metric)
    write_json(output / "evaluation.json", {"metrics": metrics, "roc_curves": curves})
    columns = sorted({k for m in metrics for k in m})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)
    report(config, metrics, quality_rows)
    print(f"Saved metrics and report: {output}", flush=True)


def report(config, metrics, quality_rows):
    output = Path(config["output_dir"])
    generations = read_jsonl(output / "generations.jsonl")
    primary = [m for m in metrics if m["deduplicate"] and m["length"] == 200]
    lines = [
        "# OPT-1.3B replication results",
        "",
        (
            f"Measured run: `{output}`. {config['prompt_count']} fixed C4 news-like prompts; "
            f"{len(generations)} generated completions, 200 tokens each."
        ),
        "",
        "This is a smaller replication of experimental trends, not the original numerical results.",
        "",
        "## Detection at 200 observed tokens",
        "",
        (
            "Primary scoring ignores repeated token pairs, omits the first completion token, "
            "and excludes all prompt tokens. Thus 200 observed tokens provide at most 199 scored pairs."
        ),
        "",
        "| γ | Condition | n | Mean z | Rate z > 4 (95% Wilson interval) | AUC vs human |",
        "|---|---|---:|---:|---|---:|",
    ]
    for m in primary:
        interval = m["rate_at_z4_ci95"]
        auc = f"{m['auc_vs_human']:.3f}" if "auc_vs_human" in m else "—"
        lines.append(
            f"| {m['gamma']:g} | {m['variant']} | {m['count']} | {m['mean_z']:.2f} | "
            f"{m['rate_at_z4']:.1%} ({interval[0]:.1%}–{interval[1]:.1%}) | {auc} |"
        )
    lines += [
        "",
        (
            "Rates for controls are false-positive rates; rates for watermarked "
            "conditions are detection rates. Intervals describe sampling uncertainty; "
            "they do not correct domain, key, or model selection effects."
        ),
        "",
        "## Generation performance",
        "",
    ]
    speeds = [r["tokens_per_second"] for r in generations]
    lines.append(
        f"Mean per-completion throughput: {np.mean(speeds):.1f} tokens/s. "
        "Timing includes processor overhead and synchronization, excludes model download/loading. "
        "The first completion includes warm-up overhead."
    )
    peak = max((r.get("mps_driver_bytes", 0) for r in generations), default=0)
    if peak:
        lines.append(
            f"Largest sampled MPS driver allocation: {peak / 2**30:.2f} GiB. "
            "This is a between-completion sample, not total process memory or peak memory."
        )
    expected_quality = config["prompt_count"] * (2 + len(config["gammas"]) * len(config["deltas"]))
    lines += ["", "## Conditional perplexity", ""]
    if len(quality_rows) != expected_quality:
        lines.append(
            f"Quality scoring incomplete: {len(quality_rows)}/{expected_quality} records. "
            "Do not interpret partial quality comparisons."
        )
    else:
        lines += [
            (
                "OPT-2.7B scores only completion tokens conditioned on the saved prompt. "
                "Perplexity is a fluency proxy, not a factuality or human-preference measure."
            ),
            "",
            "| Condition | n | Mean per-sample perplexity | Pooled perplexity |",
            "|---|---:|---:|---:|",
        ]
        for m in primary:
            if m["gamma"] != config["gammas"][0] and m["variant"] in ("baseline", "human"):
                continue
            lines.append(
                f"| {m['variant']} | {m['quality_count']} | "
                f"{m['mean_sample_perplexity']:.2f} | {m['pooled_perplexity']:.2f} |"
            )
    lines += [
        "",
        "## Protocol and limits",
        "",
        "- Model and dataset revisions, environment, and prompt hash: `run.json`.",
        "- Full generation settings and vocabulary dimensions: `generation_settings.json`.",
        "- All eligible-pair and deduplicated scores: `detections.jsonl`; summaries: `metrics.csv`.",
        (
            "- Fixed 128-token prefixes from the first 100 eligible distinct validation documents; "
            "not a random sample of the complete C4 distribution."
        ),
        (
            "- EOS is suppressed for all conditions; this differs from the original multinomial "
            "sample-selection protocol. Repetitive tails remain in the analysis."
        ),
        "- CPU PRNG partitions match the pinned reference on CPU, not its GPU PRNG streams.",
        (
            "- OPT has padded output logits beyond the tokenizer vocabulary. These are suppressed "
            "in all conditions; the watermark partitions the actual tokenizer vocabulary."
        ),
        "- Detector uses nominal γ for the reference-compatible z-score; effective γ is saved.",
        (
            "- Gaussian p-values are approximate and assume a suitable null; they are not "
            "empirically established false-positive rates or authorship probabilities."
        ),
        "- One key and one sampling seed per prompt; no tiny false-positive-rate claim is supported.",
        "- Beam search, attacks, TriviaQA, Qwen, and research extensions are deferred.",
        "",
        "## Figures",
        "",
        "![Detection strength versus length](figures/length.png)",
        "",
        "![Detection ROC curves](figures/roc.png)",
        "",
        (
            "![Detection strength versus conditional perplexity](figures/quality.png)"
            if len(quality_rows) == expected_quality
            else "Quality figure is generated only after all oracle scores are present."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    prompts = {r["id"]: r for r in read_jsonl(config["prompts_path"])}
    examples = [
        "# Representative high- and low-score outputs",
        "",
        (
            "Selected mechanically by the deduplicated 200-token z-score for γ=0.25, δ=2. "
            "Inspect these as cases, not as a representative quality survey."
        ),
        "",
    ]
    candidates = [
        r
        for r in read_jsonl(output / "detections.jsonl")
        if r["variant"] == "g0.25-d2" and r["length"] == 200 and r["deduplicate"]
    ]
    candidates.sort(key=lambda r: r["z_score"])
    for label, case in zip(("Lowest score", "Highest score"), (candidates[:1] + candidates[-1:])):
        matching = next(
            r
            for r in generations
            if r["prompt_id"] == case["prompt_id"] and r["variant"] == case["variant"]
        )
        baseline = next(
            r
            for r in generations
            if r["prompt_id"] == case["prompt_id"] and r["variant"] == "baseline"
        )
        examples += [
            f"## {label}: {case['prompt_id']} (z={case['z_score']:.2f})",
            "",
            "Prompt:",
            "",
            prompts[case["prompt_id"]]["prompt_text"],
            "",
            "Watermarked completion:",
            "",
            matching["text"],
            "",
            "Unwatermarked completion:",
            "",
            baseline["text"],
            "",
        ]
    (output / "examples.md").write_text("\n".join(examples))


def plot(config):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(config["output_dir"])
    evaluated = json.loads((output / "evaluation.json").read_text())
    metrics = [m for m in evaluated["metrics"] if m["deduplicate"]]
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(config["gammas"]), figsize=(11, 4), squeeze=False)
    for ax, gamma in zip(axes[0], config["gammas"]):
        conditions = sorted({m["variant"] for m in metrics if m["gamma"] == gamma})
        for variant in conditions:
            points = sorted(
                [m for m in metrics if m["gamma"] == gamma and m["variant"] == variant],
                key=lambda m: m["length"],
            )
            ax.plot(
                [m["length"] for m in points],
                [m["mean_z"] for m in points],
                marker="o",
                label=variant,
            )
        ax.axhline(4, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"γ = {gamma:g}", xlabel="Observed completion tokens", ylabel="Mean z-score")
        ax.legend(fontsize=8)
    fig.suptitle("Watermark strength versus length (repeated pairs excluded)")
    fig.tight_layout()
    fig.savefig(directory / "length.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, control in zip(axes, ("human", "baseline")):
        for curve in evaluated["roc_curves"]:
            if curve["length"] == 200 and curve["deduplicate"] and curve["control"] == control:
                ax.plot(
                    curve["fpr"], curve["tpr"], label=f"{curve['variant']} (AUC {curve['auc']:.3f})"
                )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set(
            title=f"Against {control}",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("200-token detection ROC (repeated pairs excluded)")
    fig.tight_layout()
    fig.savefig(directory / "roc.png", dpi=160)
    plt.close(fig)
    quality = [m for m in metrics if m["length"] == 200 and "mean_sample_perplexity" in m]
    expected = config["prompt_count"]
    if (
        quality
        and all(m.get("quality_count") == expected for m in quality)
        and len(quality) == len([m for m in metrics if m["length"] == 200])
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for gamma in config["gammas"]:
            points = [m for m in quality if m["gamma"] == gamma and m["variant"] != "human"]
            points.sort(
                key=lambda m: (
                    0 if m["variant"] == "baseline" else float(m["variant"].split("-d")[1])
                )
            )
            ax.plot(
                [m["mean_sample_perplexity"] for m in points],
                [m["mean_z"] for m in points],
                marker="o",
                label=f"γ={gamma:g}",
            )
            for point in points:
                if point["variant"] == "baseline" and gamma != config["gammas"][0]:
                    continue
                offset = (5, 7)
                if point["variant"].endswith("-d1"):
                    offset = (5, 10 if gamma == config["gammas"][0] else -15)
                ax.annotate(
                    point["variant"],
                    (point["mean_sample_perplexity"], point["mean_z"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(
            xlabel="Mean per-sample OPT-2.7B conditional perplexity (lower is better)",
            ylabel="Mean z-score",
            title="Detection strength versus fluency proxy",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "quality.png", dpi=160)
        plt.close(fig)
    print(f"Saved figures: {directory}", flush=True)


def archive(config):
    import hashlib
    import shutil

    output = Path(config["output_dir"])
    target = Path("reports") / output.name
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "report.md",
        "examples.md",
        "metrics.csv",
        "evaluation.json",
        "run.json",
        "generation_settings.json",
        "generations.jsonl",
        "quality.jsonl",
        "quality.manifest.json",
    ):
        shutil.copy2(output / name, target / name)
    shutil.copytree(output / "figures", target / "figures", dirs_exist_ok=True)
    shutil.copy2(config["prompts_path"], target / "prompts.jsonl")
    shutil.copy2(
        Path(config["prompts_path"]).with_suffix(".manifest.json"), target / "prompts.manifest.json"
    )
    if (output / "lineage.json").exists():
        shutil.copy2(output / "lineage.json", target / "lineage.json")
    replay = {**config, "output_dir": str(target), "prompts_path": str(target / "prompts.jsonl")}
    write_json(target / "replay.json", replay)
    sources = list(Path("src/watermarking").glob("*.py")) + [Path("uv.lock")]
    write_json(
        target / "source_hashes.json",
        {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    (target / "REPLAY.md").write_text(
        "# Reanalyze saved outputs without loading a model\n\n"
        "From the repository root:\n\n```sh\n"
        f".venv/bin/watermark-lab detect --config {target}/replay.json\n"
        f".venv/bin/watermark-lab evaluate --config {target}/replay.json\n"
        f".venv/bin/watermark-lab plot --config {target}/replay.json\n"
        "```\n\nThe original run manifest retains original paths and settings. "
        "The replay configuration changes only artifact paths. Source hashes describe "
        "the implementation at archival time. C4 attribution and original document "
        "URLs are retained in the prompt records; see `docs/protocol.md`.\n"
    )
    print(f"Archived measured outputs and replay inputs: {target}", flush=True)
