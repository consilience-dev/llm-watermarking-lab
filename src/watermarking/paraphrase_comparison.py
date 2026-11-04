"""Compare fixed rewriting tasks without treating short outputs as full-length texts."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .paraphrase import checked_rewrites, record_key
from .storage import fingerprint, write_json


def assert_comparable(first, second):
    ignored = {"model", "model_revision", "output_dir"}
    a, b = first["config"], second["config"]
    if {k: v for k, v in a.items() if k not in ignored} != {
        k: v for k, v in b.items() if k not in ignored
    }:
        raise ValueError("Rewrite instructions, seeds, or experimental settings differ")
    if first["sources"] != second["sources"] or first["source_config"] != second["source_config"]:
        raise ValueError("Source records or source settings differ")
    if first["source_hashes"] != second["source_hashes"]:
        raise ValueError("Generation or scoring implementation differs")
    if first["environment"] != second["environment"]:
        raise ValueError("Runtime environments differ")


def paired_counts(first, second, length, threshold):
    left, right = ({record_key(r): r for r in rows} for rows in (first, second))
    if len(left) != len(first) or len(right) != len(second) or left.keys() != right.keys():
        raise ValueError("Rewrite tasks are missing or duplicated")
    prefix = str(length)
    metrics = []
    groups = sorted({(r["variant"], r["attack"]) for r in first})
    for variant, attack in groups:
        keys = [k for k in left if k[1:] == (variant, attack)]
        eligible_a = [k for k in keys if left[k]["scores"]["rewritten"][prefix] is not None]
        eligible_b = [k for k in keys if right[k]["scores"]["rewritten"][prefix] is not None]
        common = [k for k in eligible_a if k in eligible_b]
        for key in common:
            if left[key]["scores"]["original"][prefix] != right[key]["scores"]["original"][prefix]:
                raise ValueError("Clean scores differ for identical source passages")
        metrics.append(
            {
                "variant": variant,
                "attack": attack,
                "observed_tokens": length,
                "total": len(keys),
                "first_eligible": len(eligible_a),
                "second_eligible": len(eligible_b),
                "common_eligible": len(common),
                "clean_hits": sum(
                    left[k]["scores"]["original"][prefix]["z_score"] > threshold for k in common
                ),
                "first_hits": sum(
                    left[k]["scores"]["rewritten"][prefix]["z_score"] > threshold for k in common
                ),
                "second_hits": sum(
                    right[k]["scores"]["rewritten"][prefix]["z_score"] > threshold for k in common
                ),
            }
        )
    return metrics


def review_counts(path, records):
    with Path(path).open(newline="") as handle:
        reviews = list(csv.DictReader(handle))
    expected = {record_key(r) for r in records}
    actual = {record_key(r) for r in reviews}
    allowed = {"no_clear_change", "changed_or_omitted", "unclear_source", "refusal"}
    if expected != actual or len(reviews) != len(expected):
        raise ValueError("Qualitative review is missing cases or has duplicate cases")
    if any(r["meaning"] not in allowed or not r["reviewer"] or not r["notes"] for r in reviews):
        raise ValueError("Qualitative review is incomplete")
    counts = Counter(r["meaning"] for r in reviews)
    return {label: counts[label] for label in sorted(allowed)}


def compare(config):
    paths = [Path(config[k]) for k in ("first_archive", "second_archive")]
    manifests = [json.loads((p / "run.json").read_text()) for p in paths]
    assert_comparable(*manifests)
    settings = manifests[0]["config"]
    length = config["observed_tokens"]
    if length not in settings["lengths"]:
        raise ValueError("Observation length was not scored in both pilots")
    records = [
        checked_rewrites(p, m["config"], m["sources"])
        for p, m in zip(paths, manifests, strict=True)
    ]
    reviews = [
        review_counts(p / "manual_review.csv", r) for p, r in zip(paths, records, strict=True)
    ]
    metrics = paired_counts(*records, length, settings["threshold"])
    names = [m["config"]["model"] for m in manifests]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "comparison.json",
        {
            "config": config,
            "models": names,
            "source_records_hash": fingerprint(manifests[0]["sources"]),
            "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_hashes": {
                str(p / n): hashlib.sha256((p / n).read_bytes()).hexdigest()
                for p in paths
                for n in ("run.json", "rewrites.jsonl", "manual_review.csv")
            },
            "paired_metrics": metrics,
            "qualitative_counts": reviews,
        },
    )
    lines = [
        "# Two local rewriters: exploratory comparison",
        "",
        f"First: `{names[0]}`. Second: `{names[1]}`.",
        "",
        (
            "Both pilots use the same 30 tasks, source records, instructions, seeds, generation settings, "
            "and detector implementation. This is a comparison of two complete model checkpoints, "
            "not an isolated model-size effect."
        ),
        "",
        (
            f"Detection below uses {length} observed OPT tokens, repeated-pair deduplication, "
            f"and z > {settings['threshold']:g}. Paired counts include only passages long enough "
            "in both pilots. Individual eligibility counts show the excluded short outputs. "
            "No qualitative label is used to filter the numerical table."
        ),
        "",
        "| Variant | Rewrite | First eligible/total | Second eligible/total | Common passages | Clean hits | First hits | Second hits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics:
        lines.append(
            f"| {m['variant']} | {m['attack']} | {m['first_eligible']}/{m['total']} | "
            f"{m['second_eligible']}/{m['total']} | {m['common_eligible']} | {m['clean_hits']} | "
            f"{m['first_hits']} | {m['second_hits']} |"
        )
    lines += [
        "",
        "## Provisional qualitative inspection",
        "",
        "| Label | First rewriter | Second rewriter |",
        "| --- | ---: | ---: |",
    ]
    for label in reviews[0]:
        lines.append(f"| {label} | {reviews[0][label]} | {reviews[1][label]} |")
    lines += [
        "",
        (
            "These are model-assisted screening labels, not an independent human study or a validated "
            "semantic metric. No clear change means no obvious issue was found, not certified "
            "equivalence. The inputs were already inspected during the first pilot. Review notes "
            "and all outputs remain in the two archives. Source factual accuracy was not assessed."
        ),
        "",
        (
            "With five prompt groups, different eligible subsets, and fidelity failures, these "
            "counts cannot establish general attack success rates, model superiority, or quantization "
            "effects. Lower detector scores alone are not evidence of a successful meaning-preserving attack."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Saved paired comparison: {output}", flush=True)
