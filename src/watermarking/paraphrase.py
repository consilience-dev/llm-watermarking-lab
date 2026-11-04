"""Small, resumable rewriting experiments against saved OPT completions."""

import csv
import hashlib
import json
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, GenerationConfig

from .core import Detector, WatermarkConfig
from .pipeline import checked_generations, load_model, memory, samples, synchronize, tokenizer_for
from .storage import append_jsonl, environment, fingerprint, load_config, read_jsonl, write_json


def validate(config):
    revision = config["model_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Paraphraser revision must be an immutable 40-character commit")
    if not 1 <= config["prompt_count"] <= 100:
        raise ValueError("Require 1–100 prompts")
    if not config["variants"] or len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("Require distinct source variants")
    if "baseline" not in config["variants"]:
        raise ValueError("Include the unwatermarked baseline")
    if not config["attacks"] or any(not text.strip() for text in config["attacks"].values()):
        raise ValueError("Require named, nonempty rewriting instructions")
    if not 0 < config["temperature"] or not 0 < config["top_p"] <= 1:
        raise ValueError("Invalid sampling settings")
    if config["max_new_tokens"] < 1 or any(n < 2 or n > 200 for n in config["lengths"]):
        raise ValueError("Invalid generation or observation lengths")
    if config["device"] not in ("cpu", "mps") or config["dtype"] not in (
        "float16",
        "bfloat16",
        "float32",
    ):
        raise ValueError("Unsupported device or dtype")


def inputs_for(config):
    source_config = load_config(config["source_config"])
    prompts = samples(source_config)[: config["prompt_count"]]
    if len(prompts) != config["prompt_count"]:
        raise ValueError("Source run has too few prompts")
    records = checked_generations(source_config)
    indexed = {(r["prompt_id"], r["variant"]): r for r in records}
    selected = []
    for prompt in prompts:
        for variant in config["variants"]:
            record = indexed.get((prompt["id"], variant))
            if record is None:
                raise ValueError(f"Missing source variant: {variant}")
            if record["watermark"] and record["watermark"]["gamma"] != config["gamma"]:
                raise ValueError("Source watermark gamma differs from detector gamma")
            selected.append(record)
    return source_config, selected


def tasks(config, sources):
    for index, source in enumerate(sources):
        for attack_index, (attack, instruction) in enumerate(config["attacks"].items()):
            yield (
                source,
                {
                    "prompt_id": source["prompt_id"],
                    "variant": source["variant"],
                    "attack": attack,
                    "instruction": instruction,
                    "input_hash": fingerprint(source),
                    "seed": config["seed"] + index * len(config["attacks"]) + attack_index,
                },
            )


def record_key(row):
    return row["prompt_id"], row["variant"], row["attack"]


def checked_rewrites(output, config, sources, require_complete=True):
    path = Path(output) / "rewrites.jsonl"
    records = read_jsonl(path) if path.exists() else []
    expected = {record_key(task): task for _, task in tasks(config, sources)}
    seen = set()
    for row in records:
        key = record_key(row)
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected rewrite")
        seen.add(key)
        if any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError("Rewrite provenance changed")
        payload = {k: v for k, v in row.items() if k != "record_hash"}
        if row.get("record_hash") != fingerprint(payload):
            raise ValueError("Rewrite record was modified")
    if require_complete and seen != set(expected):
        raise ValueError("Rewrite run is incomplete")
    return records


def score_versions(detector, source_ids, roundtrip_ids, rewritten_ids, lengths):
    versions = {"original": source_ids, "roundtrip": roundtrip_ids, "rewritten": rewritten_ids}
    scores = {}
    for version, ids in versions.items():
        scores[version] = {"full": detector.score(ids)}
        for length in lengths:
            # A shorter rewrite is not a measurement at the requested length.
            scores[version][str(length)] = (
                detector.score(ids[:length]) if len(ids) >= length else None
            )
    return scores


def pair_retention(source_ids, rewritten_ids):
    original = set(pairwise(source_ids))
    rewritten = set(pairwise(rewritten_ids))
    return len(original & rewritten) / len(original) if original else None


def run(config):
    validate(config)
    source_config, sources = inputs_for(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "config": config,
        "source_config": source_config,
        "sources": sources,
        "environment": environment(),
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("paraphrase.py", "core.py", "pipeline.py", "storage.py")
        },
    }
    manifest = output / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError("Run inputs, code, configuration, or environment changed")
    else:
        if (output / "rewrites.jsonl").exists():
            raise ValueError("Rewrite records exist without a manifest")
        write_json(manifest, spec)
    records = checked_rewrites(output, config, sources, require_complete=False)
    completed = {record_key(r) for r in records}
    pending = [
        (source, task)
        for source, task in tasks(config, sources)
        if record_key(task) not in completed
    ]
    if not pending:
        print(f"Verified {len(records)} existing rewrites; no model loaded", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    opt_tokenizer = tokenizer_for(source_config)
    model = load_model(config)
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    generation = GenerationConfig(
        do_sample=True,
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=config["max_new_tokens"],
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    write_json(output / "generation_settings.json", generation.to_dict())
    detector = Detector(
        WatermarkConfig(sources[0]["vocab_size"], config["gamma"], 0, source_config["key"])
    )
    for index, (source, task) in enumerate(pending):
        messages = [
            {"role": "system", "content": task["instruction"]},
            {"role": "user", "content": "Passage to rewrite:\n\n" + source["text"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(config["device"])
        torch.manual_seed(task["seed"])
        if config["device"] == "mps":
            torch.mps.manual_seed(task["seed"])
        synchronize(config["device"])
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(**inputs, generation_config=generation)
        synchronize(config["device"])
        elapsed = time.perf_counter() - start
        ids = result[0, inputs["input_ids"].shape[1] :].cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        opt_ids = opt_tokenizer.encode(text, add_special_tokens=False)
        roundtrip_ids = opt_tokenizer.encode(source["text"], add_special_tokens=False)
        ended_eos = bool(ids and ids[-1] in eos_ids)
        row = {
            **task,
            "messages": messages,
            "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "paraphraser_token_ids": ids,
            "text": text,
            "opt_token_ids": opt_ids,
            "roundtrip_opt_token_ids": roundtrip_ids,
            "ended_eos": ended_eos,
            "hit_token_limit": len(ids) == config["max_new_tokens"] and not ended_eos,
            "elapsed_seconds": elapsed,
            "token_pair_retention": pair_retention(roundtrip_ids, opt_ids),
            "scores": score_versions(
                detector, source["token_ids"], roundtrip_ids, opt_ids, config["lengths"]
            ),
            **memory(config["device"]),
        }
        append_jsonl(output / "rewrites.jsonl", {**row, "record_hash": fingerprint(row)})
        print(
            f"{len(completed) + index + 1}/{len(sources) * len(config['attacks'])} "
            f"{task['prompt_id']} {task['variant']} {task['attack']}: "
            f"{len(opt_ids)} OPT tokens in {elapsed:.1f}s",
            flush=True,
        )


def report(config):
    """Rebuild analysis entirely from saved records; no tokenizer or LLM is loaded."""
    output = Path(config["output_dir"])
    manifest = json.loads((output / "run.json").read_text())
    settings, sources = manifest["config"], manifest["sources"]
    rows = checked_rewrites(output, settings, sources)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"], row["attack"]].append(row)
    metrics = []
    for (variant, attack), group in sorted(grouped.items()):
        for length in settings["lengths"]:
            eligible = [r for r in group if r["scores"]["rewritten"][str(length)] is not None]
            metric = {
                "variant": variant,
                "attack": attack,
                "observed_tokens": length,
                "total": len(group),
                "eligible": len(eligible),
                "short": len(group) - len(eligible),
                "empty": sum(not r["opt_token_ids"] for r in group),
                "token_limit_hits": sum(r["hit_token_limit"] for r in group),
                "mean_output_tokens": mean(len(r["opt_token_ids"]) for r in group),
                "mean_pair_retention": mean(r["token_pair_retention"] for r in group),
            }
            for version in ("original", "roundtrip", "rewritten"):
                scores = [r["scores"][version][str(length)] for r in eligible]
                metric[f"{version}_hits"] = sum(
                    s["z_score"] > settings["threshold"] for s in scores
                )
                metric[f"{version}_mean_z"] = mean(s["z_score"] for s in scores) if scores else None
                metric[f"{version}_mean_pairs"] = (
                    mean(s["num_tokens_scored"] for s in scores) if scores else None
                )
            metrics.append(metric)
    write_json(output / "evaluation.json", {"metrics": metrics})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "# Paraphrasing pilot",
        "",
        (
            f"{len(sources)} saved OPT completions, {settings['prompt_count']} prompts, {len(rows)} rewrites. "
            f"Rewriter: {settings['model']} at `{settings['model_revision']}`; watermark disabled."
        ),
        "",
        (
            "Exploratory pipeline check, not a quantization result. Five prompt groups cannot establish "
            "robustness rates or small false-positive rates. No quality-based filtering is applied. "
            "Meaning preservation requires manual review; token-pair retention measures lexical overlap only."
        ),
        "",
        (
            "All detection uses OPT token IDs, CPU partitions, and repeated-pair deduplication. "
            f"A hit means z > {settings['threshold']:g}. Original = saved token IDs; "
            "roundtrip = original text re-tokenized without rewriting. Both clean columns use the "
            "same eligible subset as the rewrite. Short outputs are counted separately, not silently "
            "scored as if they reached the observation length."
        ),
        "",
        "| Variant | Rewrite | Observed | Eligible/total | Original hits | Roundtrip hits | Rewritten hits | Mean rewritten z |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics:
        z = f"{m['rewritten_mean_z']:.2f}" if m["rewritten_mean_z"] is not None else "—"
        lines.append(
            f"| {m['variant']} | {m['attack']} | {m['observed_tokens']} | "
            f"{m['eligible']}/{m['total']} | {m['original_hits']} | {m['roundtrip_hits']} | "
            f"{m['rewritten_hits']} | {z} |"
        )
    lines += [
        "",
        (
            "Original generations suppress EOS and may end mid-sentence. Rewrites allow EOS. "
            "Consequently their lengths and endings can differ; inspect examples and truncation flags. "
            "The attack instructions request different edit strengths but do not guarantee them. "
            "This pilot does not measure oracle perplexity or semantic similarity with a learned metric."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    source_index = {(s["prompt_id"], s["variant"]): s for s in sources}
    examples = ["# Source passages and all rewrites", ""]
    for row in rows:
        source = source_index[row["prompt_id"], row["variant"]]
        examples += [
            f"## {row['prompt_id']} · {row['variant']} · {row['attack']}",
            "",
            (
                f"OPT tokens: {len(row['opt_token_ids'])}; retained token pairs: "
                f"{row['token_pair_retention']:.1%}; token limit hit: {row['hit_token_limit']}."
            ),
            "",
            "### Source",
            "",
            source["text"],
            "",
            "### Rewrite",
            "",
            row["text"],
            "",
        ]
    (output / "examples.md").write_text("\n".join(examples))
    plot_scores(output, settings, rows)
    # Preserve completed human judgments when the numerical report is rebuilt.
    review = output / "manual_review.csv"
    if not review.exists():
        with review.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["prompt_id", "variant", "attack", "meaning", "notes"]
            )
            writer.writeheader()
            writer.writerows(
                {k: row[k] for k in ("prompt_id", "variant", "attack")} for row in rows
            )
    print(f"Saved offline analysis and review worksheet: {output}", flush=True)


def plot_scores(output, settings, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = 100 if 100 in settings["lengths"] else settings["lengths"][0]
    observed = str(length)
    values = [0, settings["threshold"]]
    for row in rows:
        if row["scores"]["rewritten"][observed] is not None:
            values.extend(row["scores"][v][observed]["z_score"] for v in ("roundtrip", "rewritten"))
    low, high = min(values) - 1, max(values) + 1
    figure, axes = plt.subplots(1, len(settings["attacks"]), figsize=(10, 4), squeeze=False)
    for axis, attack in zip(axes[0], settings["attacks"], strict=True):
        axis.plot([low, high], [low, high], color="0.7", linestyle="--", linewidth=1)
        axis.axhline(settings["threshold"], color="0.5", linestyle=":")
        axis.axvline(settings["threshold"], color="0.5", linestyle=":")
        for variant in settings["variants"]:
            selected = [
                r
                for r in rows
                if r["attack"] == attack
                and r["variant"] == variant
                and r["scores"]["rewritten"][observed] is not None
            ]
            axis.scatter(
                [r["scores"]["roundtrip"][observed]["z_score"] for r in selected],
                [r["scores"]["rewritten"][observed]["z_score"] for r in selected],
                label=f"{variant} (n={len(selected)})",
                alpha=0.8,
            )
        axis.set(
            xlim=(low, high),
            ylim=(low, high),
            title=f"{attack.capitalize()} rewriting",
            xlabel="Before rewriting: roundtrip z",
            ylabel="After rewriting: z",
        )
        axis.legend(fontsize=8)
    figure.suptitle(f"Same-length detection ({length} OPT tokens); each point is one passage")
    figure.tight_layout()
    figure.savefig(Path(output) / "scores.png", dpi=160)
    plt.close(figure)
