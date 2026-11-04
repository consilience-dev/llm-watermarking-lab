import gc
import gzip
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

from .core import Detector, SoftWatermark, WatermarkConfig
from .storage import append_jsonl, environment, fingerprint, read_jsonl, write_json


def tokenizer_for(config, oracle=False):
    key = "oracle" if oracle else "model"
    return AutoTokenizer.from_pretrained(config[key], revision=config[f"{key}_revision"])


def prepare(config):
    """Freeze 100 eligible documents in shard order; pilot uses the first ten."""
    target = Path(config["prompts_path"])
    spec = {
        k: config[k]
        for k in (
            "model",
            "model_revision",
            "dataset",
            "dataset_revision",
            "dataset_file",
            "prompt_tokens",
            "completion_tokens",
        )
    }
    metadata = target.with_suffix(".manifest.json")
    if target.exists():
        if not metadata.exists() or json.loads(metadata.read_text())["spec"] != spec:
            raise ValueError(
                "Existing prompt manifest does not match config; use a new prompts_path"
            )
        rows = read_jsonl(target)
        expected = json.loads(metadata.read_text())["content_hash"]
        if len(rows) != 100 or fingerprint(rows) != expected:
            raise ValueError("Prompt file is incomplete or modified")
        print(f"Verified existing 100-prompt manifest: {target}", flush=True)
        return
    tokenizer = tokenizer_for(config)
    shard = hf_hub_download(
        repo_id=config["dataset"],
        repo_type="dataset",
        revision=config["dataset_revision"],
        filename=config["dataset_file"],
    )
    rows, seen = [], set()
    total = config["prompt_tokens"] + config["completion_tokens"]
    with gzip.open(shard, "rt") as handle:
        for index, line in enumerate(handle):
            document = json.loads(line)
            content_hash = hashlib.sha256(document["text"].encode()).hexdigest()
            if content_hash in seen:
                continue
            ids = tokenizer.encode(document["text"], add_special_tokens=False)
            if len(ids) < total:
                continue
            seen.add(content_hash)
            prompt = ids[: config["prompt_tokens"]]
            human = ids[config["prompt_tokens"] : total]
            rows.append(
                {
                    "id": f"c4-validation-{index}",
                    "source_index": index,
                    "source_sha256": content_hash,
                    "url": document.get("url"),
                    "prompt_ids": prompt,
                    "human_ids": human,
                    "prompt_text": tokenizer.decode(prompt),
                    "human_text": tokenizer.decode(human),
                }
            )
            if len(rows) == 100:
                break
    if len(rows) != 100:
        raise RuntimeError("Shard did not supply 100 eligible distinct documents")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(metadata, {"spec": spec, "content_hash": fingerprint(rows), "count": len(rows)})
    temporary.replace(target)
    print(f"Prepared {len(rows)} fixed prompt/human pairs: {target}", flush=True)


def load_model(config, oracle=False):
    device = config["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process; run with Metal access or choose cpu"
        )
    key = "oracle" if oracle else "model"
    model = AutoModelForCausalLM.from_pretrained(
        config[key],
        revision=config[f"{key}_revision"],
        dtype=getattr(torch, config["dtype"]),
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def memory(device):
    if device == "mps":
        return {
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        }
    return {}


def release(model, device):
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def samples(config):
    rows = read_jsonl(config["prompts_path"])[: config["prompt_count"]]
    if len(rows) != config["prompt_count"]:
        raise ValueError("Not enough prepared prompts; run prepare first")
    return rows


def variants(config):
    yield "baseline", None
    for gamma in config["gammas"]:
        for delta in config["deltas"]:
            yield f"g{gamma:g}-d{delta:g}", (gamma, delta)


def checked_generations(config, require_complete=True):
    rows = samples(config)
    output = Path(config["output_dir"])
    manifest = json.loads((output / "run.json").read_text())
    ignored = {"output_dir", "prompts_path"}
    scientific_config = {k: v for k, v in config.items() if k not in ignored}
    saved_config = {k: v for k, v in manifest["config"].items() if k not in ignored}
    if scientific_config != saved_config or manifest["prompts_hash"] != fingerprint(rows):
        raise ValueError("Run provenance does not match configuration or prompt content")
    path = output / "generations.jsonl"
    records = read_jsonl(path) if path.exists() else []
    expected = {(p["id"], name) for p in rows for name, _ in variants(config)}
    actual = {(r["prompt_id"], r["variant"]) for r in records}
    if len(actual) != len(records) or not actual <= expected:
        raise ValueError("Duplicate or unexpected generation records")
    if require_complete and actual != expected:
        raise ValueError("Generation run is incomplete")
    indices = {p["id"]: index for index, p in enumerate(rows)}
    settings = dict(variants(config))
    vocab_sizes = {r["vocab_size"] for r in records}
    if len(vocab_sizes) > 1:
        raise ValueError("Inconsistent saved vocabulary sizes")
    for record in records:
        vocab = record["vocab_size"]
        parameters = settings[record["variant"]]
        expected_watermark = (
            asdict(WatermarkConfig(vocab, *parameters, config["key"])) if parameters else None
        )
        if (
            record["watermark"] != expected_watermark
            or record["seed"] != config["seed"] + indices[record["prompt_id"]]
            or len(record["token_ids"]) != config["completion_tokens"]
            or any(type(t) is not int or not 0 <= t < vocab for t in record["token_ids"])
        ):
            raise ValueError("Saved generation settings or token IDs are inconsistent")
    return records


def extend(config, source_config):
    """Reuse a completed nested pilot without changing any experimental settings."""
    ignored = {"prompt_count", "output_dir"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in source_config.items() if k not in ignored
    }:
        raise ValueError("Only prompt_count and output_dir may differ when extending a run")
    if config["prompt_count"] <= source_config["prompt_count"]:
        raise ValueError("The target must contain more prompts than the source")
    source = Path(source_config["output_dir"])
    source_rows = samples(source_config)
    expected_spec = {
        "config": source_config,
        "prompts_hash": fingerprint(source_rows),
        "environment": environment(),
    }
    if json.loads((source / "run.json").read_text()) != expected_spec:
        raise ValueError("Source run does not match current inputs or environment")
    records = checked_generations(source_config)
    expected = {(p["id"], name) for p in source_rows for name, _ in variants(source_config)}
    if {(r["prompt_id"], r["variant"]) for r in records} != expected or len(records) != len(
        expected
    ):
        raise ValueError("Source run is incomplete or contains duplicate records")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Extension target must be empty; subsequent runs resume with generate")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "run.json",
        {
            "config": config,
            "prompts_hash": fingerprint(samples(config)),
            "environment": environment(),
        },
    )
    for record in records:
        append_jsonl(output / "generations.jsonl", record)
    write_json(
        output / "lineage.json",
        {
            "source_output_dir": str(source),
            "imported_records": len(records),
            "source_records_hash": fingerprint(records),
        },
    )
    print(f"Imported {len(records)} validated pilot completions; resume with generate", flush=True)


def generate(config):
    rows = samples(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "generations.jsonl"
    spec = {"config": config, "prompts_hash": fingerprint(rows), "environment": environment()}
    manifest_path = output / "run.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != spec:
            raise ValueError(
                "Run settings, prompts, or environment changed; choose a new output_dir"
            )
    elif path.exists():
        raise ValueError("Generation file exists without run metadata")
    else:
        write_json(manifest_path, spec)
    existing = checked_generations(config, require_complete=False)
    completed = {(row["prompt_id"], row["variant"]) for row in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate generation records")
    if len(completed) == len(rows) * len(list(variants(config))):
        print("All generations already complete", flush=True)
        return
    tokenizer = tokenizer_for(config)
    model = load_model(config)
    # OPT pads its output embedding vocabulary beyond the tokenizer's vocabulary.
    # Partition actual tokenizer IDs and suppress unreachable padded logits.
    vocab_size = len(tokenizer)
    output_vocab = model.get_output_embeddings().weight.shape[0]
    if sorted(tokenizer.get_vocab().values()) != list(range(vocab_size)):
        raise ValueError("This implementation requires contiguous tokenizer IDs")

    class ProcessorAdapter:
        def __init__(self, watermark):
            self.watermark = watermark

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            if self.watermark is not None:
                scores[:, :vocab_size] = self.watermark(input_ids, scores[:, :vocab_size])
            scores[:, vocab_size:] = -float("inf")
            return scores

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=config["completion_tokens"],
        min_new_tokens=config["completion_tokens"],
        eos_token_id=None,
        forced_eos_token_id=None,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        suppress_tokens=[tokenizer.eos_token_id],
    )
    write_json(
        output / "generation_settings.json",
        {
            "generation_config": generation_config.to_dict(),
            "tokenizer_vocab_size": vocab_size,
            "output_vocab_size": output_vocab,
            "partition_device": "cpu",
            "first_generated_token_detection": "omitted; standalone completion detection",
        },
    )
    try:
        for index, row in enumerate(rows):
            inputs = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=config["device"])
            attention = torch.ones_like(inputs)
            for name, parameters in variants(config):
                if (row["id"], name) in completed:
                    continue
                seed = config["seed"] + index
                torch.manual_seed(seed)
                if config["device"] == "mps":
                    torch.mps.manual_seed(seed)
                watermark_config = (
                    WatermarkConfig(vocab_size, *parameters, config["key"]) if parameters else None
                )
                processor = ProcessorAdapter(
                    SoftWatermark(watermark_config) if parameters else None
                )
                synchronize(config["device"])
                start = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(
                        input_ids=inputs,
                        attention_mask=attention,
                        generation_config=generation_config,
                        logits_processor=LogitsProcessorList([processor]),
                    )
                synchronize(config["device"])
                elapsed = time.perf_counter() - start
                ids = result[0, inputs.shape[1] :].cpu().tolist()
                if len(ids) != config["completion_tokens"] or any(t >= vocab_size for t in ids):
                    raise RuntimeError("Unexpected completion length or token ID")
                record = {
                    "prompt_id": row["id"],
                    "variant": name,
                    "seed": seed,
                    "token_ids": ids,
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": len(ids) / elapsed,
                    "watermark": asdict(watermark_config) if watermark_config else None,
                    "vocab_size": vocab_size,
                    **memory(config["device"]),
                }
                append_jsonl(path, record)
                print(
                    f"{index + 1}/{len(rows)} {name}: {len(ids) / elapsed:.1f} tokens/s", flush=True
                )
    finally:
        release(model, config["device"])


def detect(config):
    output = Path(config["output_dir"])
    generations = checked_generations(config)
    prompt_rows = samples(config)
    expected = {(p["id"], name) for p in prompt_rows for name, _ in variants(config)}
    actual = {(r["prompt_id"], r["variant"]) for r in generations}
    if actual != expected or len(actual) != len(generations):
        raise ValueError("Generation run is incomplete or contains duplicate/unexpected records")
    records = list(generations)
    vocab_size = generations[0]["vocab_size"]
    records += [
        {"prompt_id": row["id"], "variant": "human", "token_ids": row["human_ids"]}
        for row in prompt_rows
    ]
    results = []
    for gamma in config["gammas"]:
        detector = Detector(WatermarkConfig(vocab_size, gamma, 0, config["key"]))
        for row in records:
            if row["variant"] not in ("baseline", "human") and row["watermark"]["gamma"] != gamma:
                continue
            for length in config["lengths"]:
                for deduplicate in (True, False):
                    result = detector.score(row["token_ids"][:length], deduplicate=deduplicate)
                    results.append(
                        {
                            "prompt_id": row["prompt_id"],
                            "variant": row["variant"],
                            "generation_watermark": row.get("watermark"),
                            "gamma": gamma,
                            "length": length,
                            **result,
                        }
                    )
    target = output / "detections.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in results))
    temporary.replace(target)
    print(f"Saved {len(results)} detection records", flush=True)


def quality(config):
    output = Path(config["output_dir"])
    generations = checked_generations(config)
    prompts = {r["id"]: r for r in samples(config)}
    expected = {(p, name) for p in prompts for name, _ in variants(config)}
    actual = {(r["prompt_id"], r["variant"]) for r in generations}
    if actual != expected or len(actual) != len(generations):
        raise ValueError("Generate all conditions before measuring quality")
    records = generations + [
        {"prompt_id": row["id"], "variant": "human", "token_ids": row["human_ids"]}
        for row in prompts.values()
    ]
    target = output / "quality.jsonl"
    metadata_path = output / "quality.manifest.json"
    spec = {"config": config, "records_hash": fingerprint(records), "environment": environment()}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != spec:
            raise ValueError("Quality inputs or environment changed; use a new output directory")
    elif target.exists():
        raise ValueError("Quality file exists without provenance manifest")
    else:
        write_json(metadata_path, spec)
    existing = read_jsonl(target) if target.exists() else []
    done = {(r["prompt_id"], r["variant"]) for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate quality records")
    if len(done) == len(records):
        print("All quality scores already complete", flush=True)
        return
    tokenizer = tokenizer_for(config)
    oracle_tokenizer = tokenizer_for(config, oracle=True)
    if tokenizer.get_vocab() != oracle_tokenizer.get_vocab():
        raise ValueError("Oracle tokenizer differs; conditional scoring needs aligned token IDs")
    model = load_model(config, oracle=True)
    try:
        for index, row in enumerate(records):
            if (row["prompt_id"], row["variant"]) in done:
                continue
            prefix = prompts[row["prompt_id"]]["prompt_ids"]
            ids = torch.tensor([prefix + row["token_ids"]], device=config["device"])
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).logits
                # Position p-1 predicts the first completion token at position p.
                completion_logits = logits[:, len(prefix) - 1 : -1].float()
                labels = ids[:, len(prefix) :]
                nll = torch.nn.functional.cross_entropy(
                    completion_logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="mean",
                ).item()
            synchronize(config["device"])
            if not __import__("math").isfinite(nll):
                raise RuntimeError("Nonfinite oracle loss")
            append_jsonl(
                target,
                {
                    "prompt_id": row["prompt_id"],
                    "variant": row["variant"],
                    "mean_nll": nll,
                    "num_tokens": len(row["token_ids"]),
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            print(f"Quality {index + 1}/{len(records)} {row['variant']}: NLL={nll:.3f}", flush=True)
    finally:
        release(model, config["device"])
