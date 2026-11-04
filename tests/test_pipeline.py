import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from watermarking import analysis, pipeline
from watermarking.core import SoftWatermark, WatermarkConfig
from watermarking.storage import append_jsonl, load_config, read_jsonl


def test_reference_file_integrity():
    path = Path(__file__).parents[1] / "vendor/lm_watermarking/watermark_processor.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "512c40644bc9e9932a8674bbf13046c1a4e92db429cff92afc9e90d2226896fc"
    )


def test_tiny_opt_full_pipeline_and_resume(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs/pilot.json")
    config.update(
        {
            "prompt_count": 2,
            "device": "cpu",
            "dtype": "float32",
            "output_dir": str(tmp_path / "run"),
            "prompts_path": str(tmp_path / "prompts.jsonl"),
        }
    )
    for i in range(2):
        append_jsonl(
            config["prompts_path"],
            {
                "id": f"p{i}",
                "prompt_ids": [2, 4, 5],
                "human_ids": [3, 4, 5, 6] * 50,
                "prompt_text": "tiny prompt",
                "human_text": "tiny human control",
            },
        )

    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 0

        def __len__(self):
            return 16

        def get_vocab(self):
            return {str(i): i for i in range(16)}

        def decode(self, ids, **kwargs):
            return " ".join(str(t) for t in ids)

    def tiny_model(*args, **kwargs):
        torch.manual_seed(1)
        return OPTForCausalLM(
            OPTConfig(
                vocab_size=20,
                hidden_size=8,
                ffn_dim=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                max_position_embeddings=512,
                dropout=0,
                attention_dropout=0,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=0,
                _attn_implementation="eager",
            )
        ).eval()

    monkeypatch.setattr(pipeline, "tokenizer_for", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(pipeline, "load_model", tiny_model)
    pipeline.generate(config)
    target = tmp_path / "run/generations.jsonl"
    first = target.read_bytes()
    pipeline.generate(config)
    assert target.read_bytes() == first
    assert len(read_jsonl(target)) == 14
    for row in read_jsonl(target):
        assert len(row["token_ids"]) == 200
        assert max(row["token_ids"]) < 16
        assert 1 not in row["token_ids"]
    pipeline.detect(config)
    pipeline.quality(config)
    pipeline.quality(config)
    assert len(read_jsonl(tmp_path / "run/quality.jsonl")) == 16
    analysis.evaluate(config)
    analysis.plot(config)
    evaluation = json.loads((tmp_path / "run/evaluation.json").read_text())
    assert len(evaluation["metrics"]) == 80
    assert (tmp_path / "run/figures/quality.png").exists()
    assert "Quality scoring incomplete" not in (tmp_path / "run/report.md").read_text()
    drifted = {**config, "seed": config["seed"] + 1}
    with pytest.raises(ValueError, match="changed"):
        pipeline.generate(drifted)
    for i in range(2, 4):
        append_jsonl(
            config["prompts_path"],
            {
                "id": f"p{i}",
                "prompt_ids": [2, 4, 5],
                "human_ids": [3, 4, 5, 6] * 50,
                "prompt_text": "tiny prompt",
                "human_text": "tiny human control",
            },
        )
    extended = {**config, "prompt_count": 4, "output_dir": str(tmp_path / "extended")}
    pipeline.extend(extended, config)
    imported = read_jsonl(tmp_path / "extended/generations.jsonl")
    assert imported == read_jsonl(target)
    with pytest.raises(ValueError, match="Only prompt_count"):
        pipeline.extend({**extended, "key": 7}, config)
    with pytest.raises(ValueError, match="empty"):
        pipeline.extend(extended, config)
    records = read_jsonl(target)
    records[0]["seed"] += 1
    target.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="inconsistent"):
        pipeline.detect(config)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU unavailable")
def test_mps_logits_and_sampling_smoke():
    config = WatermarkConfig(64)
    ids = torch.tensor([[1, 3, 5]])
    logits = torch.linspace(-1, 1, 64).reshape(1, -1)
    expected = SoftWatermark(config)(ids, logits)
    actual = SoftWatermark(config)(ids.to("mps"), logits.to("mps"))
    assert torch.equal(actual.cpu(), expected)
    with torch.inference_mode():
        model = (
            OPTForCausalLM(
                OPTConfig(
                    vocab_size=64,
                    hidden_size=16,
                    ffn_dim=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    max_position_embeddings=64,
                    _attn_implementation="eager",
                )
            )
            .eval()
            .to(device="mps", dtype=torch.float16)
        )
        result = model.generate(
            ids.to("mps"),
            attention_mask=torch.ones_like(ids, device="mps"),
            max_new_tokens=5,
            min_new_tokens=5,
            do_sample=True,
            top_k=0,
            logits_processor=[SoftWatermark(config)],
            pad_token_id=1,
        )
    assert result.shape == (1, 8)
