import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from watermarking import paraphrase
from watermarking.core import Detector, WatermarkConfig
from watermarking.storage import read_jsonl


def test_roundtrip_and_short_rewrite_scoring():
    detector = Detector(WatermarkConfig(16))
    original = [1, 2, 3, 4] * 25
    roundtrip = [1, 5, 3, 4] * 25
    rewritten = [2, 3] * 20
    scores = paraphrase.score_versions(detector, original, roundtrip, rewritten, [25, 50, 100])
    assert scores["original"]["50"] == detector.score(original[:50])
    assert scores["roundtrip"]["full"] == detector.score(roundtrip)
    assert scores["rewritten"]["25"] == detector.score(rewritten[:25])
    assert scores["rewritten"]["50"] is None
    assert scores["rewritten"]["100"] is None
    assert paraphrase.pair_retention([1, 2, 3, 4], [1, 2, 5, 6]) == pytest.approx(1 / 3)
    empty = paraphrase.score_versions(detector, original, roundtrip, [], [25])
    assert empty["rewritten"]["full"]["status"] == "insufficient_text"


def test_rewrite_pipeline_resume_provenance_and_offline_report(tmp_path, monkeypatch):
    config = json.loads(Path("configs/paraphrase_pilot.json").read_text())
    config.update(
        device="cpu",
        dtype="float32",
        prompt_count=1,
        variants=["baseline", "g0.25-d1"],
        output_dir=str(tmp_path),
    )
    sources = [
        {
            "prompt_id": "p0",
            "variant": variant,
            "token_ids": [1, 3] * 100,
            "text": "source",
            "vocab_size": 16,
        }
        for variant in config["variants"]
    ]
    monkeypatch.setattr(paraphrase, "inputs_for", lambda c: ({"key": 15485863}, sources))

    class QwenTokenizer:
        pad_token_id = 8
        bos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] in config["attacks"].values()
            assert messages[1]["content"].endswith("source")
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[8, 9]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }
            )

        def decode(self, ids, **kwargs):
            assert ids == [50, 51, 99]
            return "rewritten"

    class OptTokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3] * (30 if text == "rewritten" else 67)

    class Model:
        generation_config = SimpleNamespace(eos_token_id=99)

        def generate(self, **kwargs):
            assert kwargs["generation_config"].eos_token_id == 99
            assert "logits_processor" not in kwargs
            return torch.tensor([[8, 9, 50, 51, 99]])

    monkeypatch.setattr(
        paraphrase.AutoTokenizer, "from_pretrained", lambda *a, **k: QwenTokenizer()
    )
    monkeypatch.setattr(paraphrase, "tokenizer_for", lambda c: OptTokenizer())
    monkeypatch.setattr(paraphrase, "load_model", lambda c: Model())
    paraphrase.run(config)
    records = read_jsonl(tmp_path / "rewrites.jsonl")
    assert len(records) == 4
    assert records[0]["paraphraser_token_ids"] == [50, 51, 99]
    assert max(records[0]["opt_token_ids"]) < 16
    assert records[0]["ended_eos"] and not records[0]["hit_token_limit"]
    assert records[0]["scores"]["rewritten"]["100"] is None
    first = (tmp_path / "rewrites.jsonl").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Model loading is unnecessary")

    monkeypatch.setattr(paraphrase, "load_model", forbidden)
    paraphrase.run(config)
    assert (tmp_path / "rewrites.jsonl").read_bytes() == first
    paraphrase.report({"output_dir": str(tmp_path)})
    metrics = json.loads((tmp_path / "evaluation.json").read_text())["metrics"]
    assert all(
        m["eligible"] == 0 and m["short"] == 1 for m in metrics if m["observed_tokens"] >= 100
    )
    assert "0/1" in (tmp_path / "report.md").read_text()
    (tmp_path / "manual_review.csv").write_text("preserve completed review\n")
    paraphrase.report({"output_dir": str(tmp_path)})
    assert (tmp_path / "manual_review.csv").read_text() == "preserve completed review\n"
    with pytest.raises(ValueError, match="changed"):
        paraphrase.run({**config, "seed": 7})
    records[0]["text"] = "tampered"
    (tmp_path / "rewrites.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="modified"):
        paraphrase.report({"output_dir": str(tmp_path)})
