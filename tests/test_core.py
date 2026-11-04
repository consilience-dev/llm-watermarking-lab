import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

from watermarking.core import Detector, Greenlist, SoftWatermark, WatermarkConfig


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).parents[1] / "vendor/lm_watermarking/watermark_processor.py"
    stub = types.ModuleType("normalizers")
    stub.normalization_strategy_lookup = lambda name: (_ for _ in ()).throw(
        AssertionError("Reference tests must not request text normalization")
    )
    previous = sys.modules.get("normalizers")
    sys.modules["normalizers"] = stub
    try:
        spec = importlib.util.spec_from_file_location("upstream_watermark_processor", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            del sys.modules["normalizers"]
        else:
            sys.modules["normalizers"] = previous


@pytest.mark.parametrize("gamma", [0.25, 0.5])
@pytest.mark.parametrize("delta", [0, 1, 2, 5])
def test_reference_logits_and_detection(reference, gamma, delta):
    config = WatermarkConfig(64, gamma, delta)
    local = SoftWatermark(config)
    upstream = reference.WatermarkLogitsProcessor(
        vocab=list(range(64)),
        gamma=gamma,
        delta=delta,
        hash_key=config.key,
    )
    inputs = torch.tensor([[2, 7, 8, 9], [1, 3, 3, 2]])
    logits = torch.arange(128, dtype=torch.float32).reshape(2, 64) / 10
    assert torch.equal(local(inputs, logits), upstream(inputs, logits.clone()))
    upstream_detector = reference.WatermarkDetector(
        vocab=list(range(64)),
        gamma=gamma,
        delta=delta,
        device=torch.device("cpu"),
        tokenizer=object(),
        normalizers=[],
    )
    tokens = torch.tensor([2, 7, 8, 7, 8, 1, 1, 2, 0, 63])
    for deduplicate in (True, False):
        upstream_detector.ignore_repeated_bigrams = deduplicate
        expected = upstream_detector._score_sequence(tokens)
        actual = Detector(config).score(tokens, deduplicate)
        for key in expected:
            assert actual[key] == pytest.approx(expected[key])


def test_partition_is_local_deterministic_and_cache_bounded():
    config = WatermarkConfig(32)
    partition = Greenlist(config, cache_size=2)
    torch.manual_seed(17)
    state = torch.get_rng_state().clone()
    first = partition.ids(4).clone()
    partition.ids(5)
    partition.ids(6)
    assert len(partition.cache) == 2
    assert torch.equal(first, partition.ids(4))
    assert torch.equal(state, torch.get_rng_state())
    assert first.unique().numel() == 8
    assert torch.equal(first, Greenlist(config).ids(4))


def test_bias_and_zero_bias_do_not_mutate_input():
    inputs = torch.tensor([[1]])
    logits = torch.zeros(1, 32)
    biased = SoftWatermark(WatermarkConfig(32))(inputs, logits)
    assert (biased == 2).sum() == 8
    assert (biased == 0).sum() == 24
    assert not logits.any()
    assert torch.equal(SoftWatermark(WatermarkConfig(32, delta=0))(inputs, logits), logits)


def test_hand_calculated_scores_and_repeated_pairs():
    detector = Detector(WatermarkConfig(8, gamma=0.5))
    detector.greenlist.ids = lambda previous: torch.tensor([0, 1, 2, 3])
    score = detector.score([7, 1, 2, 6, 1, 2], deduplicate=False)
    assert score["num_green_tokens"] == 4
    assert score["num_tokens_scored"] == 5
    z = (4 - 2.5) / math.sqrt(1.25)
    assert score["z_score"] == pytest.approx(z)
    assert score["p_value"] == pytest.approx(math.erfc(z / math.sqrt(2)) / 2)
    assert detector.score([7, 1, 2, 6, 1, 2])["num_tokens_scored"] == 4
    assert "delta" not in score["detector_config"]


@pytest.mark.parametrize("tokens", [[], [1]])
def test_insufficient_text(tokens):
    result = Detector(WatermarkConfig(32)).score(tokens)
    assert result["status"] == "insufficient_text"
    assert result["z_score"] is None
    assert result["p_value"] is None


def test_invalid_inputs():
    with pytest.raises(ValueError):
        WatermarkConfig(4, gamma=0)
    with pytest.raises(ValueError):
        WatermarkConfig(4, gamma=0.01)
    with pytest.raises(ValueError):
        Detector(WatermarkConfig(8)).score([0, 8])


def test_synthetic_biased_samples_have_stronger_signal():
    config = WatermarkConfig(128)
    generator = torch.Generator().manual_seed(42)
    greenlist = Greenlist(config)
    null_scores, biased_scores = [], []
    for _ in range(50):
        null = torch.randint(0, 128, (201,), generator=generator).tolist()
        biased = [null[0]]
        for _ in range(200):
            green = greenlist.ids(biased[-1])
            biased.append(int(green[torch.randint(0, len(green), (), generator=generator)]))
        null_scores.append(Detector(config).score(null)["z_score"])
        biased_scores.append(Detector(config).score(biased)["z_score"])
    assert abs(sum(null_scores) / len(null_scores)) < 0.5
    assert min(biased_scores) > 10
