from copy import deepcopy

import pytest

from watermarking.paraphrase_comparison import assert_comparable, paired_counts


def sample(prompt, rewritten):
    return {
        "prompt_id": prompt,
        "variant": "g0.25-d2",
        "attack": "strong",
        "scores": {
            "original": {"100": {"z_score": 6.0}},
            "rewritten": {"100": None if rewritten is None else {"z_score": rewritten}},
        },
    }


def test_comparison_uses_common_passages_and_strict_threshold():
    first = [sample("p0", 4.0), sample("p1", None)]
    second = [sample("p0", 4.1), sample("p1", 7.0)]
    metric = paired_counts(first, second, 100, 4)[0]
    assert metric["first_eligible"] == 1
    assert metric["second_eligible"] == 2
    assert metric["common_eligible"] == 1
    assert metric["clean_hits"] == 1
    assert metric["first_hits"] == 0
    assert metric["second_hits"] == 1
    with pytest.raises(ValueError, match="missing or duplicated"):
        paired_counts(first, second[:1], 100, 4)
    second[0]["scores"]["original"]["100"]["z_score"] = 5.0
    with pytest.raises(ValueError, match="Clean scores differ"):
        paired_counts(first, second, 100, 4)


def test_comparison_rejects_changed_protocol_or_inputs():
    first = {
        "config": {"model": "one", "model_revision": "a", "output_dir": "x", "seed": 3},
        "sources": [{"text": "original"}],
        "source_config": {"key": 123},
        "source_hashes": {"core.py": "abc"},
        "environment": {"python": "3.12"},
    }
    second = deepcopy(first)
    second["config"].update(model="two", model_revision="b", output_dir="y")
    assert_comparable(first, second)
    second["config"]["seed"] = 4
    with pytest.raises(ValueError, match="seeds"):
        assert_comparable(first, second)
    second = deepcopy(first)
    second["sources"][0]["text"] = "different"
    with pytest.raises(ValueError, match="Source records"):
        assert_comparable(first, second)
