import numpy as np
import pytest

from watermarking.quality_uncertainty import aligned_losses, bootstrap_difference


def test_bootstrap_keeps_correlated_prompt_pairs_together():
    baseline = np.array([1.0, 10.0, 100.0])
    treatment = baseline + 1
    indices = np.random.default_rng(17).integers(0, 3, size=(1000, 3))
    result = bootstrap_difference(baseline, treatment, indices)
    assert result == {"mean_difference": 1.0, "ci95": [1.0, 1.0]}
    assert bootstrap_difference(baseline, baseline, indices)["ci95"] == [0.0, 0.0]


def test_quality_alignment_rejects_missing_and_duplicate_pairs():
    rows = [
        {"prompt_id": p, "variant": v, "mean_nll": n, "num_tokens": 200}
        for p, v, n in [
            ("p1", "baseline", 10.0),
            ("p0", "wm", 2.0),
            ("p0", "baseline", 1.0),
            ("p1", "wm", 11.0),
        ]
    ]
    losses = aligned_losses(rows, ["p0", "p1"], ["baseline", "wm"], 200)
    np.testing.assert_array_equal(losses["wm"] - losses["baseline"], [1.0, 1.0])
    with pytest.raises(ValueError, match="missing, duplicated"):
        aligned_losses(rows[:-1], ["p0", "p1"], ["baseline", "wm"], 200)
    with pytest.raises(ValueError, match="missing, duplicated"):
        aligned_losses(rows + [rows[0]], ["p0", "p1"], ["baseline", "wm"], 200)
