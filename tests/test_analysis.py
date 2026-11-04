import pytest

from watermarking.analysis import roc, wilson


def test_roc_handles_perfect_reversed_and_tied_scores():
    assert roc([2, 3], [0, 1])[2] == 1
    assert roc([0, 1], [2, 3])[2] == 0
    assert roc([1, 1], [1, 1])[2] == 0.5
    fp, tp, _ = roc([0, 2], [1, 3])
    assert fp[0] == tp[0] == 0
    assert fp[-1] == tp[-1] == 1


def test_zero_false_positives_still_have_nonzero_uncertainty():
    low, high = wilson(0, 100)
    assert low == pytest.approx(0)
    assert 0.03 < high < 0.04
    assert wilson(100, 100)[1] == pytest.approx(1)


def test_empty_roc_rejected():
    with pytest.raises(ValueError):
        roc([], [1])
