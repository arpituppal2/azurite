from __future__ import annotations

import random

from azurite.abstain import ConformalCalibrator, conformal_threshold


def test_threshold_answers_stated_coverage_on_heldout():
    random.seed(0)
    cal, held = [], []
    for bucket in (cal, held):
        for _ in range(4000):
            ok = random.random() < 0.95
            score = random.betavariate(6, 2) if ok else random.betavariate(2, 6)
            bucket.append((score, ok))
    calib = ConformalCalibrator.fit([s for s, _ in cal], [o for _, o in cal], coverage=0.90)
    answered = [ok for s, ok in held if not calib.abstain(s)]
    coverage = len(answered) / len(held)
    assert 0.87 <= coverage <= 0.93, coverage
    assert sum(answered) / len(answered) > 0.90


def test_threshold_is_a_score_quantile_not_an_error_guard():
    # A 50%-accurate model must still answer ~90% of rows; the old
    # error-as-nonconformity rule forced the threshold to 1.0 here.
    threshold = conformal_threshold([0.9] * 100, [True] * 50 + [False] * 50, coverage=0.90)
    assert threshold == 0.9


def test_threshold_rejects_bad_arguments():
    for scores, coverage in (([], 0.9), ([0.5], 0.0), ([0.5], 1.0)):
        try:
            conformal_threshold(scores, coverage=coverage)
        except ValueError:
            continue
        raise AssertionError((scores, coverage))
