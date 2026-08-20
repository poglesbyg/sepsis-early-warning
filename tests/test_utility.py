"""The vectorised utility scorer must agree with the specification exactly."""

from __future__ import annotations

import numpy as np
import pytest

from sepsis.evaluate.metrics import UtilityScorer, _group_slices
from tests.reference_utility import naive_normalized_utility, naive_patient_utility


def _random_cohort(rng, n_patients=60):
    """Mix of septic and non-septic admissions with challenge-style labels."""
    labels, groups = [], []
    for i in range(n_patients):
        n = int(rng.integers(8, 90))
        y = np.zeros(n, dtype=np.int8)
        if rng.random() < 0.4:
            onset = int(rng.integers(0, n))
            y[onset:] = 1  # labels stay on from t_sepsis + dt_optimal to discharge
        labels.append(y)
        groups.append(np.full(n, f"p{i:05d}"))
    return labels, groups


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    per_patient_labels, per_patient_groups = _random_cohort(rng)
    per_patient_preds = [
        (rng.random(len(y)) < 0.3).astype(np.int8) for y in per_patient_labels
    ]

    flat_y = np.concatenate(per_patient_labels)
    flat_g = np.concatenate(per_patient_groups)
    flat_p = np.concatenate(per_patient_preds)

    expected = naive_normalized_utility(per_patient_labels, per_patient_preds)
    got = UtilityScorer(flat_y, flat_g).score(flat_p)
    assert got == pytest.approx(expected, abs=1e-9)


def test_raw_utility_matches_per_patient_sum():
    rng = np.random.default_rng(11)
    labels, groups = _random_cohort(rng, n_patients=25)
    preds = [(rng.random(len(y)) < 0.5).astype(np.int8) for y in labels]
    expected = sum(naive_patient_utility(y, p) for y, p in zip(labels, preds))
    scorer = UtilityScorer(np.concatenate(labels), np.concatenate(groups))
    assert scorer.raw(np.concatenate(preds)) == pytest.approx(expected, abs=1e-9)


def test_never_alerting_scores_zero_and_optimal_scores_one():
    rng = np.random.default_rng(7)
    labels, groups = _random_cohort(rng)
    y, g = np.concatenate(labels), np.concatenate(groups)
    scorer = UtilityScorer(y, g)

    assert scorer.score(np.zeros_like(y)) == pytest.approx(0.0, abs=1e-12)
    optimal = (scorer.delta > 0).astype(np.int8)
    assert scorer.score(optimal) == pytest.approx(1.0, abs=1e-12)
    # Alerting on every hour is strictly worse than the optimal policy.
    assert scorer.score(np.ones_like(y)) < 1.0


def test_sweep_finds_the_argmax_of_the_grid():
    rng = np.random.default_rng(3)
    labels, groups = _random_cohort(rng)
    y, g = np.concatenate(labels), np.concatenate(groups)
    scorer = UtilityScorer(y, g)
    scores = rng.random(len(y)) * 0.5 + 0.4 * y  # weakly informative

    thr, util = scorer.best_threshold(scores)
    assert util == pytest.approx(scorer.score((scores >= thr).astype(np.int8)), abs=1e-12)
    assert util >= scorer.sweep(scores)["utility"].max() - 1e-12


def test_group_slices_are_contiguous_runs():
    g = np.array(["a", "a", "b", "c", "c", "c"])
    assert _group_slices(g) == [(0, 2), (2, 3), (3, 6)]
    assert _group_slices(np.array([])) == []
