"""The model card's guardrails.

A model card is a document whose entire purpose is to be trusted about
limitations, so the failures that matter are a subgroup quietly dropped from the
breakdown, a metric published for a group too small to support it, and a claim of
a disparity that is really two different comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sepsis.model_card import (
    AGE_LABELS,
    MIN_PER_CLASS,
    _assert_partition,
    _factors,
    _metrics,
    _ordered,
    _reading,
)


def _frame(n_patients=40, hours=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        septic = i % 3 == 0
        y = np.zeros(hours, dtype=int)
        if septic:
            y[hours // 2:] = 1
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": f"p{i:03d}",
                    "hour": np.arange(hours),
                    "y": y,
                    "risk": rng.random(hours) * (0.6 if septic else 0.3),
                    "Age": 20 + (i % 4) * 20,
                    "Gender": i % 2,
                    "Unit1": float(i % 3 == 0),
                    "Unit2": float(i % 3 == 1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# The breakdown must cover everybody
# --------------------------------------------------------------------------
def test_every_factor_partitions_the_cohort():
    frame = _frame()
    for factor, series in _factors(frame).items():
        assert not series.isna().any(), f"{factor} leaves rows ungrouped"
        assert len(series) == len(frame)


def test_an_admission_with_no_recorded_unit_still_gets_a_group():
    """The largest unit bucket in hospital A is the one with no indicator set.
    Dropping it would understate who the model was evaluated on."""
    frame = _frame()
    frame.loc[:, ["Unit1", "Unit2"]] = np.nan
    units = _factors(frame)["unit"]
    assert (units == "unit not recorded").all()


def test_partition_check_rejects_an_ungrouped_row():
    series = pd.Series(["a", None, "b"])
    with pytest.raises(ValueError, match="not a breakdown"):
        _assert_partition(series, "unit", "hospital A")


def test_age_bands_are_reported_in_age_order():
    assert _ordered(pd.Series(AGE_LABELS[::-1])) == AGE_LABELS


# --------------------------------------------------------------------------
# Small groups are shown, not scored
# --------------------------------------------------------------------------
def test_a_group_too_small_to_score_is_reported_without_metrics():
    frame = _frame(n_patients=6)
    out = _metrics(frame, threshold=0.5)

    assert out["n_admissions"] == 6
    assert np.isnan(out["auroc"]), "AUROC on a handful of patients must be withheld"


def test_a_group_large_enough_gets_scored():
    frame = _frame(n_patients=60)
    out = _metrics(frame, threshold=0.3)

    assert 0.0 <= out["auroc"] <= 1.0
    assert out["auroc_lo"] <= out["auroc"] <= out["auroc_hi"]
    assert out["n_admissions"] == 60


def test_the_suppression_rule_counts_both_classes():
    """A group of 500 controls and 3 septic admissions is not evaluable either."""
    frame = _frame(n_patients=60)
    septic_ids = frame.groupby("patient_id")["y"].max()
    keep = list(septic_ids[septic_ids == 0].index) + list(septic_ids[septic_ids == 1].index[:MIN_PER_CLASS - 1])

    out = _metrics(frame[frame["patient_id"].isin(keep)], threshold=0.3)
    assert np.isnan(out["auroc"])


# --------------------------------------------------------------------------
# The written reading must match the numbers
# --------------------------------------------------------------------------
def _table(a_low, a_high, b_low, b_high):
    rows = []
    for split, low, high in (("hospital A (test)", a_low, a_high),
                             ("hospital B (external)", b_low, b_high)):
        for group, auroc in (("male (coded 1)", low), ("female (coded 0)", high)):
            rows.append({
                "split": split, "factor": "sex", "group": group, "auroc": auroc,
                "auroc_lo": auroc - 0.01, "auroc_hi": auroc + 0.01,
                "false_alarm_rate": 30.0,
            })
    return pd.DataFrame(rows)


def test_a_gap_that_does_not_replicate_is_said_not_to():
    text = _reading(_table(0.79, 0.85, 0.80, 0.801))
    assert "does not reappear" in text


def test_a_gap_that_replicates_is_said_to():
    text = _reading(_table(0.79, 0.85, 0.78, 0.85))
    assert "reappears" in text and "does not reappear" not in text


def test_overlapping_intervals_are_reported_as_overlapping():
    assert "intervals on hospital A overlap" in _reading(_table(0.800, 0.805, 0.80, 0.81))
    assert "do not overlap" in _reading(_table(0.70, 0.90, 0.70, 0.90))


# --------------------------------------------------------------------------
# The shipped card
# --------------------------------------------------------------------------
def test_the_card_carries_the_sections_that_make_it_a_model_card():
    from sepsis.config import CFG

    path = CFG.root / "MODEL_CARD.md"
    if not path.exists():
        pytest.skip("card not generated in this checkout")

    text = path.read_text()
    flat = " ".join(text.split()).lower()
    for heading in ("## Model details", "## Intended use", "## Factors", "## Metrics",
                    "## Quantitative analyses", "## Ethical considerations",
                    "## Caveats and recommendations"):
        assert heading in text, f"missing {heading}"

    # The limitations that are easiest to leave out are the ones worth asserting.
    for claim in ("not prospectively validated", "no race or ethnicity",
                  "clinical suspicion", "not a medical device"):
        assert claim in flat, f"card does not state: {claim}"
