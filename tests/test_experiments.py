"""The experiment layer's own guardrails.

These experiments exist to publish numbers, so the failure that matters is not a
crash. It is a plausible table containing a wrong number, produced because the
mistake under study was never actually committed. Every assertion below exists to
make that outcome loud.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from sepsis.experiments.common import (
    ExperimentResult,
    admission_split,
    assert_partition,
    block_columns,
    row_split,
)
from sepsis.experiments.leakage import NOISE_FLOOR, _assert_direction, bidirectional_fill
from sepsis.features import build_features
from sepsis.features.builder import feature_columns
from tests.test_features import _toy_stays


# --------------------------------------------------------------------------
# Feature blocks must tile the matrix exactly
# --------------------------------------------------------------------------
def test_feature_blocks_partition_the_matrix_exactly():
    """No overlap, no orphans. Otherwise every ablation number is meaningless."""
    sample = _toy_stays(n_patients=4, n_hours=30)
    membership = block_columns(sample)
    all_features = feature_columns(build_features(sample))

    assert_partition(membership, all_features)
    assert sum(len(v) for v in membership.values()) == len(all_features)


def test_partition_check_catches_an_overlap():
    with pytest.raises(ValueError, match="overlap"):
        assert_partition({"a": ["x", "y"], "b": ["y", "z"]}, ["x", "y", "z"])


def test_partition_check_catches_an_orphan_feature():
    with pytest.raises(ValueError, match="do not partition"):
        assert_partition({"a": ["x"], "b": ["y"]}, ["x", "y", "z"])


# --------------------------------------------------------------------------
# The splits must actually differ in the way the experiment claims
# --------------------------------------------------------------------------
def test_admission_split_keeps_stays_whole_and_row_split_does_not():
    frame = build_features(_toy_stays(n_patients=40, n_hours=25))

    tr, te = admission_split(frame, test_size=0.25, seed=0)
    assert not (set(tr["patient_id"]) & set(te["patient_id"]))

    tr_r, te_r = row_split(frame, test_size=0.25, seed=0)
    overlap = set(tr_r["patient_id"]) & set(te_r["patient_id"])
    assert len(overlap) > 30, "row split should scatter nearly every admission across both sides"


def test_row_split_preserves_every_row_exactly_once():
    frame = build_features(_toy_stays(n_patients=12, n_hours=20))
    tr, te = row_split(frame, test_size=0.3, seed=1)
    assert len(tr) + len(te) == len(frame)
    assert len(te) == pytest.approx(len(frame) * 0.3, rel=0.02)


# --------------------------------------------------------------------------
# The lookahead mistake must actually look ahead
# --------------------------------------------------------------------------
def test_bidirectional_fill_pulls_values_backwards_from_the_future():
    """A gap before the first measurement is filled from a later hour."""
    raw = pd.DataFrame(
        {
            "patient_id": ["p0"] * 6,
            "hour": range(6),
            "HR": [np.nan, np.nan, 80.0, np.nan, np.nan, 95.0],
        }
    )
    for col in ("O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"):
        raw[col] = np.nan

    from sepsis.config import CHANNELS

    for ch in CHANNELS:
        if ch not in raw:
            raw[ch] = np.nan

    filled = bidirectional_fill(raw)
    # Hours 0 and 1 precede any measurement, so a causal fill must leave them NaN.
    assert np.isnan(raw["HR"].iloc[0])
    assert filled["HR"].iloc[0] == 80.0, "backward fill should reach back from hour 2"
    assert filled["HR"].iloc[1] == 80.0
    # Forward fill still applies where it legitimately can.
    assert filled["HR"].iloc[3] == 80.0


def test_bidirectional_fill_does_not_cross_admissions():
    from sepsis.config import CHANNELS

    raw = pd.DataFrame({"patient_id": ["p0", "p0", "p1", "p1"], "hour": [0, 1, 0, 1]})
    for ch in CHANNELS:
        raw[ch] = np.nan
    raw.loc[1, "HR"] = 70.0  # only p0 has a reading

    filled = bidirectional_fill(raw)
    assert filled["HR"].iloc[0] == 70.0
    assert np.isnan(filled["HR"].iloc[2]), "p1 must not inherit p0's value"
    assert np.isnan(filled["HR"].iloc[3])


# --------------------------------------------------------------------------
# Direction assertion: leakage cannot hurt
# --------------------------------------------------------------------------
def _table(inflations):
    rows = [{"variant": "honest", "auroc_inflation": 0.0}]
    rows += [{"variant": f"v{i}", "auroc_inflation": v} for i, v in enumerate(inflations)]
    return pd.DataFrame(rows)


def test_direction_assertion_accepts_a_real_leak():
    _assert_direction(_table([0.03, 0.01]), shared_admissions=5000)


def test_direction_assertion_rejects_a_leak_that_hurt():
    """A leaky variant scoring below honest means the leak was never committed."""
    with pytest.raises(ValueError, match="BELOW the honest"):
        _assert_direction(_table([-0.05]), shared_admissions=5000)


def test_direction_assertion_rejects_movement_inside_the_noise_floor():
    with pytest.raises(ValueError, match="noise floor"):
        _assert_direction(_table([NOISE_FLOOR / 2]), shared_admissions=5000)


def test_direction_assertion_rejects_a_split_with_no_overlap():
    with pytest.raises(ValueError, match="no admission on both sides"):
        _assert_direction(_table([0.03]), shared_admissions=0)


# --------------------------------------------------------------------------
# Results must be publishable
# --------------------------------------------------------------------------
def test_result_rejects_nan_in_a_published_column():
    bad = ExperimentResult(
        name="x", title="t",
        table=pd.DataFrame({"variant": ["a"], "auroc": [np.nan]}), prose="",
    )
    with pytest.raises(ValueError, match="NaN in published columns"):
        bad.validate()


def test_result_rejects_an_empty_table():
    with pytest.raises(ValueError, match="empty table"):
        ExperimentResult(name="x", title="t", table=pd.DataFrame(), prose="").validate()


def test_result_accepts_a_clean_table():
    ok = ExperimentResult(
        name="x", title="t",
        table=pd.DataFrame({"variant": ["a"], "auroc": [0.8]}), prose="fine",
    )
    assert ok.validate() is ok


def test_registry_is_explicit_and_callable():
    from sepsis.experiments import REGISTRY

    assert "leakage" in REGISTRY
    assert all(callable(fn) for fn in REGISTRY.values())


# --------------------------------------------------------------------------
# Ablation guards
# --------------------------------------------------------------------------
def test_ordering_only_blocks_contain_no_measured_values():
    """The headline ablation claim depends on this subset holding zero values.

    If a channel value leaked into the ordering-only set, the "97% from no
    measurements" result would be false and nothing else would catch it.
    """
    from sepsis.experiments.ablation import ORDERING_ONLY_BLOCKS

    sample = _toy_stays(n_patients=4, n_hours=30)
    membership = block_columns(sample)
    ordering = [c for b in ORDERING_ONLY_BLOCKS for c in membership[b]]

    assert ordering, "ordering-only subset must not be empty"
    for col in ordering:
        assert not col.endswith("_locf"), f"{col} is a measured value"
        assert not col.endswith("_dev"), f"{col} is derived from measured values"
        assert not re.search(r"_(mean|min|max|std|slope)\d+$", col), f"{col} summarises values"


def test_ablation_rejects_block_sizes_that_do_not_sum_to_the_matrix():
    from sepsis.experiments.ablation import _assert_blocks_are_real
    from sepsis.features.builder import FEATURE_BLOCKS

    table = pd.DataFrame({
        "block": list(FEATURE_BLOCKS),
        "n_features": [1] * len(FEATURE_BLOCKS),
        "auroc_solo": [0.7] * len(FEATURE_BLOCKS),
    })
    membership = {b: ["x"] for b in FEATURE_BLOCKS}
    with pytest.raises(ValueError, match="not measuring a partition"):
        _assert_blocks_are_real(table, n_features=345, membership=membership)


def test_ablation_rejects_a_block_scoring_below_chance():
    from sepsis.experiments.ablation import _assert_blocks_are_real
    from sepsis.features.builder import FEATURE_BLOCKS

    n = len(FEATURE_BLOCKS)
    table = pd.DataFrame({
        "block": list(FEATURE_BLOCKS),
        "n_features": [1] * n,
        "auroc_solo": [0.7] * (n - 1) + [0.42],
    })
    membership = {b: ["x"] for b in FEATURE_BLOCKS}
    with pytest.raises(ValueError, match="below chance"):
        _assert_blocks_are_real(table, n_features=n, membership=membership)


# --------------------------------------------------------------------------
# Weighted scoring: reweighting a cohort must mean what it says
# --------------------------------------------------------------------------
def _toy_cohort(n_patients=8, n_hours=30):
    """Labels and admission ids in the shape ``UtilityScorer`` expects."""
    frame = _toy_stays(n_patients=n_patients, n_hours=n_hours)
    rng = np.random.default_rng(3)
    return (
        frame["SepsisLabel"].to_numpy(),
        frame["patient_id"].to_numpy(),
        rng.random(len(frame)),
    )


def test_weighted_utility_reduces_to_the_plain_score_at_unit_weights():
    from sepsis.evaluate.metrics import UtilityScorer
    from sepsis.experiments.common import weighted_utility

    y, groups, scores = _toy_cohort()
    scorer = UtilityScorer(y, groups)
    alerts = (scores >= 0.5).astype(float)

    assert weighted_utility(scorer, alerts, np.ones(len(y))) == pytest.approx(
        scorer.score(alerts)
    )


def test_a_weight_of_two_equals_the_admission_appearing_twice():
    """The property the whole reweighting rests on.

    If doubling an admission's weight is not the same as that admission being in
    the cohort twice, the reweighted hospital B number is not the utility of any
    population and the decomposition means nothing.
    """
    from sepsis.evaluate.metrics import UtilityScorer
    from sepsis.experiments.common import weighted_utility

    frame = _toy_stays(n_patients=6, n_hours=25)
    rng = np.random.default_rng(0)
    frame["score"] = rng.random(len(frame))
    chosen = frame["patient_id"].unique()[0]

    duplicate = frame[frame["patient_id"] == chosen].copy()
    duplicate["patient_id"] = chosen + "_copy"
    doubled = pd.concat([frame, duplicate], ignore_index=True).sort_values(
        ["patient_id", "hour"], ignore_index=True
    )

    plain = UtilityScorer(frame["SepsisLabel"].to_numpy(), frame["patient_id"].to_numpy())
    twice = UtilityScorer(doubled["SepsisLabel"].to_numpy(), doubled["patient_id"].to_numpy())

    weights = np.where(frame["patient_id"].to_numpy() == chosen, 2.0, 1.0)
    weighted = weighted_utility(
        plain, (frame["score"] >= 0.5).astype(float).to_numpy(), weights
    )
    materialised = twice.score((doubled["score"] >= 0.5).astype(float).to_numpy())

    assert weighted == pytest.approx(materialised)


def test_admission_utility_parts_reconstruct_the_cohort_score():
    from sepsis.evaluate.metrics import UtilityScorer
    from sepsis.experiments.common import admission_utility_parts

    y, groups, scores = _toy_cohort()
    scorer = UtilityScorer(y, groups)
    alerts = (scores >= 0.4).astype(float)

    parts = admission_utility_parts(scorer, alerts, groups)
    assert parts["num"].sum() / parts["den"].sum() == pytest.approx(scorer.score(alerts))


def test_contiguity_check_rejects_an_interleaved_frame():
    """``UtilityScorer`` reads each admission as one run of rows; an unsorted frame
    invents extra admissions and extra onsets, silently."""
    from sepsis.experiments.common import assert_contiguous_admissions

    assert_contiguous_admissions(np.array(["a", "a", "b", "b", "c"]))
    with pytest.raises(ValueError, match="non-adjacent"):
        assert_contiguous_admissions(np.array(["a", "b", "a"]))


def test_baseline_covariates_never_read_past_the_window():
    from sepsis.experiments.common import baseline_covariates

    frame = pd.DataFrame(
        {
            "patient_id": ["p0"] * 10 + ["p1"] * 3,
            "hour": list(range(10)) + [0, 1, 2],
            "age": list(range(10)) + [100, 101, 102],
        }
    )
    out = baseline_covariates(frame, ["age"], window=6)
    assert out.loc["p0", "age"] == 5, "must take the last hour inside the window"
    assert out.loc["p1", "age"] == 102, "a short stay contributes its final hour"


def test_baseline_covariates_handle_a_stay_that_does_not_start_at_hour_zero():
    from sepsis.experiments.common import baseline_covariates

    frame = pd.DataFrame(
        {"patient_id": ["p0"] * 8, "hour": range(20, 28), "age": range(8)}
    )
    out = baseline_covariates(frame, ["age"], window=6)
    assert out.loc["p0", "age"] == 5, "the window is relative to the stay, not to hour 0"


# --------------------------------------------------------------------------
# Prevalence-shift decomposition guards
# --------------------------------------------------------------------------
def test_decomposition_must_close():
    from sepsis.experiments.prevalence import _assert_decomposition

    _assert_decomposition(gap=0.13, case_mix=0.02, degradation=0.11)
    with pytest.raises(ValueError, match="does not close"):
        _assert_decomposition(gap=0.13, case_mix=0.02, degradation=-0.11)


def _overlap_diagnostics(**overrides):
    base = {
        "propensity_auc": 0.78,
        "ess_fraction": 0.53,
        "smd_before": 0.22,
        "smd_after": 0.07,
    }
    return {**base, **overrides}


def test_overlap_guard_accepts_a_supported_reweighting():
    from sepsis.experiments.prevalence import _assert_overlap

    _assert_overlap(_overlap_diagnostics())


def test_overlap_guard_rejects_two_cohorts_that_do_not_overlap():
    from sepsis.experiments.prevalence import _assert_overlap

    with pytest.raises(ValueError, match="no covariate overlap"):
        _assert_overlap(_overlap_diagnostics(propensity_auc=0.997))


def test_overlap_guard_rejects_a_collapsed_effective_sample():
    from sepsis.experiments.prevalence import _assert_overlap

    with pytest.raises(ValueError, match="effective"):
        _assert_overlap(_overlap_diagnostics(ess_fraction=0.01))


def test_overlap_guard_rejects_a_reweighting_that_worsened_balance():
    """Weights that do not move the covariates toward the reference are not a
    case-mix adjustment, whatever the resulting number looks like."""
    from sepsis.experiments.prevalence import _assert_overlap

    with pytest.raises(ValueError, match="did not improve covariate balance"):
        _assert_overlap(_overlap_diagnostics(smd_after=0.30))


def test_smd_is_zero_between_a_cohort_and_itself():
    from sepsis.experiments.prevalence import _smd

    rng = np.random.default_rng(0)
    cohort = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    assert np.abs(_smd(cohort, cohort, np.ones(len(cohort)))).max() < 1e-9


def test_smd_detects_a_shifted_covariate():
    from sepsis.experiments.prevalence import _smd

    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    target = pd.DataFrame({"a": rng.normal(1, 1, 500)})
    assert _smd(reference, target, np.ones(len(target)))[0] > 0.5


# --------------------------------------------------------------------------
# Unit-transfer guards
# --------------------------------------------------------------------------
def _unit_pool(micu=3, sicu=3, unknown=2, n_hours=5):
    rows = []
    for label, n in (("unit_micu", micu), ("unit_sicu", sicu), ("unit_unknown", unknown)):
        for i in range(n):
            frame = pd.DataFrame({"hour": range(n_hours)})
            frame["patient_id"] = f"{label}_{i}"
            for col in ("unit_micu", "unit_sicu", "unit_unknown"):
                frame[col] = float(col == label)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_unit_buckets_must_partition_the_pool():
    from sepsis.experiments.unit_transfer import _assert_buckets, _bucket_admissions

    pool = _unit_pool()
    _assert_buckets(_bucket_admissions(pool), pool)


def test_unit_buckets_reject_an_admission_in_two_units():
    from sepsis.experiments.unit_transfer import _assert_buckets, _bucket_admissions

    pool = _unit_pool()
    pool.loc[pool["patient_id"] == "unit_micu_0", "unit_sicu"] = 1.0
    with pytest.raises(ValueError, match="more than one unit bucket"):
        _assert_buckets(_bucket_admissions(pool), pool)


def test_unit_buckets_reject_an_admission_in_no_unit():
    from sepsis.experiments.unit_transfer import _assert_buckets, _bucket_admissions

    pool = _unit_pool()
    pool.loc[pool["patient_id"] == "unit_unknown_0", "unit_unknown"] = 0.0
    with pytest.raises(ValueError, match="silently dropped"):
        _assert_buckets(_bucket_admissions(pool), pool)


def test_unit_transfer_rejects_a_training_admission_in_an_evaluation_bucket():
    from sepsis.experiments.unit_transfer import _assert_no_overlap

    _assert_no_overlap(["a", "b"], ["c"], {"SICU": np.array(["d", "e"])})
    with pytest.raises(ValueError, match="not held out"):
        _assert_no_overlap(["a", "b"], ["c"], {"SICU": np.array(["b", "e"])})


def test_unit_transfer_rejects_a_bucket_with_no_septic_admission():
    from sepsis.experiments.unit_transfer import _assert_scored

    table = pd.DataFrame(
        {"cohort": ["MICU", "SICU"], "septic_admissions_pct": [10.0, 0.0], "auroc": [0.8, 0.8]}
    )
    with pytest.raises(ValueError, match="no septic admission"):
        _assert_scored(table)


def test_unit_transfer_rejects_below_chance_transfer():
    from sepsis.experiments.unit_transfer import _assert_scored

    table = pd.DataFrame(
        {"cohort": ["MICU", "SICU"], "septic_admissions_pct": [10.0, 4.0], "auroc": [0.8, 0.41]}
    )
    with pytest.raises(ValueError, match="cohort assembly error"):
        _assert_scored(table)


def test_unit_indicators_are_excluded_from_the_transfer_feature_set():
    """Training on one unit makes these constant, and every evaluation bucket takes
    a value the model never saw. Leaving them in measures a dead column."""
    from sepsis.experiments.unit_transfer import UNIT_COLUMNS, transfer_features

    frame = build_features(_toy_stays(n_patients=4, n_hours=20))
    assert set(UNIT_COLUMNS) <= set(feature_columns(frame)), (
        "the columns must exist in the matrix to be worth excluding"
    )
    used = transfer_features(frame)
    assert not set(UNIT_COLUMNS) & set(used)
    assert len(used) == len(feature_columns(frame)) - len(UNIT_COLUMNS)


# --------------------------------------------------------------------------
# A local threshold must be held out from the set it is scored on
# --------------------------------------------------------------------------
def test_local_threshold_may_not_be_graded_on_its_own_selection_set():
    """The mistake the first version of the unit experiment published.

    Choosing a threshold to maximise utility and then reporting that utility on
    the same admissions reports the maximum of a sweep as though it were a
    measurement. Nothing else in the pipeline notices.
    """
    from sepsis.experiments.unit_transfer import _assert_local_threshold_is_held_out

    _assert_local_threshold_is_held_out(["a", "b"], ["c", "d"], "SICU")
    with pytest.raises(ValueError, match="own selection set"):
        _assert_local_threshold_is_held_out(["a", "b"], ["b", "c"], "SICU")


def test_utility_bootstrap_pairs_the_two_thresholds():
    """Both operating points are scored on the same resampled admissions.

    Resampling twice would widen the interval on their difference for no reason
    and could hide a real gain.
    """
    from sepsis.evaluate.metrics import UtilityScorer
    from sepsis.experiments.unit_transfer import _utility_on

    frame = _toy_stays(n_patients=30, n_hours=25)
    rng = np.random.default_rng(0)
    scores = rng.random(len(frame))
    eval_ids = set(frame["patient_id"].unique()[:20])

    out = _utility_on(frame, scores, eval_ids, frozen=0.9, local=0.5, seed=0, n_boot=40)
    assert out["utility_frozen_lo"] <= out["utility_frozen"] <= out["utility_frozen_hi"]
    assert out["utility_local_lo"] <= out["utility_local"] <= out["utility_local_hi"]
    # A paired interval on the gain must be narrower than the two marginals summed.
    gain_width = out["gain_hi"] - out["gain_lo"]
    marginal = (out["utility_frozen_hi"] - out["utility_frozen_lo"]) + (
        out["utility_local_hi"] - out["utility_local_lo"]
    )
    assert gain_width < marginal


# --------------------------------------------------------------------------
# The mechanism experiment's arms must differ in exactly one thing
# --------------------------------------------------------------------------
def test_mechanism_arms_partition_the_feature_matrix():
    from sepsis.experiments.mechanism import _assert_subsets

    _assert_subsets(["a", "b"], ["c"], ["a", "b", "c"])


def test_mechanism_rejects_overlapping_arms():
    with_msg = "differ in more than one thing"
    from sepsis.experiments.mechanism import _assert_subsets

    with pytest.raises(ValueError, match=with_msg):
        _assert_subsets(["a", "b"], ["b", "c"], ["a", "b", "c"])


def test_mechanism_rejects_a_feature_in_neither_arm():
    from sepsis.experiments.mechanism import _assert_subsets

    with pytest.raises(ValueError, match="belong to neither arm"):
        _assert_subsets(["a"], ["b"], ["a", "b", "c"])


def test_mechanism_rejects_an_empty_ordering_set():
    """With nothing withheld the experiment compares a model to itself."""
    from sepsis.experiments.mechanism import _assert_subsets

    with pytest.raises(ValueError, match="compare a model to itself"):
        _assert_subsets([], ["a", "b"], ["a", "b"])
