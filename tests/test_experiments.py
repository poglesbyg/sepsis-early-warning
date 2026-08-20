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
