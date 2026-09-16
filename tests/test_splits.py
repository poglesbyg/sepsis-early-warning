"""Split hygiene: no admission may appear in more than one split."""

from __future__ import annotations

import numpy as np
import nopeek
import pandas as pd
import pytest

from sepsis.data.loader import Split, patient_labels


def _toy_frame(n_patients=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = int(rng.integers(5, 30))
        septic = rng.random() < 0.3
        y = np.zeros(n, dtype=np.int8)
        if septic:
            y[int(rng.integers(0, n)) :] = 1
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": f"p{i:05d}",
                    "hour": np.arange(n),
                    "HR": rng.normal(85, 12, n),
                    "SepsisLabel": y,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_patient_labels_is_max_over_stay():
    df = _toy_frame()
    labels = patient_labels(df)
    for pid, expected in df.groupby("patient_id")["SepsisLabel"].max().items():
        assert labels[pid] == expected


def test_splits_share_no_patients():
    df = _toy_frame()
    ids = df["patient_id"].unique()
    parts = np.array_split(ids, 3)
    splits = {
        name: Split(name, df[df["patient_id"].isin(set(p))].reset_index(drop=True))
        for name, p in zip(("train", "val", "test"), parts)
    }
    seen: set[str] = set()
    for s in splits.values():
        pats = set(s.patients)
        assert not (pats & seen), "an admission leaked across splits"
        seen |= pats
    assert seen == set(ids)


def test_splits_are_disjoint_in_every_way_that_matters():
    """Beyond shared admissions: byte-identical rows on both sides leak too.

    A duplicated export or a merge that fanned out puts the same hour in train
    and test without either split sharing a patient id, which the check above
    would not see.
    """
    df = _toy_frame()
    ids = df["patient_id"].unique()
    train_ids, test_ids = ids[: len(ids) // 2], ids[len(ids) // 2 :]
    train = df[df["patient_id"].isin(set(train_ids))]
    test = df[df["patient_id"].isin(set(test_ids))]

    nopeek.assert_split_clean(train, test, group="patient_id")


def test_a_row_level_split_is_rejected():
    """The check has to be able to fail, or it is decoration."""
    df = _toy_frame()
    report = nopeek.verify_split(df.iloc[::2], df.iloc[1::2], group="patient_id")
    assert not report.ok
    assert any(leak.kind == "group_overlap" for leak in report.leaks)


def test_describe_counts_admissions_not_rows():
    df = _toy_frame()
    d = Split("all", df).describe()
    assert d["hours"] == len(df)
    assert d["admissions"] == df["patient_id"].nunique()
    assert d["admissions"] < d["hours"]
