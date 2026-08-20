"""Domain features: bedside scores a clinician would actually compute.

These exist to give the linear model a fair fight. A gradient-booster can in
principle discover "HR over SBP" on its own, but only if the interaction happens
to be reachable through axis-aligned splits; logistic regression cannot discover
it at all. Encoding the ratios and criterion counts that intensivists already use
-- SIRS, qSOFA, shock index, SaO2/FiO2 -- puts the physiology in the design
matrix instead of hoping the model reinvents it.

Every function here reads only the current hour's carried-forward values, so
nothing looks into the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def sirs_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Systemic Inflammatory Response Syndrome: four binary criteria and their sum.

    Sepsis-1/2 defined sepsis as infection plus >=2 SIRS criteria. Sepsis-3
    retired SIRS as a definition, but the individual criteria remain informative
    physiological thresholds, which is all we are using them as here.
    """
    temp, hr, resp = df["Temp"], df["HR"], df["Resp"]
    paco2, wbc = df["PaCO2"], df["WBC"]

    out = pd.DataFrame(index=df.index)
    out["sirs_temp"] = ((temp > 38.0) | (temp < 36.0)).astype("float32")
    out["sirs_hr"] = (hr > 90).astype("float32")
    out["sirs_resp"] = ((resp > 20) | (paco2 < 32)).astype("float32")
    out["sirs_wbc"] = ((wbc > 12) | (wbc < 4)).astype("float32")
    out["sirs_score"] = out[["sirs_temp", "sirs_hr", "sirs_resp", "sirs_wbc"]].sum(axis=1)
    # Mark the criteria that could not be assessed -- an unassessed criterion is
    # not the same as a negative one, and the count above silently conflates them.
    out["sirs_unmeasured"] = (
        temp.isna().astype("float32")
        + hr.isna().astype("float32")
        + (resp.isna() & paco2.isna()).astype("float32")
        + wbc.isna().astype("float32")
    )
    return out


def qsofa(df: pd.DataFrame) -> pd.DataFrame:
    """Quick SOFA. The GCS component is absent from this dataset, so this is the
    two-component variant (respiratory rate >= 22, systolic BP <= 100)."""
    out = pd.DataFrame(index=df.index)
    out["qsofa_resp"] = (df["Resp"] >= 22).astype("float32")
    out["qsofa_sbp"] = (df["SBP"] <= 100).astype("float32")
    out["qsofa_score"] = out["qsofa_resp"] + out["qsofa_sbp"]
    return out


def organ_dysfunction(df: pd.DataFrame) -> pd.DataFrame:
    """SOFA-flavoured organ markers available in this feature set.

    A full SOFA needs GCS, vasopressor dose and urine output, none of which are
    recorded here, so this is a partial surrogate over the four systems we can see.
    """
    out = pd.DataFrame(index=df.index)
    out["sofa_resp"] = _safe_div(df["SaO2"], df["FiO2"]).astype("float32")
    out["sofa_coag"] = pd.cut(
        df["Platelets"], [-np.inf, 20, 50, 100, 150, np.inf], labels=[4, 3, 2, 1, 0]
    ).astype("float32")
    out["sofa_liver"] = pd.cut(
        df["Bilirubin_total"], [-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype("float32")
    out["sofa_cardio"] = pd.cut(
        df["MAP"], [-np.inf, 70, np.inf], labels=[1, 0]
    ).astype("float32")
    out["sofa_renal"] = pd.cut(
        df["Creatinine"], [-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype("float32")
    out["sofa_partial"] = out[
        ["sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal"]
    ].sum(axis=1, min_count=1)
    return out


def hemodynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Derived circulatory quantities. Ratios and differences, not raw levels."""
    out = pd.DataFrame(index=df.index)
    out["shock_index"] = _safe_div(df["HR"], df["SBP"]).astype("float32")
    out["modified_shock_index"] = _safe_div(df["HR"], df["MAP"]).astype("float32")
    out["pulse_pressure"] = (df["SBP"] - df["DBP"]).astype("float32")
    out["age_shock_index"] = (out["shock_index"] * df["Age"]).astype("float32")
    out["rate_pressure_product"] = (df["HR"] * df["SBP"] / 1000).astype("float32")
    out["hypotensive"] = (df["MAP"] < 65).astype("float32")
    out["tachycardic"] = (df["HR"] > 100).astype("float32")
    out["hypoxic"] = (df["O2Sat"] < 92).astype("float32")
    return out


def metabolic(df: pd.DataFrame) -> pd.DataFrame:
    """Perfusion and end-organ chemistry.

    Lactate is the single most-cited early marker of tissue hypoperfusion; the
    BUN/creatinine ratio separates pre-renal (volume-depleted) from intrinsic
    renal dysfunction; the anion gap picks up metabolic acidosis.
    """
    out = pd.DataFrame(index=df.index)
    out["bun_creatinine"] = _safe_div(df["BUN"], df["Creatinine"]).astype("float32")
    # Sodium is not recorded in this dataset, so the gap is computed against a
    # nominal 140 mmol/L. That makes the level arbitrary but the *variation*
    # across patients -- which is what the model uses -- still meaningful.
    out["anion_gap"] = (140 - df["Chloride"] - df["HCO3"]).astype("float32")
    out["lactate_high"] = (df["Lactate"] > 2.0).astype("float32")
    out["lactate_shock"] = (df["Lactate"] > 4.0).astype("float32")
    out["acidotic"] = (df["pH"] < 7.35).astype("float32")
    out["hgb_hct_ratio"] = _safe_div(df["Hgb"], df["Hct"]).astype("float32")
    out["temp_hr_coupling"] = (df["HR"] - 10 * (df["Temp"] - 37)).astype("float32")
    return out


def administrative(df: pd.DataFrame) -> pd.DataFrame:
    """Non-physiological context that nonetheless carries risk information."""
    out = pd.DataFrame(index=df.index)
    out["iculos"] = df["ICULOS"].astype("float32")
    out["log_iculos"] = np.log1p(df["ICULOS"]).astype("float32")
    out["hosp_adm_time"] = df["HospAdmTime"].astype("float32")
    # Negative HospAdmTime means the ward stay preceded ICU transfer; a long
    # pre-ICU ward stay is a different clinical trajectory from a direct admit.
    out["ward_before_icu"] = (df["HospAdmTime"] < -6).astype("float32")
    out["age"] = df["Age"].astype("float32")
    out["age_over_65"] = (df["Age"] > 65).astype("float32")
    out["gender"] = df["Gender"].astype("float32")
    out["unit_micu"] = df["Unit1"].fillna(0).astype("float32")
    out["unit_sicu"] = df["Unit2"].fillna(0).astype("float32")
    out["unit_unknown"] = df["Unit1"].isna().astype("float32")
    return out


def build_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """All domain blocks, concatenated."""
    return pd.concat(
        [
            sirs_criteria(df),
            qsofa(df),
            organ_dysfunction(df),
            hemodynamics(df),
            metabolic(df),
            administrative(df),
        ],
        axis=1,
    )
