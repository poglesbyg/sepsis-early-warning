"""Central configuration.

Every path and hyperparameter the pipeline touches is resolved here so that a run
is reproducible from a single object. Paths are relative to the repository root,
which is inferred from this file's location rather than the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The 34 clinical time-series channels, grouped as the challenge documents them.
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]

# Static per-admission covariates. Unit1/Unit2 are one-hot MICU/SICU indicators.
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

CHANNELS = VITALS + LABS
RAW_COLUMNS = CHANNELS + DEMOGRAPHICS + ["SepsisLabel"]

# Labs that are drawn rarely enough that "was it ordered at all?" carries signal.
# See stats/univariate.py -- ordering behaviour is often more predictive than value.
SPARSE_LABS = ["Lactate", "WBC", "Creatinine", "BUN", "Platelets", "PTT", "Fibrinogen", "TroponinI"]


@dataclass(frozen=True)
class UtilityParams:
    """Parameters of the official PhysioNet 2019 clinical utility function.

    Hours are relative to t_sepsis (the labelled onset of clinical suspicion).
    A positive prediction first earns credit at ``dt_early``, is worth the most
    at ``dt_optimal``, and stops earning anything after ``dt_late``.
    """

    dt_early: int = -12
    dt_optimal: int = -6
    dt_late: float = 3.0
    max_u_tp: float = 1.0
    min_u_fn: float = -2.0
    u_fp: float = -0.05
    u_tn: float = 0.0


@dataclass(frozen=True)
class WindowParams:
    """Rolling-window sizes (hours) used by the temporal feature block."""

    short: int = 6
    medium: int = 12
    long: int = 24


@dataclass
class Config:
    # --- paths -------------------------------------------------------------
    root: Path = REPO_ROOT
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    # --- data --------------------------------------------------------------
    s3_bucket_url: str = "https://physionet-open.s3.amazonaws.com"
    s3_prefix: str = "challenge-2019/1.0.0/training"
    download_workers: int = 32

    # --- modelling ---------------------------------------------------------
    seed: int = 20190801
    # Fraction of *patients* (never rows) held out from hospital A.
    val_size: float = 0.15
    test_size: float = 0.15
    cv_folds: int = 5

    # Optuna budget for the XGBoost search.
    n_trials: int = 60
    optuna_timeout: int | None = 1800

    utility: UtilityParams = field(default_factory=UtilityParams)
    windows: WindowParams = field(default_factory=WindowParams)

    def __post_init__(self) -> None:
        self.raw_dir = self.root / "data" / "raw"
        self.interim_dir = self.root / "data" / "interim"
        self.processed_dir = self.root / "data" / "processed"
        self.artifacts_dir = self.root / "artifacts"
        self.reports_dir = self.root / "reports"
        self.figures_dir = self.reports_dir / "figures"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir, self.interim_dir, self.processed_dir,
            self.artifacts_dir, self.reports_dir, self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
