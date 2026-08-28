"""Command-line entry point.

    sepsis data                 download and cache the PhysioNet training sets
    sepsis features             build the causal feature tables
    sepsis explore              univariate screen, collinearity, cross-site drift
    sepsis train                fit baseline, logistic, XGBoost and GRU
    sepsis evaluate             calibrate, blend, score, write REPORT.md
    sepsis replay               pick and pre-compute the admissions the report replays
    sepsis experiments          run the registered ablations and shift experiments
    sepsis regress              check published numbers against the committed baseline
    sepsis all                  everything, in order
"""

from __future__ import annotations

import argparse
import sys

from .config import CFG
from .data import integrity
from .data.download import ensure_data


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--trials", type=int, default=CFG.n_trials, help="Optuna trial budget")
    p.add_argument("--timeout", type=int, default=CFG.optuna_timeout, help="Optuna wall-clock cap (s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sepsis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="download and cache the raw data")
    p_data.add_argument("--hospitals", nargs="+", default=["A", "B"])
    p_data.add_argument("--write-checksums", action="store_true",
                        help="pin the downloaded files in configs/data_checksums.json")

    p_feat = sub.add_parser("features", help="build feature tables")
    p_feat.add_argument("--force", action="store_true", help="rebuild even if cached")

    sub.add_parser("explore", help="statistical analysis of the training split")

    p_train = sub.add_parser("train", help="fit the models")
    p_train.add_argument("--skip", nargs="*", default=[],
                         choices=["clinical_rule", "logistic", "xgboost", "gru"])
    _add_common(p_train)

    sub.add_parser("evaluate", help="calibrate, blend and report")

    sub.add_parser("replay", help="pre-compute the report's admission replay")

    p_exp = sub.add_parser("experiments", help="run the registered experiments")
    p_exp.add_argument("--only", nargs="*", default=[], help="run only these experiments")

    p_regress = sub.add_parser("regress", help="check published numbers against the baseline")
    p_regress.add_argument("--update", action="store_true",
                           help="rewrite the baseline from the current artifacts")
    p_regress.add_argument("--tolerance", type=float, default=None)

    p_all = sub.add_parser("all", help="run every stage")
    p_all.add_argument("--skip", nargs="*", default=[],
                       choices=["clinical_rule", "logistic", "xgboost", "gru"])
    _add_common(p_all)

    args = parser.parse_args(argv)
    cfg = CFG
    for attr, value in (("seed", getattr(args, "seed", None)),
                        ("n_trials", getattr(args, "trials", None)),
                        ("optuna_timeout", getattr(args, "timeout", None))):
        if value is not None:
            setattr(cfg, attr, value)

    from . import pipeline
    from .report import write_report

    if args.command == "data":
        hospitals = tuple(args.hospitals)
        ensure_data(hospitals, cfg)
        if args.write_checksums:
            integrity.write_manifest(hospitals, cfg)
        else:
            integrity.verify(hospitals, cfg)
    elif args.command == "features":
        pipeline.stage_features(cfg, force=args.force)
    elif args.command == "explore":
        pipeline.stage_explore(pipeline.stage_features(cfg), cfg)
    elif args.command == "train":
        frames = pipeline.stage_features(cfg)
        pipeline.stage_train(frames, pipeline.stage_explore(frames, cfg), cfg, skip=tuple(args.skip))
    elif args.command == "evaluate":
        pipeline.stage_evaluate(cfg)
        write_report(cfg)
    elif args.command == "replay":
        pipeline.stage_replay(cfg)
        write_report(cfg)
    elif args.command == "experiments":
        pipeline.stage_experiments(cfg, only=tuple(args.only))
        write_report(cfg)
    elif args.command == "regress":
        from . import regress

        if args.update:
            regress.write_baseline(cfg)
            return 0
        try:
            regress.check(cfg, **({"tolerance": args.tolerance} if args.tolerance else {}))
        except (ValueError, FileNotFoundError) as exc:
            # A moved number is an expected outcome of this command, not a crash.
            # The message is the report; a traceback would bury it.
            print(f"\n{exc}", file=sys.stderr)
            return 1
    elif args.command == "all":
        pipeline.run_all(cfg, skip=tuple(args.skip))
        write_report(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
