"""训练入口：python -m scripts.train --model logreg [--fit-full]。"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import FEATURES_DIR
from src.train import SHANGHAI, TrainingConfig, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["logreg", "catboost"], default="logreg")
    parser.add_argument("--strategy", choices=["simple", "indicator", "rule_group"], default="rule_group")
    parser.add_argument("--run-name", default=None, help="实验名称；不覆盖已有同名目录")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--folds", type=Path, default=FEATURES_DIR / "folds.csv")
    parser.add_argument("--fit-full", action="store_true", help="五折后额外全量训练、保存模型及提交文件")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="LogReg 的 C")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    config = TrainingConfig(model=args.model, strategy=args.strategy, seed=args.seed, c=args.c,
                            iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
                            l2_leaf_reg=args.l2_leaf_reg, threads=args.threads)
    run_name = args.run_name or f"{args.model}_{datetime.now(SHANGHAI):%Y%m%d_%H%M%S}"
    run_training(config, run_name, data_dir=args.data_dir, folds_path=args.folds, fit_full=args.fit_full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
