"""从保存的比赛集成模型复现预测：python -m scripts.predict_ensemble --bundle ...。"""
import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization import predict_ensemble
from src.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or PROJECT_ROOT / "outputs" / args.bundle.parent.name / "reproduced"
    if output.exists():
        raise FileExistsError("输出目录已存在，请指定新的 --output-dir")
    probability, submission = predict_ensemble(args.bundle, args.data_dir)
    predictions = pd.DataFrame({"PassengerId": submission.PassengerId,
                                "probability": probability,
                                "prediction": submission.Transported})
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "test_predictions.csv", index=False,
                       encoding="utf-8-sig", float_format="%.17g")
    submission.to_csv(output / "submission.csv", index=False, encoding="utf-8-sig")
    print(f"已从保存的集成模型复现 {len(submission)} 位乘客：{output}")


if __name__ == "__main__":
    main()
