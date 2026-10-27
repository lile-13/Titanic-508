"""复现已保存模型：python -m scripts.predict --bundle models/<run>/bundle.joblib。"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PROJECT_ROOT
from src.train import predict_saved_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / args.bundle.parent.name / "reproduced"
    if output_dir.exists():
        raise FileExistsError("输出目录已存在，请指定新的 --output-dir")
    predictions, submission = predict_saved_model(args.bundle, data_dir=args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False,
                       encoding="utf-8-sig", float_format="%.17g")
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    print(f"已复现 {len(submission)} 位乘客的预测：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
