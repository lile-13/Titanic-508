"""Package the fixed final model, public data, source and author checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BEST = "outputs/kaggle_ready_4690/next_batch_20261031/submission_tabicl75_4690.csv"
BUNDLE = "models/manual_submission_batch_20261031_tabicl75/ensemble.joblib"
CHECKPOINT = "models/tabicl_pretrained/tabicl-classifier-v2-20260212.ckpt"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(destination, archive_name):
    if Path(archive_name).name != archive_name or any(c in archive_name for c in '/\\:'):
        raise ValueError("archive-name must be a single filename stem")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (archive_name + ".zip")
    if target.exists():
        raise FileExistsError(target)
    files = {}
    for folder in ["src", "scripts", "tests", "notebooks", "docs", "features", "reports", "third_party", "requirements"]:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or any(p in {"__pycache__", ".ipynb_checkpoints"} for p in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in [ROOT / "README.md", ROOT / ".gitignore", ROOT / ".gitattributes"]:
        files[path.name] = path.read_bytes()
    for relative in ["data/raw/train.csv", "data/raw/test.csv", "data/raw/sample_submission.csv", BUNDLE, CHECKPOINT, BEST]:
        files[relative] = (ROOT / relative).read_bytes()
    assert sha(files[BEST]) == "abd9924480c5ab75bb677384e9251b81ab3195bc25c6e750834b48f55ac807cf"
    assert sha(files[CHECKPOINT]) == "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
    files["交付说明.md"] = f"""# 最终课程源码包

本包包含源码、公开原始数据、固定折划分、实验记录、最终最佳组合模型及作者TabICLv2检查点。报告DOCX单独提交。历史0.81318来自已登记的平台提交，本包不产生新的平台分数。

从解压后的本目录运行。按README安装GPU环境与requirements/requirements-final.txt；作者检查点已经附带，不必再次下载。训练、推理所需代码与权重均在包内，不依赖原开发目录的搜索缓存。

```powershell
python -B -m scripts.predict_ensemble --bundle {BUNDLE} --output-dir outputs/final_replay
python -B -m scripts.rebuild_final --run-name fresh_final --dry-run
python -B -m scripts.rebuild_final --run-name fresh_final --threads 4
python -B -m unittest discover -s tests -q
```

这里的python必须是README所述已安装依赖的CUDA环境。每次使用全新run-name和输出目录。重建会拟合45个树成员、保存5个固定TabICL上下文，然后重载输出4277条预测，不更新作者网络参数。

对照文件为`{BEST}`。SHA256为`{sha(files[BEST])}`。包内MANIFEST.sha256逐项记录内容哈希；解压后可逐项复核。作者BSD许可见third_party/TabICL_LICENSE.txt。

历史搜索与报告生成脚本可能需要各自开发缓存或课程模板；复现最终模型使用上述两个入口即可。包含公共竞赛数据的课程包供课程提交使用，GitHub不保存原始数据或大型权重。
""".encode("utf-8")
    files["MANIFEST.sha256"] = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(files.items())).encode("utf-8")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(archive_name + "/" + name, data)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, data in files.items():
            assert sha(archive.read(archive_name + "/" + name)) == sha(data)
    result = dict(passed=True, files=len(files), zip_bytes=target.stat().st_size,
                  zip_sha256=sha(target.read_bytes()), manifest_verified=True,
                  includes_final_bundle=True, includes_author_checkpoint=True,
                  includes_environment=False, includes_git=False)
    (destination / "package_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--archive-name", default="final-course-source")
    args = parser.parse_args()
    main(args.destination, args.archive_name)
