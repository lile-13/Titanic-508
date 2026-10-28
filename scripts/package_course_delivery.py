"""打包课程源码与可复现产物，排除环境、Git和中间缓存。"""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"outputs/delivery_20261028_4690"
NAME="大作业源代码-待填班级-待填学号-待填姓名"


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    files={}
    allowed={"src","scripts","tests","notebooks","docs","features","reports"}
    for folder in allowed:
        for path in (ROOT/folder).rglob("*"):
            if not path.is_file():continue
            if any(part in {"__pycache__",".ipynb_checkpoints"} for part in path.parts):continue
            if path.suffix in {".pyc",".pyo"}:continue
            files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for relative in ["README.md","requirements.txt",".gitignore",
        "data/raw/train.csv","data/raw/test.csv","data/raw/sample_submission.csv",
        "models/optimization_20261025/ensemble.joblib",
        "models/full_data_candidate_20261028/ensemble.joblib",
        "outputs/optimization_20261025/submission.csv",
        "outputs/optimization_20261025/test_predictions.csv",
        "outputs/full_data_candidate_20261028/submission_fullblend4690.csv",
        "outputs/full_data_candidate_20261028/test_predictions.csv",
        "outputs/kaggle_ready_4690/submission4690.csv",
        "outputs/kaggle_ready_4690/submission_ensemble4690.csv",
        "outputs/kaggle_ready_4690/submission_fullblend4690.csv",
        "outputs/kaggle_ready_4690/manifest.json"]:
        path=ROOT/relative
        files[relative]=path.read_bytes()
    # Include native DOCX template for optional report regeneration, not internal QA renders.
    files["resources/course_report_template.docx"]=(DEST/"work/reference_report.docx").read_bytes()
    files["交付说明.md"]=("""# 课程成果与复现说明

更新：2026-10-28。此目录包含源码、原始公开数据、折划分、实验记录、图表、当前最佳模型及一份待验证的训练方式对照。个人姓名、完整学号、班级和贡献率由组长填写；Team四人名单截图需在报告中补入。

当前已观测Kaggle最佳：outputs/kaggle_ready_4690/submission_ensemble4690.csv，0.81108。
可选新对照：outputs/kaggle_ready_4690/submission_fullblend4690.csv，分数未知。它采用固定参数全量训练和原五折集成各50%，有34个预测标签变化；没有新的OOF分数，不应写成已提高成绩。

在解压后的本目录执行：
```powershell
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python -m unittest discover -s tests -v
.venv\\Scripts\\python -m scripts.predict_ensemble --bundle models/optimization_20261025/ensemble.joblib --output-dir outputs/reload_best
.venv\\Scripts\\python -m scripts.predict_ensemble --bundle models/full_data_candidate_20261028/ensemble.joblib --output-dir outputs/reload_comparison
```

不使用附带权重、按冻结配置从零训练：
```powershell
.venv\\Scripts\\python -m scripts.refit_frozen --run-name fresh_refit --threads 4
```
每次使用新的run-name，避免覆盖已有产物。测试通过环境为Windows、Python3.14.6，完整依赖见requirements.txt。隔离复现使用同机同环境，不代表已经由其他成员或另一台机器复现。

所有文件哈希见MANIFEST.sha256；包内原始数据属于竞赛公开数据，只用于课程项目。保留原始PassengerId字符串前导零。模型训练依赖不包含报告排版库；报告如需从模板重建，额外需要python-docx与Pillow，并使用Word更新目录。脚本默认使用课程模板资源。最终报告DOCX在本ZIP之外作为单独课程材料提交。

追加搜索脚本依赖未纳入ZIP的开发期概率缓存；只需复现最终冻结模型时不必重跑搜索。若要从零重做旧搜索，先按docs/成员1_模型优化与复现.md运行搜索，生成缓存后再运行追加实验。历史被否决特征的脚本也可能需要其原实验期定义，不能把空列结果当作有效复现。

不要把本包内的模型或公开分数解释为新的独立测试成绩。Kaggle提交由组长操作，雨课堂报告和源码按课程入口分别提交。ZIP名和报告名中的待填字段需按实际班级、完整学号和姓名重命名。
""").encode("utf-8")
    manifest="".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name,data in sorted(files.items()))
    files["MANIFEST.sha256"]=manifest.encode("utf-8")
    archive=DEST/(NAME+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for name,data in sorted(files.items()):package.writestr(NAME+"/"+name,data)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        for name,data in files.items():assert package.read(NAME+"/"+name)==data
        names=package.namelist()
        assert not any("/.git/" in n or "/.venv/" in n or "/__pycache__/" in n for n in names)
    check=dict(passed=True,files=len(files),zip_bytes=archive.stat().st_size,
        zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive=str(archive.relative_to(ROOT)),
        manifest_verified=True,includes_best_model=True,includes_unsubmitted_comparison=True,
        includes_python_environment=False,includes_git=False,personal_fields="user will fill",
        source_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        includes_uncommitted_work=True)
    (DEST/"package_check.json").write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
