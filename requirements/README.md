# Python 依赖清单

所有命令从仓库根目录执行。复现最终模型时，先按[项目主页](../README.md)安装匹配 CUDA 的 PyTorch，再执行：

```powershell
python -m pip install -r requirements/requirements-final.txt
```

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 数据分析、树模型训练和基础测试的固定依赖 |
| `requirements-final.txt` | 最终融合模型，包含基础依赖与 TabICL 依赖 |
| `requirements-tabicl.txt` | 独立 CUDA 环境中的 TabICL 依赖 |
| `requirements-tabpfn.txt` | 历史 TabPFN 对照实验的附加依赖 |
| `requirements-tabm.txt` | 历史 TabM 对照实验的附加依赖 |
| `requirements-finetune.txt` | 微调探索实验的附加依赖 |
| `requirements-ebm.txt` | EBM 对照实验，包含基础依赖 |

各文件中的版本号保持不变；`-r` 引用相对于当前依赖文件所在目录解析。可选实验按各自文档使用独立环境，不需要为复现最终模型安装全部实验依赖。
