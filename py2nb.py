"""把 `# %%` 分块的 .py 源码转换成 .ipynb。

为什么要这个：notebook 的 JSON 手写容易出错、也难 diff。这里用普通 .py 写分析代码
（照常能被 IDE 检查、能被 git 看清改了什么），再一键生成 .ipynb。

约定（jupytext 的 light 格式子集）：
    # %%                     -> 新代码单元
    # %% [markdown]          -> 新 markdown 单元（其后每行前面的 "# " 会被去掉）
    # %% 标题：随便写          -> 同上，方括号里的注释只是给人看的分隔标题

用法：
    python py2nb.py src/01_数据体检.py notebooks/01_数据体检.ipynb
    python py2nb.py src/*.py --outdir notebooks      # 批量
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

MARKER = re.compile(r"^#\s*%%(.*)$")
KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "name": "python",
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}


def _clean_markdown(lines: list[str]) -> str:
    """markdown 单元里允许每行写 '# '，这里统一去掉，方便在 .py 里书写。"""
    stripped = []
    for ln in lines:
        if ln.startswith("# "):
            stripped.append(ln[2:])
        elif ln.strip() == "#":
            stripped.append("")
        else:
            stripped.append(ln)
    return "".join(stripped)


def parse_cells(text: str) -> list[dict]:
    lines = text.splitlines(keepends=True)
    cells: list[dict] = []
    current: list[str] | None = None
    is_md = False

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        src = "".join(current)
        if src.strip():
            cells.append(
                {
                    "cell_type": "markdown" if is_md else "code",
                    "metadata": {},
                    "source": _clean_markdown(current) if is_md else src,
                }
            )
        current = None

    for ln in lines:
        m = MARKER.match(ln.rstrip("\n"))
        if m is not None:
            flush()
            label = m.group(1).strip()
            is_md = label.startswith("[markdown]")
            current = []
            continue
        if current is not None:
            current.append(ln)
    flush()
    return cells


def to_notebook(cells: list[dict]) -> dict:
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNELSPEC, "language_info": LANGUAGE_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert(src: str, dst: str) -> int:
    with open(src, encoding="utf-8") as fh:
        cells = parse_cells(fh.read())
    nb = to_notebook(cells)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{src} -> {dst}  ({len(cells)} 单元, {n_code} 代码单元)")
    return len(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    files = [f for pat in args.src for f in (glob.glob(pat) or [pat])]
    if args.outdir:
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0] + ".ipynb"
            convert(f, os.path.join(args.outdir, name))
        return 0
    if len(files) != 1 or not args.dst:
        ap.error("单个源文件时需提供目标 .ipynb；多个源文件请用 --outdir")
    convert(files[0], args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
