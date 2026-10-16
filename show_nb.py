"""打印 .ipynb 的文本输出（用于无 GUI 环境下核对执行结果）。

用法: python show_nb.py notebooks/01_数据体检.ipynb [--max-chars 400]
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--max-chars", type=int, default=350)
    args = ap.parse_args()

    with open(args.notebook, encoding="utf-8") as fh:
        nb = json.load(fh)

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        outs = cell.get("outputs", [])
        kinds = []
        texts = []
        for o in outs:
            if o.get("output_type") == "stream":
                texts.append("".join(o["text"]))
                kinds.append("stream")
            elif o.get("output_type") == "execute_result":
                texts.append("".join(o["data"].get("text/plain", [])))
                kinds.append("result")
            elif o.get("output_type") == "display_data":
                kinds.append("image" if "image/png" in o.get("data", {}) else "display")
            elif o.get("output_type") == "error":
                kinds.append("ERROR")
                texts.append(f"{o['ename']}: {o['evalue']}")
        body = "".join(texts).strip()
        if len(body) > args.max_chars:
            body = body[: args.max_chars] + f"\n... [截断，共 {len(body)} 字符]"
        print(f"\n===== cell {i} ===== outputs={kinds}")
        if body:
            print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
