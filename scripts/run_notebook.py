"""Execute a .ipynb with the local Jupyter kernel.

Why this exists: launching a Jupyter kernel inside the DSH Windows sandbox fails because
jupyter_core.paths.secure_write() calls win32security.SetFileSecurity() to tighten the ACL of
the connection file, and the sandboxed process token has no WRITE_DAC -> pywintypes.error (5,
'SetFileSecurity', 'access denied'). jupyter_core ships an official opt-out for exactly this
kind of environment: JUPYTER_ALLOW_INSECURE_WRITES=true. It only skips the ACL tightening on
the kernel connection file (which holds the ports + HMAC key of a loopback-only kernel), so
the file inherits normal user-profile permissions instead of being narrowed to the single user.

Usage:
    python -m scripts.run_notebook <notebook.ipynb> [--timeout 600]
Executes in place (outputs written back into the .ipynb) and prints a compact report.
"""
import os
import sys
import argparse

os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "true")

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--cwd", default=None)
    args = ap.parse_args()

    nb_path = os.path.abspath(args.notebook)
    workdir = os.path.abspath(args.cwd) if args.cwd else os.path.dirname(nb_path)
    os.chdir(workdir)

    with open(nb_path, encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": workdir}},
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)

    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    errs = [
        (i, o)
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    print(f"executed {nb_path}: {n_code} code cells, errors={len(errs)}")
    for i, o in errs:
        print(f"  cell {i}: {o['ename']}: {o['evalue']}")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
