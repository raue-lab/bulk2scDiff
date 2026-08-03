#!/usr/bin/env python
"""Offline notebook executor: run a notebook's code cells as a linear script in one
namespace, headless, saving every matplotlib figure to PNG. No jupyter/nbconvert needed.

    python nbrun.py <notebook.ipynb> [fig_out_dir]

Figures are written as <stem>_figs/fig_NNN.png. All printed output goes to stdout.
Exits non-zero (and prints a traceback) on the first failing cell.
"""
import sys, json, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = Path(sys.argv[1]).resolve()
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path.parent / f"{nb_path.stem}_figs"
outdir.mkdir(parents=True, exist_ok=True)

_n = {"i": 0}
def _save_open_figs():
    for num in plt.get_fignums():
        _n["i"] += 1
        try:
            plt.figure(num).savefig(outdir / f"fig_{_n['i']:03d}.png", dpi=200, bbox_inches="tight")
        except Exception as e:
            print(f"[nbrun] warn: could not save fig {num}: {e}")
    plt.close("all")

def _show(*a, **k):
    _save_open_figs()
plt.show = _show

def _display(*a, **k):
    pass
class _IP:
    def __getattr__(self, name):
        return lambda *a, **k: None
def get_ipython():
    return _IP()

ns = {"__name__": "__main__", "display": _display, "get_ipython": get_ipython}

j = json.load(open(nb_path))
code_cells = [c for c in j["cells"] if c["cell_type"] == "code"]
print(f"[nbrun] {nb_path.name}: {len(code_cells)} code cells; figs -> {outdir}")
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    lines = [ln for ln in src.splitlines()
             if not ln.lstrip().startswith(("%", "!"))]  # strip magics / shell escapes
    code = "\n".join(lines)
    try:
        exec(compile(code, f"{nb_path.name}#cell{i}", "exec"), ns)
        _save_open_figs()   # capture figures even if the cell never called show()
    except Exception:
        print(f"\n!!! ERROR in cell {i} of {nb_path.name}:\n")
        traceback.print_exc()
        sys.exit(3)

_save_open_figs()
print(f"\n[nbrun] DONE {nb_path.name}: saved {_n['i']} figure(s) to {outdir}")
