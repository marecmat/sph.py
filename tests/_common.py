"""Shared helpers for the case scripts in this folder."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGDIR = Path(__file__).resolve().parent / "figures"
SHOW = "--no-show" not in sys.argv


def finish(fig, name):
    """Save a figure and, unless --no-show was passed, display it."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")
    if SHOW:
        import matplotlib.pyplot as plt
        plt.show()
