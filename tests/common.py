"""Shared helpers for the test scripts."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt


def finish(fig, name, show=True):
    """Save a test figure next to the calling script, then display it."""
    path = Path(sys.argv[0]).resolve().parent / name
    fig.savefig(path, dpi=150)
    if show and "--no-show" not in sys.argv:
        plt.show()
    return path
