"""Shared typography for raster figures embedded in the thesis."""

from pathlib import Path

import matplotlib
from matplotlib import font_manager


BLACK = "#000000"


def apply_thesis_style() -> str:
    """Register Latin Modern Roman Demi and use solid black figure text."""
    font_path = (
        Path.home()
        / ".TinyTeX"
        / "texmf-dist"
        / "fonts"
        / "opentype"
        / "public"
        / "lm"
        / "lmromandemi10-regular.otf"
    )
    assert font_path.exists(), font_path
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    assert family == "Latin Modern Roman Demi", family
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.weight": 600,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.labelweight": 600,
            "axes.titlecolor": BLACK,
            "axes.titleweight": 600,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
        }
    )
    return family
