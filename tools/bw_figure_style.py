"""Shared black-and-white, PowerPoint-style figure helpers.

Editor requirements enforced here:
  * canvas 10 in x 10 in at 200 DPI -> 2000 x 2000 px (>= 1500 x 1500 px),
  * Arial 12 pt for every text element (Liberation Sans, which is metrically
    identical to Arial, is used automatically if Arial is not installed),
  * pure black / white / grayscale palette, no colour.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FIG_SIZE_IN = (10, 10)
DPI = 200
FONT_PT = 12

# Grayscale palette
BLACK = "#000000"
DARK = "#404040"
MID = "#808080"
LIGHT = "#D3D3D3"
FILL_LIGHT = "#F0F0F0"
FILL_MID = "#D9D9D9"
WHITE = "#FFFFFF"


def _pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


FONT_NAME = _pick_font()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_NAME, "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_PT,
    "axes.titlesize": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "figure.titlesize": FONT_PT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_NAME,
    "mathtext.it": f"{FONT_NAME}:italic",
    "mathtext.bf": f"{FONT_NAME}:bold",
    "mathtext.cal": FONT_NAME,
    "mathtext.sf": FONT_NAME,
    "text.color": BLACK,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
})


def new_canvas(axis_off=True):
    fig, ax = plt.subplots(figsize=FIG_SIZE_IN, dpi=DPI)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    if axis_off:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fill=FILL_LIGHT, lw=1.5, bold=False, style="round", ls="-",
        text_kw=None):
    """Rounded PowerPoint-style box; (x, y) is the lower-left corner."""
    bs = "round,pad=0.005,rounding_size=0.015" if style == "round" else "square,pad=0.005"
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=bs, linewidth=lw, edgecolor=BLACK,
                                  facecolor=fill, linestyle=ls)
    ax.add_patch(rect)
    if text:
        kw = dict(ha="center", va="center", fontsize=FONT_PT, color=BLACK,
                  fontweight="bold" if bold else "normal", linespacing=1.3)
        kw.update(text_kw or {})
        ax.text(x + w / 2, y + h / 2, text, **kw)
    return rect


def arrow(ax, xy_from, xy_to, style="-|>", lw=1.5, ls="-", color=BLACK, rad=0.0, label=None,
          label_xy=None, label_kw=None):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=18,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        lx, ly = label_xy if label_xy else ((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2)
        kw = dict(ha="center", va="center", fontsize=FONT_PT, color=BLACK,
                  bbox=dict(boxstyle="square,pad=0.15", fc=WHITE, ec="none"))
        kw.update(label_kw or {})
        ax.text(lx, ly, label, **kw)


def title(ax, text):
    ax.set_title(text, fontsize=FONT_PT, fontweight="bold", pad=14, color=BLACK)


def save(fig, path):
    fig.savefig(path, dpi=DPI, facecolor=WHITE)
    plt.close(fig)
    _check(path)


def _check(path):
    """Verify size and that the image contains no colour (R == G == B everywhere)."""
    img = plt.imread(path)
    h, w = img.shape[:2]
    rgb = img[..., :3]
    colour = float(abs(rgb[..., 0] - rgb[..., 1]).max() + abs(rgb[..., 1] - rgb[..., 2]).max())
    assert w >= 1500 and h >= 1500, f"{path}: {w}x{h} px is below 1500x1500"
    assert colour < 1e-6, f"{path}: non-grayscale pixels detected"
    print(f"saved {path}: {w}x{h} px, grayscale OK, font={FONT_NAME} {FONT_PT} pt")
