#!/usr/bin/env python3
"""Figures for Chapter 2: AI-Based Climate Change Analysis and Prediction.

Four 2000 x 2000 px (10 in x 10 in @ 200 DPI) black-and-white figures, 12 pt Arial,
written to ../figures/.   Run: python3 Chapter2_Climate_AI/figures_code/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import bw_figure_style as S  # noqa: E402

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)


def figure1_convlstm_transformer():
    fig, ax = S.new_canvas()
    # Input sequence
    S.box(ax, 0.20, 0.86, 0.60, 0.06, "Input sequence $X_{t-k:t}$ (T2m, Z500, SST, q)", fill=S.WHITE)
    S.arrow(ax, (0.5, 0.86), (0.5, 0.81))
    stages = [
        (0.70, "ConvLSTM encoder", "Local spatiotemporal dynamics\n(gated convolutions, Eq. 2)", S.FILL_LIGHT),
        (0.55, "Patch embedding", "Feature maps $\\rightarrow$ tokens\n+ positional encoding", S.FILL_MID),
        (0.40, "Transformer encoder (N blocks)", "Multi-head self-attention (Eq. 3)\nglobal teleconnections", S.FILL_LIGHT),
        (0.25, "Decoder", "Upsampling + skip connections\nfrom ConvLSTM", S.FILL_MID),
    ]
    for y, name, desc, fill in stages:
        S.box(ax, 0.22, y, 0.56, 0.10, fill=fill)
        ax.text(0.5, y + 0.068, name, ha="center", va="center", fontweight="bold")
        ax.text(0.5, y + 0.03, desc, ha="center", va="center", linespacing=1.2)
    for y in (0.70, 0.55, 0.40):
        S.arrow(ax, (0.5, y), (0.5, y - 0.05))
    # Skip connection (routed outside the boxes on the left)
    ax.plot([0.22, 0.16, 0.16], [0.75, 0.75, 0.30], color=S.BLACK, lw=1.5, ls="--")
    S.arrow(ax, (0.16, 0.30), (0.22, 0.30), ls="--")
    ax.text(0.135, 0.525, "skip connection", rotation=90, ha="center", va="center")
    # Probabilistic heads
    S.arrow(ax, (0.5, 0.25), (0.5, 0.225), style="-")
    ax.plot([0.345, 0.655], [0.225, 0.225], color=S.BLACK, lw=1.5)
    S.arrow(ax, (0.345, 0.225), (0.345, 0.20))
    S.arrow(ax, (0.655, 0.225), (0.655, 0.20))
    S.box(ax, 0.22, 0.10, 0.25, 0.10, "Mean head\n$\\mu_\\theta(x)$", fill=S.WHITE, bold=True)
    S.box(ax, 0.53, 0.10, 0.25, 0.10, "Variance head\n$\\sigma^2_\\theta(x)$", fill=S.WHITE, bold=True)
    S.box(ax, 0.22, 0.01, 0.56, 0.055, "Loss: Gaussian NLL (Eq. 14), area-weighted by cos $\\phi$", fill=S.FILL_LIGHT)
    S.arrow(ax, (0.345, 0.10), (0.345, 0.065))
    S.arrow(ax, (0.655, 0.10), (0.655, 0.065))
    S.title(ax, "Figure 1: Hybrid ConvLSTM–Transformer Architecture for Climate Prediction")
    S.save(fig, OUT / "Figure1_ConvLSTM_Transformer.png")


def figure2_downscaling():
    fig, ax = S.new_canvas()
    S.box(ax, 0.05, 0.78, 0.38, 0.11, "Coarse predictors $X_c$\n(GCM / reanalysis, 25–250 km)", fill=S.WHITE)
    S.box(ax, 0.57, 0.78, 0.38, 0.11, "Static covariates $S$\n(orography, land–sea mask)", fill=S.WHITE)
    S.box(ax, 0.20, 0.58, 0.60, 0.11, "Convolutional encoder\n(multi-scale feature extraction)", fill=S.FILL_LIGHT, bold=True)
    S.box(ax, 0.20, 0.40, 0.60, 0.11, "Upsampling decoder (super-resolution)\n$\\hat{Y}_f = f_\\theta(X_c, S)$ (Eq. 4)", fill=S.FILL_MID, bold=True)
    S.box(ax, 0.20, 0.22, 0.60, 0.11, "Bias correction: quantile mapping\n$x_{corr}=F_o^{-1}(F_m(x))$ (Eq. 5)", fill=S.FILL_LIGHT, bold=True)
    S.box(ax, 0.20, 0.04, 0.60, 0.11, "High-resolution field (1–10 km)\nvalidated against stations / gridded obs.", fill=S.WHITE)
    S.arrow(ax, (0.24, 0.78), (0.38, 0.69))
    S.arrow(ax, (0.76, 0.78), (0.62, 0.69))
    for y in (0.58, 0.40, 0.22):
        S.arrow(ax, (0.5, y), (0.5, y - 0.07))
    S.box(ax, 0.84, 0.40, 0.14, 0.11, "Loss:\nMSE, GAN\nor diffusion", fill=S.FILL_LIGHT, style="square")
    S.arrow(ax, (0.80, 0.455), (0.84, 0.455), style="<|-|>")
    S.title(ax, "Figure 2: Machine-Learning Downscaling Pipeline")
    S.save(fig, OUT / "Figure2_Downscaling_Pipeline.png")


def figure3_pinn():
    fig, ax = S.new_canvas()
    S.box(ax, 0.30, 0.86, 0.40, 0.07, "Inputs: $(x, y, z, t)$", fill=S.WHITE, bold=True)
    S.box(ax, 0.30, 0.70, 0.40, 0.09, "Neural network\n$\\hat{C}(x,y,z,t;\\theta)$", fill=S.FILL_MID, bold=True)
    S.arrow(ax, (0.5, 0.86), (0.5, 0.79))
    # Observations feed the data term
    S.box(ax, 0.04, 0.86, 0.22, 0.07, "Observations $C_i$", fill=S.WHITE)
    # Residual row
    S.box(ax, 0.04, 0.48, 0.26, 0.14, "Data misfit\n$\\hat{C}(x_i,t_i)-C_i$", fill=S.FILL_LIGHT)
    S.box(ax, 0.36, 0.48, 0.28, 0.14, "PDE residual (Eq. 9)\n$R=\\partial_t\\hat{C}+u\\cdot\\nabla\\hat{C}-D\\nabla^2\\hat{C}-S$\n(auto-diff; known $u$, $D$, $S$)", fill=S.FILL_LIGHT)
    S.box(ax, 0.70, 0.48, 0.26, 0.14, "Boundary / initial\nconditions\n$\\mathcal{B}[\\hat{C}]-g$", fill=S.FILL_LIGHT)
    S.arrow(ax, (0.42, 0.70), (0.20, 0.62))
    S.arrow(ax, (0.50, 0.70), (0.50, 0.62))
    S.arrow(ax, (0.58, 0.70), (0.80, 0.62))
    ax.plot([0.08, 0.08], [0.86, 0.62], color=S.BLACK, lw=1.5, ls="--")
    S.arrow(ax, (0.08, 0.64), (0.08, 0.62), ls="--")
    # Loss terms
    losses = [(0.04, 0.26, "$\\mathcal{L}_{data}$\nweight $w_d$"),
              (0.36, 0.28, "$\\mathcal{L}_{physics}$\nweight $w_p$"),
              (0.70, 0.26, "$\\mathcal{L}_{boundary}$\nweight $w_b$")]
    for x, w, t in losses:
        S.box(ax, x, 0.30, w, 0.10, t, fill=S.FILL_MID, bold=True)
        S.arrow(ax, (x + w / 2, 0.48), (x + w / 2, 0.40))
    S.box(ax, 0.20, 0.14, 0.60, 0.09, "$\\mathcal{L}=w_d\\mathcal{L}_{data}+w_p\\mathcal{L}_{physics}+w_b\\mathcal{L}_{boundary}$ (Eq. 10)", fill=S.FILL_LIGHT, bold=True)
    S.arrow(ax, (0.17, 0.30), (0.34, 0.23))
    S.arrow(ax, (0.50, 0.30), (0.50, 0.23))
    S.arrow(ax, (0.83, 0.30), (0.66, 0.23))
    S.box(ax, 0.20, 0.02, 0.60, 0.07, "Gradient-descent update of $\\theta$ (and of $D$, $S$ if unknown)", fill=S.WHITE)
    S.arrow(ax, (0.5, 0.14), (0.5, 0.09))
    # Iteration loop routed in the right margin
    ax.plot([0.80, 0.985, 0.985], [0.055, 0.055, 0.745], color=S.BLACK, lw=1.5, ls="--")
    S.arrow(ax, (0.985, 0.745), (0.70, 0.745), ls="--")
    ax.text(0.89, 0.09, "iterate until\nconverged", ha="center", va="center")
    S.title(ax, "Figure 3: Physics-Informed Neural Network (PINN) Coupling Flowchart")
    S.save(fig, OUT / "Figure3_PINN_Flowchart.png")


def figure4_reliability():
    """Reliability diagram on illustrative synthetic data (fixed seed)."""
    rng = np.random.default_rng(11)
    fig, ax = S.new_canvas(axis_off=False)
    # Over-confident forecast: observed frequency is pulled toward the base rate
    p = np.linspace(0.05, 0.95, 10)
    obs_over = np.clip(0.5 + 0.55 * (p - 0.5) + rng.normal(0, 0.012, p.size), 0, 1)
    obs_cal = np.clip(p + rng.normal(0, 0.015, p.size), 0, 1)
    ax.plot([0, 1], [0, 1], "-", color=S.BLACK, lw=1.5, label="Perfect reliability")
    ax.plot(p, obs_over, "-o", color=S.BLACK, mfc=S.BLACK, ms=8, lw=1.5, label="Over-confident forecast")
    ax.plot(p, obs_cal, "--s", color=S.MID, mfc=S.WHITE, mec=S.DARK, ms=8, lw=1.5, label="After recalibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed relative frequency")
    ax.grid(True, color=S.LIGHT, lw=0.8)
    ax.legend(loc="upper left", frameon=True, edgecolor=S.BLACK)
    ax.set_aspect("equal")
    # Sharpness inset
    ins = ax.inset_axes([0.62, 0.10, 0.34, 0.20])
    counts = np.array([30, 18, 12, 9, 7, 7, 9, 12, 18, 30])
    ins.bar(p, counts, width=0.08, color=S.MID, edgecolor=S.BLACK)
    ins.set_xticks([0, 0.5, 1])
    ins.set_yticks([])
    ins.set_title("Sharpness (count)", fontsize=S.FONT_PT, pad=3)
    ins.tick_params(labelsize=S.FONT_PT)
    ax.text(0.99, 0.42, "Illustrative synthetic data", transform=ax.transAxes, ha="right", va="top",
            style="italic")
    S.title(ax, "Figure 4: Reliability Diagram for a Probabilistic Event Forecast")
    fig.tight_layout()
    S.save(fig, OUT / "Figure4_Reliability_Diagram.png")


if __name__ == "__main__":
    figure1_convlstm_transformer()
    figure2_downscaling()
    figure3_pinn()
    figure4_reliability()
