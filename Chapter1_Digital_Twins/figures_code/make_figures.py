#!/usr/bin/env python3
"""Figures for Chapter 1: Digital Twins for Sustainable Systems.

Renders four 2000 x 2000 px (10 in x 10 in @ 200 DPI) black-and-white,
PowerPoint-style figures with 12 pt Arial text into ../figures/.
Run:  python3 Chapter1_Digital_Twins/figures_code/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import bw_figure_style as S  # noqa: E402

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)


def figure1_architecture():
    fig, ax = S.new_canvas()
    layers = [
        ("Layer 6: Decision & Control Layer", "Optimisation, MPC, policy rules, operator HMI, actuation"),
        ("Layer 5: Analytics & AI Layer", "Data assimilation, PINNs, surrogates, anomaly detection, XAI"),
        ("Layer 4: Modelling & Simulation Layer", "Physics FEM/CFD, hydraulic & energy models, ROM/POD"),
        ("Layer 3: Communication & Data Layer", "OPC UA, MQTT, edge/cloud, time-series stores, metadata"),
        ("Layer 2: Sensing & IoT Layer", "In-situ sensors, smart meters, remote sensing, LiDAR"),
        ("Layer 1: Physical Entity Layer", "Environment, grid, water basin, building, city district"),
    ]
    x0, w, h = 0.16, 0.68, 0.105
    ys = np.linspace(0.80, 0.08, len(layers))
    for i, ((name, comp), y) in enumerate(zip(layers, ys)):
        fill = S.FILL_LIGHT if i % 2 == 0 else S.FILL_MID
        S.box(ax, x0, y, w, h, fill=fill)
        ax.text(0.5, y + h * 0.66, name, ha="center", va="center", fontweight="bold")
        ax.text(0.5, y + h * 0.30, comp, ha="center", va="center")
        if i < len(layers) - 1:
            S.arrow(ax, (0.5, y), (0.5, ys[i + 1] + h), style="<|-|>")
    # Upward and downward information-flow rails
    top, bot = ys[0] + h, ys[-1]
    S.arrow(ax, (0.08, bot), (0.08, top), lw=2.0)
    ax.text(0.05, (top + bot) / 2, "Observations and state estimates (upward)", rotation=90,
            ha="center", va="center")
    S.arrow(ax, (0.92, top), (0.92, bot), lw=2.0, ls="--")
    ax.text(0.95, (top + bot) / 2, "Decisions, set-points and actuation (downward)", rotation=90,
            ha="center", va="center")
    S.title(ax, "Figure 1: Six-Tier Architectural Layering for Sustainable Digital Twins")
    S.save(fig, OUT / "Figure1_DigitalTwin_Architecture.png")


def figure2_microgrid_loop():
    fig, ax = S.new_canvas()
    # Physical domain (bottom) and virtual domain (top), separated by a dashed boundary
    S.box(ax, 0.04, 0.05, 0.92, 0.35, fill=S.WHITE, ls="--", lw=1.2)
    ax.text(0.13, 0.37, "PHYSICAL MICROGRID", ha="left", va="center", fontweight="bold")
    S.box(ax, 0.04, 0.49, 0.92, 0.38, fill=S.WHITE, ls="--", lw=1.2)
    ax.text(0.06, 0.845, "VIRTUAL MICROGRID (DIGITAL TWIN)", ha="left", va="center", fontweight="bold")

    # Physical assets
    assets = [("PV array", 0.16), ("Battery\nstorage", 0.36), ("Controllable\nloads", 0.56),
              ("Grid\nconnection", 0.76)]
    for name, x in assets:
        S.box(ax, x, 0.21, 0.17, 0.11, name, fill=S.FILL_MID, bold=True)
        S.arrow(ax, (x + 0.085, 0.21), (x + 0.085, 0.15), style="-")
    S.box(ax, 0.07, 0.08, 0.86, 0.07, "Meters, inverters, BMS telemetry (P, V, I, SoC, T)",
          fill=S.FILL_LIGHT)

    # Virtual components
    S.box(ax, 0.07, 0.53, 0.26, 0.12, "State estimation\n(EnKF: SoC, loads)", fill=S.FILL_LIGHT, bold=True)
    S.box(ax, 0.37, 0.53, 0.26, 0.12, "Forecasting\n(PV, demand, $\\gamma_t$)", fill=S.FILL_LIGHT, bold=True)
    S.box(ax, 0.67, 0.53, 0.26, 0.12, "Carbon-aware\noptimiser (MPC)", fill=S.FILL_MID, bold=True)
    S.box(ax, 0.07, 0.71, 0.26, 0.10, "Sensor error\nmodels ($\\mathbf{R}_k$)", fill=S.WHITE)
    S.box(ax, 0.37, 0.71, 0.26, 0.10, "Price & carbon-\nintensity signals", fill=S.WHITE)
    S.box(ax, 0.67, 0.71, 0.26, 0.10, "Operating limits\n& governance", fill=S.WHITE)
    S.arrow(ax, (0.33, 0.59), (0.37, 0.59))
    S.arrow(ax, (0.63, 0.59), (0.67, 0.59))
    for xc in (0.20, 0.50, 0.80):
        S.arrow(ax, (xc, 0.71), (xc, 0.65))

    # Loop closure: measurements up (left), set-points down (right)
    S.arrow(ax, (0.10, 0.15), (0.10, 0.53), lw=2.2)
    ax.text(0.115, 0.445, "1. Measurements", ha="left", va="center", fontweight="bold")
    S.arrow(ax, (0.90, 0.53), (0.90, 0.32), lw=2.2, ls="--")
    ax.text(0.885, 0.445, "3. Dispatch\nset-points", ha="right", va="center", fontweight="bold")
    ax.text(0.50, 0.445, "2. Assimilate \u2192 forecast \u2192 optimise", ha="center", va="center",
            fontweight="bold")
    ax.text(0.50, 0.025, "4. Effects of dispatch are measured at the next cycle (closed loop)",
            ha="center", va="center")
    S.title(ax, "Figure 2: Microgrid Energy Digital Twin Feedback Loop")
    S.save(fig, OUT / "Figure2_Microgrid_Feedback_Loop.png")


def figure3_pareto():
    """Conceptual cost-emission frontier on illustrative synthetic data (fixed seed)."""
    rng = np.random.default_rng(7)
    fig, ax = S.new_canvas(axis_off=False)
    # Synthetic candidate strategies: frontier J2 = a / (J1 - b) + c plus dominated cloud
    j1_front = np.linspace(1.0, 4.0, 14)
    j2_front = 1.2 / (j1_front - 0.6) + 0.4
    j1_dom = rng.uniform(1.2, 4.2, 45)
    j2_dom = 1.2 / (j1_dom - 0.6) + 0.4 + rng.uniform(0.15, 1.3, 45)
    ax.plot(j1_front, j2_front, "-", color=S.BLACK, lw=1.5, zorder=2)
    ax.scatter(j1_dom, j2_dom, s=55, facecolors=S.WHITE, edgecolors=S.MID, linewidths=1.3,
               label="Dominated strategies", zorder=3)
    ax.scatter(j1_front, j2_front, s=70, c=S.BLACK, marker="s", label="Pareto-optimal strategies",
               zorder=4)
    # Knee: point of maximum distance from the chord joining the extremes
    p0 = np.array([j1_front[0], j2_front[0]])
    p1 = np.array([j1_front[-1], j2_front[-1]])
    v, q = p1 - p0, np.c_[j1_front, j2_front] - p0
    d = np.abs(v[0] * q[:, 1] - v[1] * q[:, 0]) / np.linalg.norm(v)
    k = int(np.argmax(d))
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ":", color=S.DARK, lw=1.2)
    ax.scatter([j1_front[k]], [j2_front[k]], s=420, facecolors="none", edgecolors=S.BLACK,
               linewidths=2.0, zorder=5)
    ax.annotate("Knee point\n(balanced trade-off)", xy=(j1_front[k], j2_front[k]),
                xytext=(j1_front[k] + 0.45, j2_front[k] + 0.75),
                arrowprops=dict(arrowstyle="-|>", color=S.BLACK, lw=1.3))
    ax.annotate("Minimum-cost\nextreme", xy=tuple(p0), xytext=(p0[0] + 0.25, p0[1] + 0.25))
    ax.annotate("Minimum-emission\nextreme", xy=tuple(p1), xytext=(p1[0] - 0.75, p1[1] - 0.4))
    ax.set_xlabel("$J_1$: Operating cost (normalised units)")
    ax.set_ylabel("$J_2$: Operational CO$_2$ emissions (normalised units)")
    ax.set_xlim(0.8, 4.5)
    ax.set_ylim(0, 4.6)
    ax.grid(True, color=S.LIGHT, lw=0.8)
    ax.legend(loc="upper right", frameon=True, edgecolor=S.BLACK)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.text(0.99, 0.01, "Illustrative synthetic data", transform=ax.transAxes, ha="right",
            va="bottom", style="italic")
    S.title(ax, "Figure 3: Conceptual Cost–Emission Pareto Frontier of a Digital Twin")
    fig.tight_layout()
    S.save(fig, OUT / "Figure3_Pareto_Frontier.png")


def figure4_dtcl():
    fig, ax = S.new_canvas()
    levels = [
        ("DTCL-0", "Disconnected\nmodel", "Design data\nonly"),
        ("DTCL-1", "Batch\nshadow", "Periodic\ndata ingest"),
        ("DTCL-2", "Streaming\nshadow", "Continuous\nstreaming"),
        ("DTCL-3", "Assimilating\ntwin", "Formal state\nestimation"),
        ("DTCL-4", "Advisory\ntwin", "Optimised\nrecommendation"),
        ("DTCL-5", "Autonomous\ntwin", "Governed\nactuation"),
    ]
    fills = [S.WHITE, S.FILL_LIGHT, S.FILL_LIGHT, S.FILL_MID, S.FILL_MID, S.LIGHT]
    w, step_h, x0, y0 = 0.145, 0.085, 0.05, 0.30
    for i, ((code, name, enabler), f) in enumerate(zip(levels, fills)):
        x = x0 + i * (w + 0.007)
        h = step_h * (i + 1) + 0.12
        S.box(ax, x, y0, w, h, fill=f, style="square")
        ax.text(x + w / 2, y0 + h - 0.035, code, ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y0 + h - 0.10, name, ha="center", va="center")
        ax.text(x + w / 2, y0 - 0.065, enabler, ha="center", va="center")
    S.arrow(ax, (0.05, 0.155), (0.95, 0.155), lw=2.0)
    ax.text(0.50, 0.12, "Increasing coupling, statistical rigour and autonomy", ha="center",
            va="center", fontweight="bold")
    # Mapping to Kritzinger et al. classes
    spans = [(0, 0, "Digital model"), (1, 2, "Digital shadow"), (3, 5, "Digital twin")]
    for a, b, lab in spans:
        xa = x0 + a * (w + 0.007)
        xb = x0 + b * (w + 0.007) + w
        S.arrow(ax, (xa, 0.06), (xb, 0.06), style="|-|", lw=1.2)
        ax.text((xa + xb) / 2, 0.03, lab, ha="center", va="center")
    ax.text(0.05, 0.90, "Physical → virtual flow automated from DTCL-1;\n"
            "virtual → physical flow automated at DTCL-5;\n"
            "human closes the loop at DTCL-4.", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc=S.FILL_LIGHT, ec=S.BLACK, lw=1.2))
    S.title(ax, "Figure 4: Digital Twin Coupling Level (DTCL) Maturity Pathway")
    S.save(fig, OUT / "Figure4_DTCL_Maturity_Pathway.png")


if __name__ == "__main__":
    figure1_architecture()
    figure2_microgrid_loop()
    figure3_pareto()
    figure4_dtcl()
