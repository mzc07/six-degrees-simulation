"""
visualization.py
────────────────
Genera todas las figuras del experimento:

  Fig 1 – L(p)/L(0) y C(p)/C(0) vs p  (replica Fig. 2 del paper W&S 1998).
  Fig 2 – L(p) absoluta con banda de desviación estándar.
  Fig 3 – C(p) absoluta con banda de desviación estándar.
  Fig 4 – Verificación "seis grados": histogram L sobre iteraciones.
  Fig 5 – Red visualizada para p=0, p≈0.01, p=1 (muestra topología).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin GUI para scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import pandas as pd


# ── Paleta de colores ────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
BG     = "#F9FAFB"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.facecolor"   : BG,
    "figure.facecolor" : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "grid.linestyle"   : "--",
})


def _log_p_axis(ax: plt.Axes, df: pd.DataFrame):
    """Configura eje x logarítmico con ticks en valores de p."""
    p_vals = df["p"].values
    p_nonzero = p_vals[p_vals > 0]
    ax.set_xscale("log")
    ax.set_xlim(p_nonzero.min() * 0.5, 1.2)
    ax.set_xlabel("Probabilidad de rewiring  p", fontsize=12)


# ── Figura 1: Normalizada (réplica Fig. 2 W&S) ───────────────────────────────
def plot_normalized(df: pd.DataFrame, fname: str = "fig1_normalized.png"):
    """
    Reproduce la figura emblemática del paper:
    L(p)/L(0) y C(p)/C(0) en función de p (escala log).
    """
    df_p = df[df["p"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_p["p"], df_p["L_norm"], "o-", color=BLUE,
            lw=2, ms=6, label=r"$L(p)\,/\,L(0)$  (camino promedio)")
    ax.plot(df_p["p"], df_p["C_norm"], "s-", color=RED,
            lw=2, ms=6, label=r"$C(p)\,/\,C(0)$  (clustering)")

    ax.axhspan(0, 0.15, alpha=0.07, color=GREEN,
               label="Zona small-world  (L pequeño, C alto)")

    ax.set_title("Watts-Strogatz (1998) – Figura 2\n"
                 "Transición de red regular a aleatoria",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor normalizado  (relativo a p = 0)", fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    _log_p_axis(ax, df)
    ax.legend(fontsize=11, loc="center left")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figura 2: L absoluta ─────────────────────────────────────────────────────
def plot_path_length(df: pd.DataFrame, fname: str = "fig2_path_length.png"):
    """L(p) absoluta con banda ±σ y línea de referencia seis grados."""
    df_p = df[df["p"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_p["p"], df_p["L_mean"], "o-", color=BLUE, lw=2, ms=6,
            label="L(p)  promedio")
    ax.fill_between(
        df_p["p"],
        df_p["L_mean"] - df_p["L_std"],
        df_p["L_mean"] + df_p["L_std"],
        alpha=0.2, color=BLUE, label="±1σ",
    )
    ax.axhline(6, ls="--", color=RED, lw=1.5,
               label="6 grados de separación")

    ax.set_title("Distancia promedio  L(p)  vs probabilidad de rewiring",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distancia promedio  L", fontsize=12)
    _log_p_axis(ax, df)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figura 3: C absoluta ─────────────────────────────────────────────────────
def plot_clustering(df: pd.DataFrame, fname: str = "fig3_clustering.png"):
    """C(p) absoluta con banda ±σ."""
    df_p = df[df["p"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_p["p"], df_p["C_mean"], "s-", color=RED, lw=2, ms=6,
            label="C(p)  promedio")
    ax.fill_between(
        df_p["p"],
        df_p["C_mean"] - df_p["C_std"],
        df_p["C_mean"] + df_p["C_std"],
        alpha=0.2, color=RED, label="±1σ",
    )
    ax.axhline(1 / 3, ls="--", color=GRAY, lw=1.5,
               label="C teórico red aleatoria  ≈ k/n")

    ax.set_title("Coeficiente de clustering  C(p)  vs probabilidad de rewiring",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Clustering  C", fontsize=12)
    _log_p_axis(ax, df)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figura 4: Panel combinado (dashboard resumen) ─────────────────────────────
def plot_dashboard(df: pd.DataFrame, fname: str = "fig4_dashboard.png"):
    """Panel 2×2 con L norm, C norm, L abs y C abs."""
    df_p = df[df["p"] > 0].copy()
    p = df_p["p"].values

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Top-left: normalizada (réplica paper) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p, df_p["L_norm"], "o-", color=BLUE, lw=2, ms=5,
             label=r"$L/L_0$")
    ax1.plot(p, df_p["C_norm"], "s-", color=RED,  lw=2, ms=5,
             label=r"$C/C_0$")
    ax1.set_xscale("log"); ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Normalizado  (réplica Fig. 2 W&S 1998)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("p"); ax1.set_ylabel("Valor normalizado")
    ax1.legend(fontsize=9)

    # ── Top-right: L abs ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p, df_p["L_mean"], "o-", color=BLUE, lw=2, ms=5)
    ax2.fill_between(p, df_p["L_mean"]-df_p["L_std"],
                        df_p["L_mean"]+df_p["L_std"],
                     alpha=0.2, color=BLUE)
    ax2.axhline(6, ls="--", color=RED, lw=1.5, label="6 grados")
    ax2.set_xscale("log")
    ax2.set_title("Distancia promedio  L(p)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("p"); ax2.set_ylabel("L"); ax2.legend(fontsize=9)

    # ── Bottom-left: C abs ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(p, df_p["C_mean"], "s-", color=RED, lw=2, ms=5)
    ax3.fill_between(p, df_p["C_mean"]-df_p["C_std"],
                        df_p["C_mean"]+df_p["C_std"],
                     alpha=0.2, color=RED)
    ax3.set_xscale("log")
    ax3.set_title("Clustering  C(p)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("p"); ax3.set_ylabel("C")

    # ── Bottom-right: tabla p óptimo ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    sw_mask = (df_p["L_norm"] < 0.3) & (df_p["C_norm"] > 0.5)
    sw_rows = df_p[sw_mask][["p", "L_mean", "C_mean", "L_norm", "C_norm"]]
    if not sw_rows.empty:
        headers = ["p", "L", "C", "L/L₀", "C/C₀"]
        cell_text = sw_rows.apply(
            lambda r: [f"{r.p:.4f}", f"{r.L_mean:.2f}",
                       f"{r.C_mean:.4f}", f"{r.L_norm:.3f}",
                       f"{r.C_norm:.3f}"], axis=1
        ).tolist()
        tbl = ax4.table(cellText=cell_text, colLabels=headers,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
    ax4.set_title("Zona small-world\n(L pequeño + C alto)",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Simulación Seis Grados de Separación  –  Watts-Strogatz",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figura 5: Visualización de redes (n pequeño) ─────────────────────────────
def plot_network_examples(n_viz: int = 50, k: int = 4,
                          fname: str = "fig5_networks.png"):
    """
    Dibuja tres topologías:
      - p=0   → anillo regular
      - p=0.1 → small-world
      - p=1   → aleatoria
    Usa n pequeño para legibilidad visual.
    """
    import networkx as nx

    configs = [
        (0.0,  "Regular (p=0)\nL grande, C alto"),
        (0.1,  "Small-World (p=0.1)\nL pequeño, C alto  ← ÓPTIMO"),
        (1.0,  "Aleatoria (p=1)\nL pequeño, C bajo"),
    ]
    colors_node = [BLUE, GREEN, RED]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pos_ring = nx.circular_layout(nx.cycle_graph(n_viz))

    for ax, (p, title), nc in zip(axes, configs, colors_node):
        G = nx.watts_strogatz_graph(n_viz, k, p, seed=42)
        nx.draw_networkx(
            G, pos=pos_ring, ax=ax,
            node_size=60, node_color=nc, alpha=0.8,
            with_labels=False,
            edge_color=GRAY, width=0.6,
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Topologías de red Watts-Strogatz  (n={n_viz}, k={k})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Entry point ───────────────────────────────────────────────────────────────
def plot_all(df: pd.DataFrame):
    """Genera todas las figuras."""
    print("\n── Generando gráficas ──")
    plot_normalized(df)
    plot_path_length(df)
    plot_clustering(df)
    plot_dashboard(df)
    plot_network_examples()
