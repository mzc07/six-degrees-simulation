"""
simulation.py
─────────────
Orquesta el experimento completo:
  • Para cada valor de p en P_VALUES,
  • Repite N_ITER veces (para promediar la estocasticidad del rewiring),
  • Construye grafo Watts-Strogatz y calcula métricas L y C.

Retorna lista de resultados raw listos para ser resumidos por metrics.py.
"""

import numpy as np
from typing import List, Dict

from graph_builder import build_small_world
from metrics import compute_metrics


def run_experiment(
    n: int,
    k: int,
    p_values: List[float],
    n_iter: int,
) -> List[Dict]:
    """
    Ejecuta el barrido completo de probabilidades de rewiring.

    Args:
        n       : Número de nodos de la red.
        k       : Conexiones por nodo en la red anillo base.
        p_values: Array/lista de valores de p a explorar.
        n_iter  : Número de realizaciones por valor de p.

    Returns:
        Lista de dicts: [{'p': float, 'iter': int, 'L': float, 'C': float}, ...]
    """
    results: List[Dict] = []
    total = len(p_values) * n_iter

    for i, p in enumerate(p_values):
        L_vals: List[float] = []
        C_vals: List[float] = []

        for it in range(n_iter):
            seed = int(np.random.randint(0, 2**31))
            G = build_small_world(n, k, p, seed=seed)
            m = compute_metrics(G)
            L_vals.append(m["L"])
            C_vals.append(m["C"])
            results.append({"p": p, "iter": it, "L": m["L"], "C": m["C"]})

            done = i * n_iter + it + 1
            pct  = done / total * 100
            bar  = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(
                f"\r[{bar}] {pct:5.1f}%  p={p:.4f}  iter={it+1}/{n_iter}"
                f"  L={m['L']:.3f}  C={m['C']:.4f}",
                end="", flush=True,
            )

        print(
            f"\r  p={p:.5f}  |  L={np.mean(L_vals):.3f}±{np.std(L_vals):.3f}"
            f"  |  C={np.mean(C_vals):.4f}±{np.std(C_vals):.4f}"
            + " " * 20
        )

    return results
