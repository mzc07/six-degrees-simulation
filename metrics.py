"""
metrics.py
──────────
Calcula métricas de red del modelo Watts-Strogatz:

  • L(p)  – Distancia promedio característica (path length global).
  • C(p)  – Coeficiente de clustering promedio (local).

También normaliza L y C respecto a los valores de la red regular (p=0)
para reproducir la Figura 2 del paper de Watts & Strogatz (1998).
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict


def compute_avg_path_length(G: nx.Graph) -> float:
    """
    Calcula L: longitud de camino mínimo promedio entre todos los pares.

    Si el grafo no es conexo, usa solo el componente más grande
    (el grafo Watts-Strogatz casi nunca se desconecta para parámetros válidos).

    Args:
        G: Grafo NetworkX.

    Returns:
        Distancia promedio L.
    """
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    # Fallback: componente gigante
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    return nx.average_shortest_path_length(G_sub)


def compute_clustering(G: nx.Graph) -> float:
    """
    Calcula C: coeficiente de clustering promedio (transitivity local).

    Args:
        G: Grafo NetworkX.

    Returns:
        Clustering promedio C.
    """
    return nx.average_clustering(G)


def compute_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Devuelve dict con L y C para un grafo dado.

    Args:
        G: Grafo NetworkX.

    Returns:
        {'L': float, 'C': float}
    """
    return {
        "L": compute_avg_path_length(G),
        "C": compute_clustering(G),
    }


def summarize_results(raw: List[Dict]) -> pd.DataFrame:
    """
    Recibe lista de dicts con resultados por iteración y calcula:
      - Promedios de L y C por valor de p.
      - Normalización: L_norm = L(p)/L(0),  C_norm = C(p)/C(0).
      - Verifica "seis grados" (L ≈ 6).

    Args:
        raw: Lista de {'p', 'iter', 'L', 'C'}.

    Returns:
        DataFrame con columnas [p, L_mean, L_std, C_mean, C_std,
                                 L_norm, C_norm, six_degrees].
    """
    df_raw = pd.DataFrame(raw)
    grouped = df_raw.groupby("p").agg(
        L_mean=("L", "mean"),
        L_std=("L", "std"),
        C_mean=("C", "mean"),
        C_std=("C", "std"),
    ).reset_index()

    # Normalización respecto a p=0 (red regular)
    L0 = grouped.loc[grouped["p"] == 0, "L_mean"].values[0]
    C0 = grouped.loc[grouped["p"] == 0, "C_mean"].values[0]

    grouped["L_norm"] = grouped["L_mean"] / L0
    grouped["C_norm"] = grouped["C_mean"] / C0

    # Verificación seis grados: L ≈ 6
    grouped["six_degrees"] = grouped["L_mean"].apply(
        lambda x: f"{'✓' if 4 <= x <= 8 else '✗'}  L={x:.2f}"
    )

    # Redondear para display
    for col in ["L_mean", "L_std", "C_mean", "C_std", "L_norm", "C_norm"]:
        grouped[col] = grouped[col].round(4)

    return grouped
