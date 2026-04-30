"""
graph_builder.py
────────────────
Construye redes Watts-Strogatz (small-world).

Pasos del modelo (Watts & Strogatz, 1998):
  1. Red tipo anillo con n nodos, cada nodo conectado a k vecinos cercanos.
  2. Cada enlace se reconecta (rewire) con probabilidad p a un nodo aleatorio.

La implementación usa networkx.watts_strogatz_graph, que sigue exactamente
el algoritmo del paper original.
"""

import networkx as nx
import numpy as np


def build_ring_lattice(n: int, k: int) -> nx.Graph:
    """
    Construye la red regular inicial (anillo con k vecinos).

    Args:
        n: Número de nodos.
        k: Número de conexiones por nodo (debe ser par).

    Returns:
        Grafo NetworkX regular tipo anillo.
    """
    if k % 2 != 0:
        raise ValueError("k debe ser par para la red anillo.")
    return nx.watts_strogatz_graph(n, k, p=0, seed=None)


def build_small_world(n: int, k: int, p: float,
                      seed: int | None = None) -> nx.Graph:
    """
    Construye red small-world Watts-Strogatz.

    Args:
        n    : Número de nodos.
        k    : Conexiones por nodo en el anillo base.
        p    : Probabilidad de rewiring (0 = regular, 1 = aleatoria).
        seed : Semilla para reproducibilidad (None = aleatoria).

    Returns:
        Grafo NetworkX small-world.
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)
