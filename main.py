"""
Six Degrees of Separation - Watts-Strogatz Small-World Simulation
Entry point: runs full experiment and generates all plots.
"""

from simulation import run_experiment
from metrics import summarize_results
from visualization import plot_all
import pandas as pd

# ── Parámetros del experimento ──────────────────────────────────────────────
N        = 1000   # número de nodos (tamaño de la red social)
K        = 10     # conexiones por nodo (vecinos en anillo inicial)
N_ITER   = 10     # iteraciones por valor de p (promediar estocasticidad)
P_VALUES = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01,
            0.05, 0.1, 0.2, 0.5, 1.0]   # barrido de p

if __name__ == "__main__":
    print("=" * 60)
    print(" SIMULACIÓN SEIS GRADOS DE SEPARACIÓN")
    print(" Modelo Watts-Strogatz (1998)")
    print("=" * 60)
    print(f" n={N}, k={K}, iteraciones={N_ITER}")
    print("=" * 60)

    # 1. Correr simulaciones
    raw_results = run_experiment(N, K, P_VALUES, N_ITER)

    # 2. Resumir / normalizar resultados
    df = summarize_results(raw_results)

    # 3. Mostrar tabla
    print("\n── Resultados (promedios sobre iteraciones) ──")
    print(df.to_string(index=False))

    # 4. Guardar CSV
    df.to_csv("resultados.csv", index=False)
    print("\nCSV guardado: resultados.csv")

    # 5. Generar gráficas
    plot_all(df)
    print("Gráficas guardadas en PNG.")
