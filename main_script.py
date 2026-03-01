# main_script.py
# Loads Solomon instance, runs SA + GA, computes statistics
# Requires: numpy, scipy, matplotlib, freshness_model.py, fuzzy_time_window_penalty.py, sa_solver.py, ga_benchmark.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind  # for approximate Hedges' g (or use custom function)
from typing import List, Tuple

# Import your modules (save them in the same folder)
from freshness_model import arrhenius_gompertz_decay, freshness_aggregate
from fuzzy_time_window_penalty import fuzzy_time_window_penalty
from sa_solver import SimulatedAnnealing
from ga_benchmark import GeneticAlgorithm  # assume you have this from earlier


def load_solomon_instance(filepath: str) -> dict:
    """
    Basic Solomon .txt loader (R/C/RC format)
    Returns dict with depot, customers, etc.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines until node section
    i = 0
    while not lines[i].strip().startswith("CUST NO."):
        i += 1
    i += 1  # skip header

    nodes = []
    for line in lines[i:]:
        if line.strip():
            parts = line.split()
            if len(parts) >= 7:
                cust_no = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready = float(parts[4])
                due = float(parts[5])
                service = float(parts[6])
                nodes.append({
                    'id': cust_no,
                    'x': x,
                    'y': y,
                    'demand': demand,
                    'ready': ready,
                    'due': due,
                    'service': service
                })

    depot = nodes[0]
    customers = nodes[1:]

    # Create distance matrix (Euclidean)
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes[i]['x'] - nodes[j]['x']
                dy = nodes[i]['y'] - nodes[j]['y']
                dist[i][j] = math.sqrt(dx*dx + dy*dy)

    # Fuzzy time windows (example: ±30 min tolerance)
    fuzzy_tw = []
    for node in nodes:
        ready, due = node['ready'], node['due']
        a1 = ready - 0.5
        a2 = ready
        b2 = due
        b1 = due + 0.5
        fuzzy_tw.append((a1, a2, b2, b1))

    return {
        'dist': dist,
        'demands': [n['demand'] for n in nodes],
        'fuzzy_tw': fuzzy_tw,
        'service': [n['service'] for n in nodes],
        'depot': depot,
        'customers': customers
    }


def run_experiment(instance_path: str, n_runs: int = 10):
    data = load_solomon_instance(instance_path)
    dist = data['dist']
    demands = data['demands']
    fuzzy_tw = data['fuzzy_tw']
    service = data['service']
    Q = 200  # example vehicle capacity
    priorities = [1] * len(demands)  # example: all priority 1
    priorities[0] = 0  # depot

    sa_fitnesses = []
    ga_fitnesses = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")

        # SA
        sa = SimulatedAnnealing(
            distance_matrix=dist,
            demands=demands,
            fuzzy_time_windows=fuzzy_tw,
            service_times=service,
            vehicle_capacity=Q,
            priority_levels=priorities,
            num_vehicles=25
        )
        _, sa_fit, _, _, _ = sa.run()
        sa_fitnesses.append(sa_fit)

        # GA (assume you have ga_benchmark.py)
        ga = GeneticAlgorithm(
            distance_matrix=dist,
            demands=demands,
            time_windows=fuzzy_tw,
            service_times=service,
            vehicle_capacity=Q,
            priority_levels=priorities,
            num_vehicles=25,
            max_generations=500,
            population_size=200
        )
        _, ga_fit = ga.run()
        ga_fitnesses.append(ga_fit)

    # Statistical comparison
    stat, p_value = wilcoxon(sa_fitnesses, ga_fitnesses)
    mean_sa = np.mean(sa_fitnesses)
    mean_ga = np.mean(ga_fitnesses)
    hedges = (mean_sa - mean_ga) / np.sqrt((np.var(sa_fitnesses) + np.var(ga_fitnesses)) / 2)  # approx

    print(f"Wilcoxon p-value: {p_value:.6f}")
    print(f"Mean SA fitness: {mean_sa:.2f}, Mean GA fitness: {mean_ga:.2f}")
    print(f"Approximate Hedges' g: {hedges:.2f}")

    return sa_fitnesses, ga_fitnesses


if __name__ == "__main__":
    # Replace with actual Solomon file path
    instance_file = "solomon_r101.txt"  # download from https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
    run_experiment(instance_file, n_runs=10)
