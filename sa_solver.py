# sa_solver.py
# Full Simulated Annealing Solver for Multi-Objective Perishable VRP
# Minimizes Cost + Emissions, Maximizes Freshness (Arrhenius-Gompertz)
# Supports fuzzy time windows and customer priority levels

import numpy as np
import random
import copy
import math
from typing import List, Tuple
from freshness_model import arrhenius_gompertz_decay   # Import from your freshness_model.py


class SimulatedAnnealing:
    def __init__(self,
                 distance_matrix: np.ndarray,
                 demands: List[float],
                 fuzzy_time_windows: List[Tuple[float, float, float, float]],  # [a1, a2, b2, b1]
                 service_times: List[float],
                 vehicle_capacity: float,
                 priority_levels: List[int],          # 1-5 (higher = more important)
                 num_vehicles: int,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 15000):
        
        self.dist = distance_matrix
        self.demands = demands
        self.tw = fuzzy_time_windows
        self.service = service_times
        self.Q = vehicle_capacity
        self.priority = priority_levels
        self.n_vehicles = num_vehicles
        self.n_customers = len(demands) - 1  # exclude depot
        
        self.T0 = initial_temp
        self.alpha = cooling_rate
        self.max_iter = max_iterations
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_freshness = 0.0
        self.best_cost = 0.0
        self.best_emissions = 0.0

    # ====================== Fuzzy Time Window Penalty ======================
    def fuzzy_penalty(self, arrival_time: float, tw: Tuple[float, float, float, float], priority: int) -> float:
        a1, a2, b2, b1 = tw
        penalty = 0.0
        if arrival_time < a1:
            penalty = (a1 - arrival_time) * priority * 80.0   # early penalty
        elif arrival_time > b1:
            penalty = (arrival_time - b1) * priority * 120.0  # late penalty
        elif arrival_time < a2:
            penalty = (a2 - arrival_time) * priority * 30.0   # mild early
        elif arrival_time > b2:
            penalty = (arrival_time - b2) * priority * 50.0   # mild late
        return penalty

    # ====================== Fitness Function ======================
    def evaluate_fitness(self, routes: List[List[int]]) -> Tuple[float, float, float, float]:
        """Returns (scalarized_fitness, total_cost, total_emissions, avg_freshness)"""
        total_cost = 0.0
        total_emissions = 0.0
        total_freshness = 0.0
        total_priority_weight = 0.0

        for route in routes:
            if not route:
                continue
            load = sum(self.demands[i] for i in route)
            if load > self.Q:
                return float('inf'), 0, 0, 0  # infeasible

            route_cost = 0.0
            route_emissions = 0.0
            arrival_time = 0.0
            prev = 0  # depot

            for cust in route:
                dist = self.dist[prev][cust]
                route_cost += dist * 1.38          # variable cost per km (refrigerated)
                route_emissions += dist * 0.25     # kg CO2 per km (example factor)

                arrival_time += dist / 60.0        # assume 60 km/h
                penalty = self.fuzzy_penalty(arrival_time, self.tw[cust], self.priority[cust])
                route_cost += penalty

                # Freshness using Arrhenius-Gompertz
                freshness = arrhenius_gompertz_decay(arrival_time, temperature=5.0)
                weight = self.priority[cust]
                total_freshness += freshness * weight
                total_priority_weight += weight

                arrival_time += self.service[cust]
                prev = cust

            # Return to depot
            route_cost += self.dist[prev][0] * 1.38
            route_emissions += self.dist[prev][0] * 0.25
            total_cost += route_cost
            total_emissions += route_emissions

        avg_freshness = total_freshness / max(total_priority_weight, 1.0)

        # Scalarized fitness (lower is better)
        w_cost = 0.35
        w_emis = 0.35
        w_fresh = 0.30
        scalarized = w_cost * total_cost + w_emis * total_emissions - w_fresh * avg_freshness

        return scalarized, total_cost, total_emissions, avg_freshness

    # ====================== Neighborhood Operators ======================
    def generate_neighbor(self, solution: List[List[int]]) -> List[List[int]]:
        neighbor = copy.deepcopy(solution)
        route_idx = random.randint(0, len(neighbor)-1)
        route = neighbor[route_idx]

        if len(route) < 2:
            return neighbor

        op = random.choice(['2opt', 'relocate', 'exchange'])
        
        if op == '2opt' and len(route) >= 3:
            i = random.randint(0, len(route)-3)
            j = random.randint(i+2, len(route)-1)
            route[i:j+1] = reversed(route[i:j+1])
            
        elif op == 'relocate' and len(route) >= 2:
            i = random.randint(0, len(route)-1)
            cust = route.pop(i)
            # Try to insert into same or different route
            if random.random() < 0.6 and len(neighbor) > 1:
                new_route = random.choice([r for idx, r in enumerate(neighbor) if idx != route_idx])
                pos = random.randint(0, len(new_route))
                new_route.insert(pos, cust)
            else:
                pos = random.randint(0, len(route))
                route.insert(pos, cust)
                
        elif op == 'exchange' and len(route) >= 2:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

        return neighbor

    # ====================== Main SA Algorithm ======================
    def run(self, initial_solution: List[List[int]] = None):
        if initial_solution is None:
            # Greedy initial solution (cheapest insertion)
            initial_solution = self.generate_initial_solution()

        current_solution = initial_solution
        current_fitness, *_ = self.evaluate_fitness(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_fitness = current_fitness
        best_cost = best_emissions = best_freshness = 0

        T = self.T0
        stagnation = 0

        for iteration in range(self.max_iter):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness, cost, emissions, freshness = self.evaluate_fitness(neighbor)

            delta = neighbor_fitness - current_fitness

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if neighbor_fitness < best_fitness:
                    best_solution = copy.deepcopy(neighbor)
                    best_fitness = neighbor_fitness
                    best_cost = cost
                    best_emissions = emissions
                    best_freshness = freshness
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            T *= self.alpha

            if stagnation > 2000:  # early stopping
                break

        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.best_cost = best_cost
        self.best_emissions = best_emissions
        self.best_freshness = best_freshness

        return best_solution, best_fitness, best_cost, best_emissions, best_freshness

    # ====================== Initial Solution ======================
    def generate_initial_solution(self):
        # Simple greedy insertion (can be improved)
        customers = list(range(1, self.n_customers + 1))
        random.shuffle(customers)
        routes = [[] for _ in range(self.n_vehicles)]
        for cust in customers:
            best_route = 0
            best_increase = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    temp_route = route[:pos] + [cust] + route[pos:]
                    fitness, *_ = self.evaluate_fitness([temp_route])
                    if fitness < best_increase:
                        best_increase = fitness
                        best_route = r_idx
            routes[best_route].append(cust)
        return [r for r in routes if r]  # remove empty routes


# ====================== Example Usage ======================
if __name__ == "__main__":
    print("SA Solver loaded. Ready to integrate with distance matrix and Solomon data.")
    # Example:
    # sa = SimulatedAnnealing(distance_matrix, demands, fuzzy_tw, service_times, Q=200, ...)
    # best_route, fitness, cost, emissions, freshness = sa.run()
