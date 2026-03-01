# freshness_model.py
# Arrhenius-Gompertz based freshness decay model for perishable food
# Used in multi-objective VRP to compute freshness at delivery

import math


def arrhenius_gompertz_decay(
    time_delivered: float,
    temperature: float,
    base_shelf_life: float = 1450.0,      # hours at reference temperature
    reference_temp: float = 4.0,           # °C (refrigerated)
    activation_energy: float = 85000.0,    # J/mol (typical 80–100 kJ/mol)
    gas_constant: float = 8.314,           # J/(mol·K)
    initial_freshness: float = 100.0       # starting freshness (%)
) -> float:
    """
    Compute remaining freshness (%) using simplified Arrhenius-Gompertz kinetics.
    
    Parameters:
    -----------
    time_delivered : float
        Cumulative delivery time (hours) from depot to customer
    temperature : float
        Average compartment temperature during transit (°C)
    base_shelf_life : float
        Shelf life at reference temperature (hours)
    reference_temp : float
        Reference refrigerated temperature (°C)
    activation_energy : float
        Activation energy for spoilage reaction (J/mol)
    gas_constant : float
        Universal gas constant (J/(mol·K))
    initial_freshness : float
        Initial freshness level (%)

    Returns:
    --------
    freshness : float
        Remaining freshness (%) at delivery (0–100)
    """
    # Convert temperatures to Kelvin
    T_kelvin = temperature + 273.15
    T_ref_kelvin = reference_temp + 273.15

    # Temperature-dependent decay rate (Arrhenius law)
    # k = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
    k_ref = 1.0 / base_shelf_life
    k = k_ref * math.exp(
        -activation_energy / gas_constant * (1 / T_kelvin - 1 / T_ref_kelvin)
    )

    # Simplified Gompertz-like exponential decay for freshness
    # freshness = initial * exp(-k * t)
    # (full Gompertz would use microbial count, but this is common simplification for shelf-life)
    freshness = initial_freshness * math.exp(-k * time_delivered)

    return max(0.0, freshness)


def freshness_aggregate(
    routes: list[list[int]],
    arrival_times: list[list[float]],
    temperatures: list[float],
    priority_levels: list[int],
    base_shelf_life: float = 1450.0,
    reference_temp: float = 4.0,
    activation_energy: float = 85000.0
) -> float:
    """
    Compute weighted average freshness across all delivered customers.
    
    Higher priority customers contribute more to the aggregate score.
    
    Parameters:
    -----------
    routes : list[list[int]]
        List of routes (each route is a list of customer indices)
    arrival_times : list[list[float]]
        Arrival time at each customer in each route (hours)
    temperatures : list[float]
        Average transit temperature for each customer (°C)
    priority_levels : list[int]
        Priority weight (1–5) for each customer (index 0 = depot, usually 0)
    ... (other Arrhenius-Gompertz params)

    Returns:
    --------
    avg_freshness : float
        Priority-weighted average freshness (%)
    """
    total_freshness = 0.0
    total_weight = 0.0

    for route_idx, route in enumerate(routes):
        if not route:
            continue
        for cust_idx, cust in enumerate(route):
            t_del = arrival_times[route_idx][cust_idx]
            temp = temperatures[cust] if isinstance(temperatures, list) else temperatures
            fresh = arrhenius_gompertz_decay(
                t_del, temp, base_shelf_life, reference_temp, activation_energy
            )
            weight = priority_levels[cust] if cust < len(priority_levels) else 1.0
            total_freshness += fresh * weight
            total_weight += weight

    return total_freshness / max(total_weight, 1e-6)  # avoid division by zero


# Example usage / testing
if __name__ == "__main__":
    # Test single customer
    t = 12.5          # hours to delivery
    temp = 5.0        # °C
    fresh = arrhenius_gompertz_decay(t, temp)
    print(f"Freshness after {t} h at {temp}°C: {fresh:.2f}%")

    # Test aggregate (mock routes)
    mock_routes = [[1, 2, 3], [4, 5]]
    mock_arrival_times = [[2.5, 5.0, 9.0], [3.0, 7.5]]
    mock_temps = [4.5] * 5
    mock_priorities = [0, 5, 3, 4, 2, 1]  # depot=0, customers 1-5

    avg_f = freshness_aggregate(mock_routes, mock_arrival_times, mock_temps, mock_priorities)
    print(f"Priority-weighted average freshness: {avg_f:.2f}%")
