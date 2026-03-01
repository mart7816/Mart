# fuzzy_time_window_penalty.py
# Trapezoidal fuzzy time window penalty function with priority weighting

def fuzzy_time_window_penalty(
    arrival_time: float,
    tw: tuple[float, float, float, float],  # (a1: very early, a2: early, b2: late, b1: very late)
    priority: int = 1,                       # 1–5 (higher = more penalty weight)
    early_penalty_factor: float = 80.0,      # cost per hour very early
    mild_early_factor: float = 30.0,
    mild_late_factor: float = 50.0,
    late_penalty_factor: float = 120.0       # cost per hour very late
) -> float:
    """
    Compute penalty for violating trapezoidal fuzzy time window.
    
    Fuzzy window structure:
      a1 ───── a2 ───────────── b2 ───── b1
      |<-- very early -->|<-- on time -->|<-- very late -->|
    
    Returns penalty cost (USD or arbitrary unit)
    """
    a1, a2, b2, b1 = tw
    
    if arrival_time < a1:
        # Very early
        return (a1 - arrival_time) * early_penalty_factor * priority
    
    elif arrival_time < a2:
        # Mild early
        return (a2 - arrival_time) * mild_early_factor * priority
    
    elif arrival_time <= b2:
        # On time → no penalty
        return 0.0
    
    elif arrival_time <= b1:
        # Mild late
        return (arrival_time - b2) * mild_late_factor * priority
    
    else:
        # Very late
        return (arrival_time - b1) * late_penalty_factor * priority


# Example usage
if __name__ == "__main__":
    # Example fuzzy window: [a1=8, a2=9, b2=11, b1=12]
    tw_example = (8.0, 9.0, 11.0, 12.0)
    
    print(f"Arrival 7.5h (very early): {fuzzy_time_window_penalty(7.5, tw_example, priority=4):.2f}")
    print(f"Arrival 8.5h (mild early): {fuzzy_time_window_penalty(8.5, tw_example, priority=4):.2f}")
    print(f"Arrival 10.0h (on time)  : {fuzzy_time_window_penalty(10.0, tw_example, priority=4):.2f}")
    print(f"Arrival 11.5h (mild late) : {fuzzy_time_window_penalty(11.5, tw_example, priority=4):.2f}")
    print(f"Arrival 13.0h (very late) : {fuzzy_time_window_penalty(13.0, tw_example, priority=4):.2f}")
