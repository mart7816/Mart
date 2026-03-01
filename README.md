# MOVRP – Multi-Objective Vehicle Routing for Sustainable Perishable Food Distribution

This repository contains the code for the paper:

**A Multi-Objective Vehicle Routing Model for Sustainable Perishable Food Distribution: Minimizing Cost and Emissions While Maximizing Freshness**

## Features
- Multi-objective optimization: cost, CO₂ emissions, freshness
- Simulated Annealing (primary solver)
- Genetic Algorithm (benchmark)
- Fuzzy time windows with priority-weighted penalties
- Arrhenius-Gompertz freshness decay model
- Weight sensitivity analysis
- Statistical validation (Wilcoxon signed-rank test, Hedges' g)

## Requirements
Python 3.9+
```bash
pip install -r requirements.txt
