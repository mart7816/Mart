# MOVRP Perishable Food Distribution Code

This repository contains the Python implementation for the paper:

"A Multi-Objective Vehicle Routing Model for Sustainable Perishable Food Distribution: Minimizing Cost and Emissions While Maximizing Freshness"

## Contents
- `sa_solver.py`: Simulated Annealing primary solver
- `ga_benchmark.py`: Genetic Algorithm benchmark
- `freshness_model.py`: Arrhenius-Gompertz freshness decay
- `sensitivity_analysis.py`: Weight sensitivity grid search
- `utils.py`: Helper functions (scalarization, fuzzy windows, priority penalties)

## Requirements
Python 3.12+
See `requirements.txt`

## How to run
```bash
pip install -r requirements.txt
python src/sa_solver.py --instance solomon_r101.txt --weights 0.33 0.33 0.34# Mart
