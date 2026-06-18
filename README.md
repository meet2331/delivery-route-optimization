# delivery-route-optimization
Mixed-integer linear program for delivery route optimization using PuLP, minimizing operating costs across a courier network serving Bonn postal districts

# Bonn Delivery Route Optimization

Mixed-integer linear program (MILP) to minimize daily operating costs for a courier network serving 15 postal districts in Bonn, Germany. Built with Python, PuLP, and Flask.

## Problem

A DHL Express depot must decide which delivery routes to operate daily to:
- Minimize total operating cost
- Meet all package demand across 15 zones
- Respect vehicle capacity limits
- Follow route-zone eligibility rules

## Model

| Component | Description |
|---|---|
| **Variables** | Binary route activation + continuous package allocation |
| **Objective** | Minimize sum of fixed route costs |
| **Constraints** | Demand satisfaction, capacity limits, eligibility rules, minimum load thresholds |
| **Solver** | CBC (via PuLP) with Branch-and-Bound |

## Results

- **Optimal cost:** €1,505/day
- **Routes activated:** 9 of 10
- **Packages delivered:** 1,751 (100% coverage)
- **Average utilization:** 72.3%

## Files

| File | Purpose |
|---|---|
| `optimize.py` | Core MILP model + visualization + sensitivity analysis |
| `api.py` | Flask REST API for dynamic re-optimization |
| `data/` | Input data (zones, routes, demand) |
| `results/` | Output charts, CSV, and JSON |

## Installation

```bash
pip install -r requirements.txt
