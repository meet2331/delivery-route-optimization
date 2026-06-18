"""
Flask REST API — Bonn Route Optimizer
Exposes the LP optimization model as an API endpoint
so business users can query optimal routes with custom demand.

Usage:
    python api.py
    
    POST /optimize
    Body: {"zone_demand": {"53111": 250, "53113": 180, ...}}
    
    GET /zones     → returns all available zones and baseline demand
    GET /routes    → returns all available routes and their properties
    GET /health    → health check
"""

from flask import Flask, request, jsonify
from optimize import (
    run_optimization,
    zone_demand,
    zone_names,
    bonn_zones,
    routes,
    route_cost,
    route_capacity,
    routes_data
)
import traceback

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model': 'Bonn Delivery Route Optimizer',
        'solver': 'PuLP CBC',
        'version': '1.0'
    })


@app.route('/zones', methods=['GET'])
def get_zones():
    """Return all Bonn PLZ zones with baseline demand."""
    return jsonify({
        'zones': {
            plz: {
                'name': info[0],
                'population': info[1],
                'type': info[2],
                'baseline_demand': zone_demand[plz]
            }
            for plz, info in bonn_zones.items()
        },
        'total_baseline_demand': sum(zone_demand.values())
    })


@app.route('/routes', methods=['GET'])
def get_routes():
    """Return all available routes and their properties."""
    return jsonify({
        'routes': {
            r: {
                'cost_eur': route_cost[r],
                'capacity_packages': route_capacity[r],
                'description': routes_data[r]['description'],
                'serves_zones': routes_data[r]['serves']
            }
            for r in routes
        }
    })


@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Run route optimization with custom or baseline demand.
    
    Request body (optional):
    {
        "zone_demand": {
            "53111": 250,
            "53113": 180,
            ...
        },
        "demand_multiplier": 1.2   // alternative: scale all zones by factor
    }
    
    If no body provided, uses baseline demand.
    """
    try:
        data = request.get_json(silent=True) or {}

        # Option 1: Custom demand per zone
        if 'zone_demand' in data:
            custom_demand = data['zone_demand']
            # Fill missing zones with baseline
            demand_input = {
                z: custom_demand.get(z, zone_demand[z])
                for z in zone_demand.keys()
            }

        # Option 2: Scale all zones by a multiplier
        elif 'demand_multiplier' in data:
            m = float(data['demand_multiplier'])
            if not 0.1 <= m <= 5.0:
                return jsonify({'error': 'demand_multiplier must be between 0.1 and 5.0'}), 400
            demand_input = {z: round(d * m) for z, d in zone_demand.items()}

        # Default: baseline demand
        else:
            demand_input = zone_demand

        # Run optimization
        results = run_optimization(zone_demand_input=demand_input, verbose=False)

        if results['status'] != 'Optimal':
            return jsonify({
                'status': results['status'],
                'message': 'No optimal solution found. '
                           'Try reducing demand or check zone coverage.'
            }), 422

        # Format response
        total_delivered = sum(
            v['delivered'] for v in results['zone_assignments'].values()
        )
        total_demand_val = sum(demand_input.values())

        response = {
            'status': 'Optimal',
            'summary': {
                'total_cost_eur': round(results['total_cost'], 2),
                'routes_activated': len(results['active_routes']),
                'total_routes_available': len(routes),
                'packages_demanded': total_demand_val,
                'packages_delivered': round(total_delivered),
                'coverage_pct': round(total_delivered / total_demand_val * 100, 1),
                'cost_per_package_eur': round(
                    results['total_cost'] / total_delivered, 4
                ) if total_delivered > 0 else None
            },
            'active_routes': {
                r: {
                    'load': round(info['load']),
                    'capacity': info['capacity'],
                    'utilization_pct': info['utilization_pct'],
                    'cost_eur': info['cost'],
                    'unused_capacity': round(info['slack'])
                }
                for r, info in results['active_routes'].items()
            },
            'zone_delivery': {
                z: {
                    'name': info['name'],
                    'demand': info['demand'],
                    'delivered': round(info['delivered']),
                    'served_by': info['routes']
                }
                for z, info in results['zone_assignments'].items()
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("\nBonn Route Optimizer API")
    print("Endpoints:")
    print("  GET  /health    → API health check")
    print("  GET  /zones     → Available PLZ zones")
    print("  GET  /routes    → Available delivery routes")
    print("  POST /optimize  → Run optimization\n")
    app.run(debug=True, port=5000)
