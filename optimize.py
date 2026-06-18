"""
Delivery Route Optimization for Bonn, Germany
Using Linear Programming (PuLP) with real Bonn postal district data

Author: Meet Pala
Problem: A DHL Express depot in Bonn must decide which routes to operate 
         daily to minimize operating costs while meeting all package demand 
         across Bonn's postal districts (PLZ zones).
"""

from pulp import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# ============================================================
# REAL DATA: Bonn Postal Districts (PLZ)
# Source: Statistisches Amt Bonn / Deutsche Post PLZ system
# Population: Bundesstadt Bonn Statistikstelle (2020)
# ============================================================

bonn_zones = {
    # PLZ   : (District Name,           Population,  Zone Type)
    '53111': ('Bonn-Zentrum',            15200,       'commercial'),
    '53113': ('Südliche Innenstadt',     12800,       'mixed'),
    '53115': ('Poppelsdorf/Weststadt',   18500,       'residential'),
    '53117': ('Nordstadt/Castell',       14300,       'residential'),
    '53119': ('Tannenbusch/Auerberg',    22100,       'residential'),
    '53121': ('Endenich',                13600,       'mixed'),
    '53123': ('Duisdorf/Medinghoven',    21400,       'residential'),
    '53125': ('Brüser Berg',             16900,       'residential'),
    '53129': ('Kessenich/Dottendorf',    17800,       'mixed'),
    '53173': ('Bad Godesberg-Zentrum',   19200,       'commercial'),
    '53175': ('Rüngsdorf/Plittersdorf',  14600,       'mixed'),
    '53177': ('Mehlem/Friesdorf',        11800,       'residential'),
    '53225': ('Beuel-Pützchen',          13400,       'residential'),
    '53227': ('Beuel-Geislar/Holzlar',   10900,       'residential'),
    '53229': ('Beuel-Zentrum',           18700,       'commercial'),
}

# ============================================================
# DEMAND CALCULATION
# We model packages per 1000 residents per day for each zone type.
# This represents the daily load for the Bonn regional depot.
# Commercial: ~12 pkgs/1000 residents (high B2B density)
# Mixed:      ~8  pkgs/1000 residents
# Residential:~5  pkgs/1000 residents (standard B2C)
# ============================================================

DEMAND_RATES = {'commercial': 12, 'mixed': 8, 'residential': 5}

def calculate_demand(population, zone_type):
    return max(10, round(population / 1000 * DEMAND_RATES[zone_type]))

zones = list(bonn_zones.keys())
zone_names = {plz: info[0] for plz, info in bonn_zones.items()}
zone_demand = {
    plz: calculate_demand(info[1], info[2])
    for plz, info in bonn_zones.items()
}

# Step 2: Override specific zones for scenario testing
# (Comment/uncomment lines to test different scenarios)

# zone_demand['53177'] = 250   # Mehlem: surge from 59 to 250
# # zone_demand['53173'] = 300   # Bad Godesberg: surge from 230 to 300
# # zone_demand['53111'] = 250   # City center: surge from 182 to 250

# # Step 3: Print demands for verification
# print("\nZone demands for this run:")
# for plz in sorted(zone_demand.keys()):
#     print(f"  {plz} ({zone_names[plz][:25]}): {zone_demand[plz]} packages")


# ============================================================
# DELIVERY ROUTES FROM BONN DEPOT
# Depot: DHL Express Bonn (~53121 Endenich area)
# Routes cluster geographically adjacent PLZ zones.
# Costs (EUR) = fixed vehicle + driver costs per route per day
# Capacities = max packages per vehicle on that route
# ============================================================

routes_data = {
    'Route_North': {
        'cost': 145, 'capacity': 250,
        'description': 'Nordstadt, Tannenbusch, Auerberg',
        'serves': ['53117', '53119'],
    },
    'Route_City_Core': {
        'cost': 120, 'capacity': 200,
        'description': 'City Centre + Südliche Innenstadt (high density)',
        'serves': ['53111', '53113'],
    },
    'Route_West': {
        'cost': 155, 'capacity': 280,
        'description': 'Poppelsdorf, Weststadt, Endenich',
        'serves': ['53115', '53121'],
    },
    'Route_South_West': {
        'cost': 160, 'capacity': 270,
        'description': 'Kessenich, Dottendorf',
        'serves': ['53129'],
    },
    'Route_Hardtberg': {
        'cost': 175, 'capacity': 300,
        'description': 'Duisdorf, Medinghoven, Brüser Berg',
        'serves': ['53123', '53125'],
    },
    'Route_Godesberg_Nord': {
        'cost': 185, 'capacity': 260,
        'description': 'Bad Godesberg Zentrum + Rüngsdorf',
        'serves': ['53173', '53175'],
    },
    'Route_Godesberg_Sud': {
        'cost': 170, 'capacity': 220,
        'description': 'Mehlem, Friesdorf',
        'serves': ['53177'],
    },
    'Route_Beuel': {
        'cost': 165, 'capacity': 290,
        'description': 'Beuel Zentrum, Pützchen, Geislar (Rhine crossing)',
        'serves': ['53225', '53227', '53229'],
    },
    'Route_Express_City': {
        'cost': 200, 'capacity': 160,
        'description': 'Priority express loop: City + Godesberg B2B',
        'serves': ['53111', '53113', '53173'],
    },
    'Route_South_Extended': {
        'cost': 215, 'capacity': 340,
        'description': 'Full south coverage: Godesberg + Beuel south',
        'serves': ['53173', '53175', '53177', '53229'],
    },
}

routes = list(routes_data.keys())
route_cost     = {r: routes_data[r]['cost']     for r in routes}
route_capacity = {r: routes_data[r]['capacity'] for r in routes}

route_zone_eligible = {
    (r, z): (1 if z in routes_data[r]['serves'] else 0)
    for r in routes for z in zones
}

# ============================================================
# LP MODEL
# ============================================================

def run_optimization(zone_demand_input=None, verbose=True):
    if zone_demand_input is None:
        zone_demand_input = zone_demand

    prob = LpProblem("Bonn_Route_Optimization", LpMinimize)

    route_active = LpVariable.dicts(
        "Active", routes, cat='Binary'
    )
    packages = LpVariable.dicts(
        "Pkgs",
        [(r, z) for r in routes for z in zones],
        lowBound=0, cat='Continuous'
    )

    # Objective
    prob += lpSum(route_cost[r] * route_active[r] for r in routes)

    # C1: Meet all zone demand
    for z in zones:
        prob += lpSum(packages[(r, z)] for r in routes) == zone_demand_input[z], f"D_{z}"

    # C2: Respect route capacity
    for r in routes:
        prob += lpSum(packages[(r, z)] for z in zones) <= \
               route_capacity[r] * route_active[r], f"Cap_{r}"

    # C3: Eligibility
    for r in routes:
        for z in zones:
            if route_zone_eligible[(r, z)] == 0:
                prob += packages[(r, z)] == 0, f"Elig_{r}_{z}"

    # C4: Minimum load if active
    for r in routes:
        prob += lpSum(packages[(r, z)] for z in zones) >= \
               5 * route_active[r], f"Min_{r}"

    prob.solve(PULP_CBC_CMD(msg=0))

    results = {
        'status': LpStatus[prob.status],
        'total_cost': value(prob.objective) if value(prob.objective) else 0,
        'active_routes': {},
        'zone_assignments': {}
    }

    if LpStatus[prob.status] == 'Optimal':
        for r in routes:
            if value(route_active[r]) and value(route_active[r]) > 0.5:
                load = sum(value(packages[(r, z)]) or 0 for z in zones)
                results['active_routes'][r] = {
                    'load': load,
                    'capacity': route_capacity[r],
                    'utilization_pct': round(load / route_capacity[r] * 100, 1),
                    'cost': route_cost[r],
                    'slack': route_capacity[r] - load
                }

    for z in zones:
        delivered = sum(
            (value(packages[(r, z)]) or 0) for r in routes
        )
        serving = [
            r for r in routes
            if (value(packages[(r, z)]) or 0) > 0.01
        ]
        results['zone_assignments'][z] = {
            'name': zone_names[z],
            'demand': zone_demand_input[z],
            'delivered': delivered,
            'routes': serving
        }

    if verbose:
        _print_results(results)

    return results


def _print_results(r):
    print(f"\n{'='*62}")
    print(f"  BONN ROUTE OPTIMIZATION — RESULTS")
    print(f"{'='*62}")
    print(f"  Status:            {r['status']}")
    print(f"  Total Daily Cost:  €{r['total_cost']:.2f}")
    print(f"  Routes Active:     {len(r['active_routes'])} of {len(routes)}")
    total_d = sum(v['demand']    for v in r['zone_assignments'].values())
    total_l = sum(v['delivered'] for v in r['zone_assignments'].values())
    print(f"  Packages Demanded: {total_d}")
    print(f"  Packages Covered:  {total_l:.0f}  ({total_l/total_d*100:.1f}%)")
    print(f"{'='*62}\n")

    if r['active_routes']:
        print(f"  {'Route':<22} {'Load':>6} {'Cap':>6} {'Util':>7} {'Cost':>7} {'Slack':>7}")
        print(f"  {'-'*57}")
        for name, info in r['active_routes'].items():
            short = name.replace('Route_','')
            print(f"  {short:<22} {info['load']:>6.0f} {info['capacity']:>6} "
                  f"{info['utilization_pct']:>6.1f}% €{info['cost']:>5} "
                  f"{info['slack']:>6.0f}")

    print(f"\n  {'PLZ':<7} {'District':<28} {'Demand':>8} {'Delivered':>10}")
    print(f"  {'-'*56}")
    for z, info in r['zone_assignments'].items():
        print(f"  {z:<7} {info['name']:<28} {info['demand']:>8} "
              f"{info['delivered']:>10.0f}")


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    if not results['active_routes']:
        print("No optimal solution — skipping charts.")
        return None

    active  = results['active_routes']
    rlabels = [r.replace('Route_', '') for r in active]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'Bonn Delivery Route Optimization — Dashboard\n'
        'Real PLZ Districts | DHL Express Scenario | PuLP LP Solver',
        fontsize=14, fontweight='bold', y=0.99
    )

    # 1 — Utilization
    ax1 = fig.add_subplot(2, 3, 1)
    utils = [i['utilization_pct'] for i in active.values()]
    cols  = ['#d9534f' if u > 85 else '#f0ad4e' if u > 60 else '#5cb85c' for u in utils]
    bars  = ax1.bar(rlabels, utils, color=cols, edgecolor='white')
    ax1.axhline(85, color='red',    linestyle='--', lw=1.2, label='85% warning')
    ax1.axhline(60, color='orange', linestyle='--', lw=1.2, label='60% target')
    ax1.set_title('Route Utilization (%)', fontweight='bold')
    ax1.set_ylabel('Utilization %'); ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8); ax1.tick_params(axis='x', rotation=35)
    for b, v in zip(bars, utils):
        ax1.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%',
                 ha='center', va='bottom', fontsize=8)

    # 2 — Cost per route
    ax2 = fig.add_subplot(2, 3, 2)
    costs = [i['cost'] for i in active.values()]
    ax2.bar(rlabels, costs, color='steelblue', edgecolor='white')
    ax2.set_title('Operating Cost per Route (€)', fontweight='bold')
    ax2.set_ylabel('Cost (€)'); ax2.tick_params(axis='x', rotation=35)
    for i, c in enumerate(costs):
        ax2.text(i, c+1, f'€{c}', ha='center', va='bottom', fontsize=8)

    # 3 — Zone demand vs delivered
    ax3 = fig.add_subplot(2, 3, 3)
    zlabels   = [v['name'].split('/')[0][:10] for v in results['zone_assignments'].values()]
    demands   = [v['demand']    for v in results['zone_assignments'].values()]
    delivered = [v['delivered'] for v in results['zone_assignments'].values()]
    x = np.arange(len(zlabels)); w = 0.38
    ax3.bar(x-w/2, demands,   w, label='Demand',    color='#5bc0de')
    ax3.bar(x+w/2, delivered, w, label='Delivered', color='#5cb85c', alpha=0.85)
    ax3.set_title('Demand vs Delivered by Zone', fontweight='bold')
    ax3.set_ylabel('Packages')
    ax3.set_xticks(x); ax3.set_xticklabels(zlabels, rotation=45, ha='right', fontsize=7)
    ax3.legend()

    # 4 — Slack (unused capacity)
    ax4 = fig.add_subplot(2, 3, 4)
    slacks    = [i['slack'] for i in active.values()]
    slack_pct = [s/route_capacity[r]*100 for r, s in zip(active.keys(), slacks)]
    scols     = ['#d9534f' if p > 30 else '#f0ad4e' if p > 15 else '#5cb85c'
                 for p in slack_pct]
    ax4.barh(rlabels, slacks, color=scols, edgecolor='white')
    ax4.set_title('Unused Capacity (packages)', fontweight='bold')
    ax4.set_xlabel('Unused Packages')
    for i, (s, p) in enumerate(zip(slacks, slack_pct)):
        ax4.text(s+0.2, i, f'{s:.0f} ({p:.0f}%)', va='center', fontsize=8)

    # 5 — Cost efficiency
    ax5 = fig.add_subplot(2, 3, 5)
    cpp = [i['cost']/i['load'] if i['load'] > 0 else 0 for i in active.values()]
    b5  = ax5.bar(rlabels, cpp, color='mediumpurple', edgecolor='white')
    ax5.set_title('Cost Efficiency (€ per Package)', fontweight='bold')
    ax5.set_ylabel('€ / Package'); ax5.tick_params(axis='x', rotation=35)
    for b, v in zip(b5, cpp):
        ax5.text(b.get_x()+b.get_width()/2, v+0.002, f'€{v:.2f}',
                 ha='center', va='bottom', fontsize=8)

    # 6 — KPI summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    tot_d   = sum(v['demand']    for v in results['zone_assignments'].values())
    tot_l   = sum(v['delivered'] for v in results['zone_assignments'].values())
    avg_u   = np.mean(utils)
    tot_s   = sum(slacks)
    cpp_tot = results['total_cost'] / tot_l if tot_l > 0 else 0

    kpis = [
        ('Status',            results['status'],             '#5cb85c'),
        ('Total Daily Cost',  f"€{results['total_cost']:.2f}", '#d9534f'),
        ('Routes Activated',  f"{len(active)} / {len(routes)}", '#5bc0de'),
        ('Total Packages',    f"{tot_l:.0f} / {tot_d}",      '#5cb85c'),
        ('Avg Utilization',   f"{avg_u:.1f}%",               '#f0ad4e'),
        ('Total Slack',       f"{tot_s:.0f} packages",       '#d9534f'),
        ('Cost per Package',  f"€{cpp_tot:.3f}",             '#9b59b6'),
        ('Coverage Rate',     f"{tot_l/tot_d*100:.1f}%",     '#5cb85c'),
    ]
    ax6.set_title('Summary KPIs', fontweight='bold', pad=10)
    for i, (label, val, color) in enumerate(kpis):
        y = 0.92 - i * 0.115
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.02, y-0.045), 0.96, 0.095,
            boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.12, edgecolor=color
        ))
        ax6.text(0.05, y+0.002, label,     fontsize=9, va='center', color='#333')
        ax6.text(0.97, y+0.002, val,       fontsize=9, va='center',
                 ha='right', fontweight='bold', color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'optimization_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] Dashboard  → {path}")
    return path


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity_analysis(output_dir='results'):
    print("\nRunning sensitivity analysis...")
    os.makedirs(output_dir, exist_ok=True)

    multipliers = np.round(np.arange(0.70, 1.55, 0.05), 2)
    rows = []

    for m in multipliers:
        scaled = {z: max(1, round(d * m)) for z, d in zone_demand.items()}
        res    = run_optimization(zone_demand_input=scaled, verbose=False)
        if res['status'] == 'Optimal' and res['active_routes']:
            tot_del  = sum(v['delivered'] for v in res['zone_assignments'].values())
            avg_util = np.mean([i['utilization_pct']
                                for i in res['active_routes'].values()])
            rows.append({
                'multiplier':       round(m, 2),
                'demand_pct_change': round((m - 1) * 100),
                'total_cost':       res['total_cost'],
                'routes_active':    len(res['active_routes']),
                'packages_delivered': tot_del,
                'avg_utilization':  avg_util
            })

    if not rows:
        print("  No optimal solutions found in sensitivity range.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Sensitivity Analysis — Impact of Demand Variation\n'
        'Bonn Delivery Route Optimizer',
        fontsize=13, fontweight='bold'
    )

    axes[0].plot(df['demand_pct_change'], df['total_cost'], 'b-o', lw=2, ms=6)
    axes[0].axvline(0, color='red', linestyle='--', lw=1, label='Baseline')
    axes[0].fill_between(df['demand_pct_change'], df['total_cost'], alpha=0.12, color='blue')
    axes[0].set_title('Total Cost vs Demand', fontweight='bold')
    axes[0].set_xlabel('Demand Change (%)'); axes[0].set_ylabel('Total Cost (€)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].step(df['demand_pct_change'], df['routes_active'],
                 'g-o', lw=2, ms=6, where='mid')
    axes[1].axvline(0, color='red', linestyle='--', lw=1, label='Baseline')
    axes[1].set_title('Routes Activated vs Demand', fontweight='bold')
    axes[1].set_xlabel('Demand Change (%)'); axes[1].set_ylabel('# Routes')
    axes[1].set_yticks(range(0, len(routes)+1))
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['demand_pct_change'], df['avg_utilization'],
                 color='purple', marker='s', lw=2, ms=6)
    axes[2].axvline(0,  color='red',    linestyle='--', lw=1, label='Baseline')
    axes[2].axhline(85, color='orange', linestyle=':',  lw=1.5, label='85% warning')
    axes[2].set_title('Avg Fleet Utilization vs Demand', fontweight='bold')
    axes[2].set_xlabel('Demand Change (%)'); axes[2].set_ylabel('Avg Utilization (%)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    spath = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(spath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Sensitivity → {spath}")

    cpath = os.path.join(output_dir, 'sensitivity_data.csv')
    df.to_csv(cpath, index=False)
    print(f"[Saved] CSV data    → {cpath}")

    base_cost = df[df['demand_pct_change'] == 0]['total_cost']
    high_cost = df[df['demand_pct_change'] == 50]['total_cost']
    if not base_cost.empty and not high_cost.empty:
        pct = (high_cost.values[0] - base_cost.values[0]) / base_cost.values[0] * 100
        print(f"\n  INSIGHT: +50% demand surge → cost increases by {pct:.1f}%  "
              f"(€{base_cost.values[0]:.0f} → €{high_cost.values[0]:.0f})")
    return df


# ============================================================
# JSON EXPORT
# ============================================================

def export_results(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'status':           results['status'],
        'total_cost_eur':   results['total_cost'],
        'routes_activated': len(results['active_routes']),
        'routes': {
            r: {
                'load':             round(i['load']),
                'capacity':         i['capacity'],
                'utilization_pct':  i['utilization_pct'],
                'cost_eur':         i['cost'],
                'unused_capacity':  round(i['slack'])
            } for r, i in results['active_routes'].items()
        },
        'zones': {
            z: {
                'name':      info['name'],
                'demand':    info['demand'],
                'delivered': round(info['delivered']),
                'served_by': info['routes']
            } for z, info in results['zone_assignments'].items()
        }
    }
    jpath = os.path.join(output_dir, 'optimization_results.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON        → {jpath}")
    return jpath


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\nBonn Delivery Route Optimizer")
    print("Real PLZ data | PuLP MILP | DHL Express scenario\n")

    print(f"{'PLZ':<7} {'District':<30} {'Pop':>7} {'Type':<14} {'Demand':>7}")
    print('-' * 65)
    total_pkgs = 0
    for plz, (name, pop, ztype) in bonn_zones.items():
        d = zone_demand[plz]
        print(f"{plz:<7} {name:<30} {pop:>7,} {ztype:<14} {d:>6} pkgs")
        total_pkgs += d
    print(f"\n  Total daily demand: {total_pkgs} packages across 15 PLZ zones\n")

    results = run_optimization()

    chart_path = create_visualizations(results)
    sens_df    = run_sensitivity_analysis()
    json_path  = export_results(results)

    print(f"\n{'='*62}")
    print("  All outputs saved to /results/")
    print(f"{'='*62}\n")
