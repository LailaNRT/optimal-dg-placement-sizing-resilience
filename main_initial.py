import opendssdirect as dss
import os
import pulp
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import json
from datetime import datetime
import time
from collections import Counter

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

dss_file = "ieee37.dss"

# --- FINANCIALS (Global Constants) ---
COST_DG_FIXED        = 70250        # Interconnection, switchgear, concrete pad
COST_DG_PER_KW       = 500          # Capital cost per kW installed Utility-scale economies of scale
COST_DG_OM_PER_KW_YR = 9.33         # Annual O&M — oil, filters, wet-stack prevention

# Resilience cost during disaster
COST_GEN_PER_KWH     = 0.45          # Disaster logistics diesel pricing $/kWh — diesel fuel during blackout

# --- DYNAMIC VoLL TIERS ---
VOLL_TIER_1    = 3.3    # Residential (< 50 kW)
VOLL_TIER_2    = 21.8   # Commercial / Industrial (>= 50 kW)
VOLL_CRITICAL  = 100  # Critical infrastructure (bus-specific override) 
CRITICAL_BUSES = {'712', '729', '738', '720'}   # Bus 712: hospital/clinic; Bus 729: emergency services

# Seismic Event Timeframes
HOURS_NORMAL = 8760
MAX_INVESTMENT_BUDGET       = 1000000.0

# Q-dispatch within_limit tolerance -- matches Q_TOL_FRAC in checkPQbranch.py
# so the "within limit" flag written here doesn't flag trivial solver noise
# that checkPQbranch.py's own Q_LIMIT_ODS check wouldn't.
Q_LIMIT_TOL_FRAC = 0.010   # 1.0% of q_limit
# --- CONDITIONAL RESILIENCE MODELING ---
PLANNING_YEARS         = 15
INTEREST_RATE          = 0.1
PWF = ((1 + INTEREST_RATE)**PLANNING_YEARS - 1) / (INTEREST_RATE * (1 + INTEREST_RATE)**PLANNING_YEARS)

# Decouple the disaster from its low annual probability.
# Optimize the grid's survival GIVEN that the extreme hazard has occurred.
CONDITIONAL_EVENT_OCCURRENCE = 1

# For testing annual probability
# CONDITIONAL_EVENT_OCCURRENCE = 1 - (1 - 1/475)**PLANNING_YEARS  # ≈ 0.031

# ============================================================
# STANDALONE RUN CONFIG — edit only this block.
# When called by saa_convergence_study.py, sys.argv values
# automatically override these defaults.
# ============================================================
_CFG_ITERATION  = "3"      # Replicate index → loads saa_scenario_<LABEL>_<iter>.json
_CFG_LABEL      = ""       # File label (e.g. "pga030N30"). Blank → generic filenames.
_CFG_MIP_GAP    = 0.01     # Gurobi MIP gap: 0.005 = 0.5% | 0.01 = 1% | 0.05 = 5%
_CFG_LAMBDA     = 0.00       # Risk aversion λ: 0.0 = risk-neutral | 1.0 = pure CVaR
_CFG_ALPHA      = 0.90     # CVaR confidence level (tail = worst 1–α fraction)

LINE_CAPACITY = 500.0
TX_CAPACITY   = 500.0 / 3.0   # XFM1 rated 500 kVA total 3-phase → per-phase limit
sqrt3 = math.sqrt(3)

# Feeder base voltage for the LinDistFlow pu² normalization. The IEEE 37-bus
# test feeder is 4.8 kV L-L → 2.7713 kV L-N. LinDistFlow operates per-phase
# (single-phase equivalent), so the correct base is the LINE-TO-NEUTRAL voltage.
# Using L-L here overstated VDROP_DENOM by 3×, making voltage-drop constraints
# 3× too loose. VDROP_DENOM = V_base_LN[V]² / 1000 converts 2(rP+xQ) [kW·Ω] to pu².
V_BASE_LN_KV = 4.8 / math.sqrt(3)              # = 2.7713 kV (L-N)
VDROP_DENOM  = (V_BASE_LN_KV ** 2) * 1000.0    # = 7680.0  (3× tighter than L-L form)

# The active fault list (dynamically updated)
faulted_lines = []

# --- 1. LOAD COORDINATES ---
def load_bus_coords():
    """Extracts bus coordinates from the IEEE 37-bus CSV file."""
    coords = {}
    if os.path.exists("IEEE37_BusXY.csv"):
        with open("IEEE37_BusXY.csv", "r") as f:
            reader = csv.reader(f)
            for parts in reader:
                if len(parts) >= 3:
                    try:
                        coords[parts[0].strip().lower()] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
    return coords

pos = load_bus_coords()

# --- 2. CORE FUNCTIONS ---
def apply_initial_conditions():
    """Compiles OpenDSS and applies the current scenario's faults."""
    dss.Command('Clear')
    dss.Command(f'Compile "{dss_file}"')
    for line in faulted_lines:
        dss.Command(f'Open Line.{line} terminal=1')
        dss.Command(f'Open Line.{line} terminal=2')

def refresh_grid_state():
    """Extracts the exact base-grid parameters for the MILP."""
    global nodes, loads, load_weights, switchable_lines, switch_names
    global all_lines, transformers, blocks, bus_to_block
    global G_plot, fault_edges

    apply_initial_conditions()

    # 1. Nodes
    nodes = [bus.lower() for bus in dss.Circuit.AllBusNames()]

    # 2. Loads
    loads = []
    load_weights = {}
    seen_loads = set()

    dss.Loads.First()
    for _ in range(dss.Loads.Count()):
        lname = dss.Loads.Name().lower().strip()
        if lname in seen_loads:
            dss.Loads.Next()
            continue

        bname_full = dss.CktElement.BusNames()[0].lower()
        bname = bname_full.split('.')[0].strip()
        load_nodes = bname_full.split('.')[1:] if '.' in bname_full else [str(i+1) for i in range(dss.Loads.Phases())]
        phase_map = {'1': 'a', '2': 'b', '3': 'c'}
        present_phases = [phase_map.get(n) for n in load_nodes if n in phase_map]

        kw_per_phase   = dss.Loads.kW()   / len(present_phases) if present_phases else 0
        kvar_per_phase = dss.Loads.kvar() / len(present_phases) if present_phases else 0

        loads.append(lname)
        seen_loads.add(lname)
        load_weights[lname] = {
            'bus': bname, 'phases': present_phases,
            'kW_phase': kw_per_phase, 'kvar_phase': kvar_per_phase, 'priority': 1.0,
            'aZ': 0.2, 'aI': 0.3, 'aP': 0.5, 'aZ_q': 0.5, 'aI_q': 0.3, 'aP_q': 0.2
        }
        dss.Loads.Next()

    # Disable components for MILP
    dss.Loads.First()
    for _ in range(dss.Loads.Count()):
        dss.Command(f'Load.{dss.Loads.Name()}.enabled=no')
        dss.Loads.Next()
    dss.Generators.First()
    for _ in range(dss.Generators.Count()):
        dss.Command(f'Generator.{dss.Generators.Name()}.enabled=no')
        dss.Generators.Next()

    # 3. Switchable Lines
    switchable_lines = []
    seen_switches = set()
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        line_name = dss.Lines.Name().lower()
        if line_name in faulted_lines:
            dss.Lines.Next()
            continue

        is_switch = dss.Lines.IsSwitch() if hasattr(dss.Lines, 'IsSwitch') else False
        if line_name.startswith('sw'):
            is_switch = True

        if is_switch and line_name not in seen_switches:
            b1 = dss.Lines.Bus1().split('.')[0].lower().strip()
            b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
            switchable_lines.append({'name': line_name, 'bus1': b1, 'bus2': b2})
            seen_switches.add(line_name)
        dss.Lines.Next()
    switch_names = [sw['name'] for sw in switchable_lines]

    # 4. All Lines
    all_lines = []
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        name = dss.Lines.Name().lower()
        if name in faulted_lines:
            dss.Lines.Next()
            continue

        bus1_full = dss.Lines.Bus1().lower()
        b1 = bus1_full.split('.')[0].strip()
        b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
        nodes1 = bus1_full.split('.')[1:] if '.' in bus1_full else [str(i+1) for i in range(dss.Lines.Phases())]
        phase_map = {'1': 'a', '2': 'b', '3': 'c'}
        present_phases = [phase_map.get(n) for n in nodes1 if n in phase_map]

        rmat_flat = dss.Lines.RMatrix()
        xmat_flat = dss.Lines.XMatrix()
        r_3x3 = {p1: {p2: 0.0 for p2 in ['a', 'b', 'c']} for p1 in ['a', 'b', 'c']}
        x_3x3 = {p1: {p2: 0.0 for p2 in ['a', 'b', 'c']} for p1 in ['a', 'b', 'c']}
        num_phases = dss.Lines.Phases()
        if len(rmat_flat) == num_phases ** 2:
            idx = 0
            for i in range(num_phases):
                for j in range(num_phases):
                    if i < len(present_phases) and j < len(present_phases):
                        r_3x3[present_phases[i]][present_phases[j]] = rmat_flat[idx]
                        x_3x3[present_phases[i]][present_phases[j]] = xmat_flat[idx]
                    idx += 1

        all_lines.append({
            'name': name, 'bus1': b1, 'bus2': b2,
            'r_matrix': r_3x3, 'x_matrix': x_3x3, 'phases': present_phases
        })
        dss.Lines.Next()

    # 5. Transformers
    transformers = []
    dss.Transformers.First()
    for _ in range(dss.Transformers.Count()):
        t_name = dss.Transformers.Name().lower()
        buses = dss.CktElement.BusNames()
        b1 = buses[0].split('.')[0].lower().strip()
        b2 = buses[1].split('.')[0].lower().strip() if len(buses) > 1 else b1
        phase_map = {'1': 'a', '2': 'b', '3': 'c'}
        nodes2 = buses[1].split('.')[1:] if '.' in buses[1] else [str(i+1) for i in range(dss.CktElement.NumPhases())]
        sec_phases = [phase_map.get(n) for n in nodes2 if n in phase_map]
        dss.Transformers.Wdg(2)
        tap_ratio = dss.Transformers.Tap()
        a_ratio = {p: (tap_ratio if p in sec_phases else 1.0) for p in ['a', 'b', 'c']}
        transformers.append({
            'name': t_name, 'bus1': b1, 'bus2': b2, 'a_ratio': a_ratio, 'phases': sec_phases
        })
        dss.Transformers.Next()

    # 6. Capacitors
    global capacitors
    capacitors = []
    dss.Capacitors.First()
    for _ in range(dss.Capacitors.Count()):
        c_name = dss.Capacitors.Name().lower().strip()
        bname_full = dss.CktElement.BusNames()[0].lower()
        bname = bname_full.split('.')[0].strip()
        cap_nodes = bname_full.split('.')[1:] if '.' in bname_full else ['1', '2', '3']
        phase_map = {'1': 'a', '2': 'b', '3': 'c'}
        present_phases = [phase_map.get(n) for n in cap_nodes if n in phase_map]
        kvar_per_phase = dss.Capacitors.kvar() / len(present_phases) if present_phases else 0
        capacitors.append({
            'name': c_name, 'bus': bname,
            'phases': present_phases, 'kvar_phase': kvar_per_phase
        })
        dss.Command(f'Capacitor.{c_name}.enabled=no')
        dss.Capacitors.Next()

    # 7. Graph and Blocks
    G_plot = nx.Graph()
    G_internal = nx.Graph()
    fault_edges = []

    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        b1 = dss.Lines.Bus1().split('.')[0].lower().strip()
        b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
        line_name = dss.Lines.Name().lower()

        if b1 in pos and b2 in pos:
            G_plot.add_edge(b1, b2)

        if line_name in faulted_lines:
            fault_edges.append((b1, b2))
            dss.Lines.Next()
            continue

        is_switch = dss.Lines.IsSwitch() if hasattr(dss.Lines, 'IsSwitch') else False
        if line_name.startswith('sw'):
            is_switch = True

        if not is_switch:
            G_internal.add_edge(b1, b2)
        dss.Lines.Next()

    dss.Transformers.First()
    for _ in range(dss.Transformers.Count()):
        buses = dss.CktElement.BusNames()
        if len(buses) > 1:
            b1 = buses[0].split('.')[0].lower().strip()
            b2 = buses[1].split('.')[0].lower().strip()
            if b1 in pos and b2 in pos:
                G_plot.add_edge(b1, b2)
            G_internal.add_edge(b1, b2)
        dss.Transformers.Next()

    blocks = list(nx.connected_components(G_internal))
    bus_to_block = {}
    for k, block in enumerate(blocks):
        for bus in block:
            bus_to_block[bus] = k


# ==========================================
# 2. THE VISUALIZATION ENGINE
# ==========================================
def plot_saa_results(purchased_dgs, dg_sizes, scenario_results, pos, sub_bus):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    G = nx.Graph()
    dss.Command('Clear')
    dss.Command(f'Compile "{dss_file}"')

    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        b1 = dss.Lines.Bus1().split('.')[0].lower().strip()
        b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
        if b1 in pos and b2 in pos:
            G.add_edge(b1, b2)
        dss.Lines.Next()

    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='lightgray', width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=30, node_color='black')
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=5, font_color='dimgray')

    if sub_bus in pos:
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[sub_bus], node_size=150, node_shape='s', node_color='blue')
        ax1.text(pos[sub_bus][0]+50, pos[sub_bus][1]+50, "SUBSTATION", color='blue', fontweight='bold', fontsize=7)

    valid_dg_nodes = [dg for dg in purchased_dgs if dg in pos]
    if valid_dg_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=valid_dg_nodes, node_size=200, node_color='orange', edgecolors='red')
        for dg in valid_dg_nodes:
            ax1.text(pos[dg][0]+50, pos[dg][1]+50, f"DG ({dg_sizes[dg]:.0f} kW)", color='darkred', fontweight='bold', fontsize=7)

    ax1.set_title("SAA Master Blueprint: Physical DG Placements", fontsize=14, fontweight='bold')
    ax1.axis('off')

    storms = list(scenario_results.keys())
    shed_values = [scenario_results[s]['shed'] for s in storms]

    ax2.bar(storms, shed_values, color='tomato', edgecolor='black')
    ax2.set_title("Resilience Stress Test: Load Shedding per Storm", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Load Shedding (kW)", fontsize=12)
    ax2.set_xlabel("Monte Carlo Scenarios", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. THE SAA MASTER SOLVER
# ==========================================
def solve_stochastic_saa(fault_scenarios_dict, export_filename,
                         mip_gap=0.005, lambda_risk=0.0, alpha=0.90,
                         budget_max=MAX_INVESTMENT_BUDGET,
                         conditional_occ=CONDITIONAL_EVENT_OCCURRENCE):

    print(f"\n🚀 INITIATING PURE SAA MILP (N={len(fault_scenarios_dict)})...")

    # 1. Refresh the OpenDSS grid to base state
    refresh_grid_state()

    dss.Vsources.First()
    SUB_BUS = dss.CktElement.BusNames()[0].split('.')[0].lower().strip()
    print(f"Substation Bus Identified As: {SUB_BUS}")

    # Cache bus phase and voltage info early
    global bus_phase_info
    bus_phase_info = {}
    dss.Command('Clear')
    dss.Command(f'Compile "{dss_file}"')
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        bus_phase_info[bus_name.lower().strip()] = {
            'num_phases': dss.Bus.NumNodes(),
            'nodes': list(dss.Bus.Nodes()),
            'kv_base': dss.Bus.kVBase()
        }

    # 2. Total grid load
    global total_grid_kw
    total_grid_kw = sum(load_weights[l]['kW_phase'] * len(load_weights[l]['phases']) for l in loads)
    print(f"Total Base Grid Load: {total_grid_kw:.2f} kW")

    # Loads-at-node lookup
    loads_at_node = {i: [] for i in nodes}
    for l in loads:
        bus = load_weights[l]['bus']
        if bus in loads_at_node:
            loads_at_node[bus].append(l)

    # Detect voltage regulator secondary buses
    regulator_buses = set()
    dss.RegControls.First()
    for _ in range(dss.RegControls.Count()):
        tx_name = dss.RegControls.Transformer()
        dss.Transformers.Name(tx_name)
        buses = dss.CktElement.BusNames()
        if len(buses) > 1:
            regulator_buses.add(buses[1].split('.')[0].lower().strip())
        dss.RegControls.Next()
    if regulator_buses:
        print(f"  Regulator secondary buses excluded from DG candidates: {regulator_buses}")

    # 3. Extract Scenarios
    scenarios = list(fault_scenarios_dict.keys())

    phases = ['a', 'b', 'c']

    # Degree map -- exclude leaf nodes (degree==1) as a candidate-screening heuristic:
    # utility-scale DG at terminal laterals provides limited restoration reach and may
    # require additional interconnection facilities not modelled here.
    _degree = Counter()
    for _l in all_lines:
        _degree[_l['bus1']] += 1
        _degree[_l['bus2']] += 1
    for _t in transformers:
        _degree[_t['bus1']] += 1
        _degree[_t['bus2']] += 1

    candidate_buses = [
        b for b in nodes
        if b != SUB_BUS
        and bus_phase_info.get(b, {}).get('num_phases', 0) == 3
        and _degree.get(b, 0) > 1
        and b not in regulator_buses
    ]
    print(f"  {len(candidate_buses)} eligible DG candidate buses.")

    # Bus adjacency lookups -- built once instead of scanning all_lines/transformers
    # for every (node, phase, scenario) inside the constraint loops.
    lines_into_bus   = {i: [] for i in nodes}
    lines_out_of_bus = {i: [] for i in nodes}
    for line in all_lines:
        lines_into_bus.setdefault(line['bus2'], []).append(line['name'])
        lines_out_of_bus.setdefault(line['bus1'], []).append(line['name'])
    tx_into_bus   = {i: [] for i in nodes}
    tx_out_of_bus = {i: [] for i in nodes}
    for tx in transformers:
        tx_into_bus.setdefault(tx['bus2'], []).append(tx['name'])
        tx_out_of_bus.setdefault(tx['bus1'], []).append(tx['name'])

    saa_model = pulp.LpProblem("FISR_SAA", pulp.LpMinimize)

    # -- First-stage planning variables --
    x_G   = pulp.LpVariable.dicts("x_G",   candidate_buses, cat='Binary')
    S_Cap = pulp.LpVariable.dicts("S_Cap", candidate_buses, lowBound=0, cat='Continuous')

    s_N   = pulp.LpVariable.dicts("s_N",   ((i, s) for i in nodes for s in scenarios), cat='Binary')
    x_BR  = pulp.LpVariable.dicts("x_BR",  ((sw, s) for sw in switch_names for s in scenarios), cat='Binary')
    x_Cap = pulp.LpVariable.dicts("x_Cap", ((cap['name'], s) for cap in capacitors for s in scenarios), cat='Binary')

    # Big-M must be tight: LP relaxation quality (and BnB speed) degrades linearly
    # with M size. M_Power only needs to exceed max possible net power at any node:
    # max DG per phase (1000/3 * 0.8 ≈ 267 kW) + LINE_CAPACITY (500 kW) with margin.
    # M_Volt only needs to exceed max U difference across any open element:
    # U range is [0.95², 1.05²] = [0.9025, 1.1025], so delta < 0.21 pu².
    M_Power = 1000.0   # was 10000 — tighter M gives stronger LP relaxation
    # M_Volt must be >= U_max (1.1025 pu²): when a live bus (U≈1.0) borders a dead
    # bus (U=0) across an open switch, the gate constraint is 1.0 <= v_drop + M_Volt.
    # v_drop is tiny (~0.01), so M_Volt < 1.0 makes the model infeasible at presolve.
    M_Volt  = 1.5      # was 2.0 — safe lower bound is 1.1; 1.5 gives margin

    P_G    = pulp.LpVariable.dicts("P_G",    ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases for s in scenarios), lowBound=0, cat='Continuous')
    Q_G    = pulp.LpVariable.dicts("Q_G",    ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases for s in scenarios), lowBound=-300, upBound=300, cat='Continuous')
    U_N    = pulp.LpVariable.dicts("U_N",    ((i, p, s) for i in nodes for p in phases for s in scenarios), cat='Continuous')
    P_Line = pulp.LpVariable.dicts("P_Line", ((line['name'], p, s) for line in all_lines for p in phases for s in scenarios), lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')
    Q_Line = pulp.LpVariable.dicts("Q_Line", ((line['name'], p, s) for line in all_lines for p in phases for s in scenarios), lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')

    # Radiality topology variables
    N_nodes = len(nodes)
    # x_Line is declared Continuous [0,1] but is integral in every feasible
    # solution: faulted lines are fixed to 0, switch lines are forced equal to
    # the Binary x_BR, and hardwired lines equal the AND of s_N[bus1]/s_N[bus2]
    # (which the connectivity propagation constraint forces to be equal Binary
    # values). It therefore always resolves to an exact 0/1 "line conducting"
    # status; relaxing it just removes ~|lines|x|scenarios| binaries from B&B.
    x_Line  = pulp.LpVariable.dicts("x_Line", ((line['name'], s) for line in all_lines for s in scenarios), lowBound=0, upBound=1, cat='Continuous')
    f_Line  = pulp.LpVariable.dicts("f_Line", ((line['name'], s) for line in all_lines for s in scenarios), cat='Continuous')
    z_Vir   = pulp.LpVariable.dicts("z_Vir",  ((i, s) for i in nodes for s in scenarios), cat='Binary')
    f_Vir   = pulp.LpVariable.dicts("f_Vir",  ((i, s) for i in nodes for s in scenarios), lowBound=0, cat='Continuous')
    f_Tx    = pulp.LpVariable.dicts("f_Tx",   ((tx['name'], s) for tx in transformers for s in scenarios), cat='Continuous')
    P_Tx    = pulp.LpVariable.dicts("P_Tx",   ((tx['name'], p, s) for tx in transformers for p in phases for s in scenarios), cat='Continuous')
    Q_Tx    = pulp.LpVariable.dicts("Q_Tx",   ((tx['name'], p, s) for tx in transformers for p in phases for s in scenarios), cat='Continuous')

    # Linearization: S_Cap[c] * s_N[c,s] auxiliary variable (McCormick envelope)
    # S_Cap_max = 1000 kW / 0.8 pf = 1250 kVA (nameplate upper bound)
    S_Cap_max = 1250.0
    S_Cap_active = pulp.LpVariable.dicts(
        "S_Cap_active",
        ((c, s) for c in candidate_buses for s in scenarios),
        lowBound=0, cat='Continuous'
    )
    for c in candidate_buses:
        for s in scenarios:
            saa_model += S_Cap_active[c, s] <= S_Cap_max * s_N[c, s]
            saa_model += S_Cap_active[c, s] <= S_Cap[c]
            saa_model += S_Cap_active[c, s] >= S_Cap[c] - S_Cap_max * (1 - s_N[c, s])

    # ==========================================
    # OBJECTIVE FUNCTION
    # minimize: capital investment + lifetime O&M + expected disaster cost
    # No normal-operation energy costs -- DGs are standby-only assets.
    # ==========================================
    cost_dg = pulp.lpSum([COST_DG_FIXED * x_G[c] + COST_DG_PER_KW * 0.8 * S_Cap[c] for c in candidate_buses])

    # O&M is paid annually; PWF discounts the uniform series to present value
    maintenance_pv = pulp.lpSum([
        COST_DG_OM_PER_KW_YR * PWF * 0.8 * S_Cap[c]
        for c in candidate_buses
    ])

    # Load priority factors (f_i)
    # Bus-specific critical override takes precedence over kW-threshold tiers.
    f_priority = {}
    for l in loads:
        bus = load_weights[l]['bus']
        total_load_kw = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
        if bus in CRITICAL_BUSES:
            f_priority[l] = VOLL_CRITICAL          # hospital / emergency services
        elif total_load_kw >= 50.0:
            f_priority[l] = VOLL_TIER_2            # commercial / industrial
        else:
            f_priority[l] = VOLL_TIER_1            # residential

    # ==========================================
    # RISK-AVERSE CVaR OBJECTIVE FUNCTION
    # ==========================================
    ALPHA       = alpha        # set via _CFG_ALPHA or orchestrator arg
    LAMBDA_RISK = lambda_risk  # set via _CFG_LAMBDA or orchestrator arg
    PROB_S = 1.0 / len(scenarios)

    # CVaR Auxiliary Variables (Rockafellar & Uryasev Formulation)
    # NOTE: R&U define zeta (VaR) as a free variable. lowBound=0 is safe here
    # only because every scenario cost is nonnegative (fuel + shedding). If a
    # revenue/credit term is ever added, remove this bound.
    zeta = pulp.LpVariable("VaR_Zeta", lowBound=0, cat='Continuous')
    eta = pulp.LpVariable.dicts("CVaR_Eta", scenarios, lowBound=0, cat='Continuous')

    # Explicit load-shedding power variable (kW shed at load l in scenario s).
    # Replaces the algebraic f*P*(1 - s_N) penalty, which expands to a large
    # decision-INDEPENDENT constant (the cost of shedding every load) minus a
    # reward term. PuLP does not pass an objective constant to Gurobi, so that
    # form made the solver minimise a constant-shifted objective: the logged
    # "best objective"/"best bound" came out NEGATIVE and the reported MIP gap
    # was inflated (its denominator was missing the constant), wasting solve
    # time. With shed as a real variable the objective is constant-free, so the
    # logged bound is positive and comparable to the incumbent, and Gurobi hits
    # the true gap sooner. The positive shedding coefficient drives shed down to
    # its lower bound = P_dem*(1 - s_N) at the optimum, so the cost is identical.
    shed = pulp.LpVariable.dicts(
        "shed", ((l, s) for l in loads for s in scenarios),
        lowBound=0, cat='Continuous'
    )

    scenario_costs = {}

    for s in scenarios:
        scenario_duration = fault_scenarios_dict[s]["duration_hours"]

        # 1. Fuel cost for THIS scenario
        s_fuel_cost = pulp.lpSum([
            P_G[c, p, s] * COST_GEN_PER_KWH * scenario_duration
            for c in candidate_buses for p in phases
        ])

        # 2. Load shedding cost for THIS scenario.
        # Link shed >= de-energized load; the positive penalty pins it to equality.
        for l in loads:
            bus = load_weights[l]['bus']
            num_phases = len(load_weights[l]['phases'])
            saa_model += shed[l, s] >= load_weights[l]['kW_phase'] * num_phases * (1 - s_N[bus, s])
        s_shed_cost = pulp.lpSum([
            f_priority[l] * shed[l, s] * scenario_duration for l in loads
        ])

        # 3. Total Financial Risk for THIS scenario
        # Evaluated as a CONDITIONAL event (Multiplier = 1.0)
        scenario_costs[s] = (s_fuel_cost + s_shed_cost) * conditional_occ

        # 4. CVaR Linear Constraint: eta[s] >= Cost[s] - VaR_Threshold
        saa_model += eta[s] >= scenario_costs[s] - zeta

    # Expected Value (Average risk of all scenarios)
    expected_disaster_cost = pulp.lpSum([PROB_S * scenario_costs[s] for s in scenarios])

    # CVaR (Average risk of the worst 10% tail)
    cvar_disaster_cost = zeta + (1.0 / (1.0 - ALPHA)) * pulp.lpSum([PROB_S * eta[s] for s in scenarios])

    # Final Objective: Combine CAPEX, O&M (PV), Expected Risk, and Tail Risk
    # CapEx paid at t=0 — already at present value (no CRF needed)
    capex_present_value = cost_dg
    saa_model += capex_present_value + maintenance_pv + ((1.0 - LAMBDA_RISK) * expected_disaster_cost) + (LAMBDA_RISK * cvar_disaster_cost), "Minimize_Risk_Averse_Cost"
    # ==========================================
    # Investment constraints
    # S_Cap bounds in kVA: P_max = 0.8×S_cap ∈ [100, 1000] kW
    for c in candidate_buses:
        saa_model += S_Cap[c] <= 1250.0 * x_G[c]   # 1000 kW / 0.8 pf
        saa_model += S_Cap[c] >= 125.0  * x_G[c]   # 100 kW / 0.8 pf
    saa_model += pulp.lpSum([x_G[c] for c in candidate_buses]) <= 4
    saa_model += cost_dg <= budget_max, "Budget_Limit"

    # Operational constraints
    sqrt3_2 = math.sqrt(3) / 2.0
    alpha_real = {'a': {'a': 1.0, 'b': -0.5, 'c': -0.5}, 'b': {'a': -0.5, 'b': 1.0, 'c': -0.5}, 'c': {'a': -0.5, 'b': -0.5, 'c': 1.0}}
    alpha_imag = {'a': {'a': 0.0, 'b': -sqrt3_2, 'c': sqrt3_2}, 'b': {'a': sqrt3_2, 'b': 0.0, 'c': -sqrt3_2}, 'c': {'a': -sqrt3_2, 'b': sqrt3_2, 'c': 0.0}}
    U_max, U_min = 1.05 ** 2, 0.95 ** 2

    for s in scenarios:
        current_faults = {f.lower() for f in fault_scenarios_dict[s]["faults"]}

        # Substation is DEAD in all disaster scenarios
        saa_model += s_N[SUB_BUS, s] == 0, f"Force_Substation_DEAD_{s}"
        for p in phases:
            saa_model += U_N[SUB_BUS, p, s] == 0, f"Substation_V_Dead_{p}_{s}"
            saa_model += P_G['MAIN_SUBSTATION', p, s] == 0, f"Sub_P_Dead_{p}_{s}"
            saa_model += Q_G['MAIN_SUBSTATION', p, s] == 0, f"Sub_Q_Dead_{p}_{s}"

        # DG buses anchor island voltage when alive
        for c in candidate_buses:
            for p in phases:
                saa_model += U_N[c, p, s] >= 1.0 - M_Volt * (2 - x_G[c] - s_N[c, s]), f"DG_V_lo_{c}_{p}_{s}"
                saa_model += U_N[c, p, s] <= 1.0 + M_Volt * (2 - x_G[c] - s_N[c, s]), f"DG_V_hi_{c}_{p}_{s}"

        # DG output limits (per-phase, gated by bus status)
        for c in candidate_buses:
            for p in phases:# Real power limit: P_max = 0.8 × S_cap (nameplate kVA → rated kW)
                saa_model += P_G[c, p, s] <= 0.8 * (S_Cap[c] / 3.0)
                saa_model += P_G[c, p, s] >= 0
                saa_model += P_G[c, p, s] <= M_Power * s_N[c, s]
                # Reactive power limit (0.8 pf → Q_max = 0.6 × S_cap)
                saa_model += Q_G[c, p, s] <= 0.6 * (S_Cap[c] / 3.0)
                saa_model += Q_G[c, p, s] >= -0.6 * (S_Cap[c] / 3.0)
                saa_model += Q_G[c, p, s] <=  M_Power * s_N[c, s]
                saa_model += Q_G[c, p, s] >= -M_Power * s_N[c, s]
            p_total = pulp.lpSum([P_G[c, p, s] for p in phases])
            saa_model += p_total <= 0.8 * S_Cap_active[c, s]

        # -- RADIALITY: Line Status --
        for line in all_lines:
            name = line['name']
            b1, b2 = line['bus1'], line['bus2']
            if name in current_faults:
                saa_model += x_Line[name, s] == 0
            elif name in switch_names:
                saa_model += x_Line[name, s] == x_BR[name, s]
            else:
                saa_model += x_Line[name, s] <= s_N[b1, s]
                saa_model += x_Line[name, s] <= s_N[b2, s]
                saa_model += x_Line[name, s] >= s_N[b1, s] + s_N[b2, s] - 1

        # -- RADIALITY: Virtual node connects ONLY to DG buses --
        for i in nodes:
            if i == SUB_BUS:
                saa_model += z_Vir[i, s] == 0
            elif i in candidate_buses:
                saa_model += z_Vir[i, s] <= x_G[i]
                saa_model += z_Vir[i, s] <= s_N[i, s]
            else:
                saa_model += z_Vir[i, s] == 0

        # Spanning forest: line edges + transformer edges + virtual roots = live nodes
        saa_model += (
            pulp.lpSum([x_Line[line['name'], s] for line in all_lines]) +
            pulp.lpSum([s_N[tx['bus1'], s] for tx in transformers]) +
            pulp.lpSum([z_Vir[i, s] for i in nodes])
            == pulp.lpSum([s_N[i, s] for i in nodes]),
            f"Spanning_Tree_Edges_{s}"
        )

        # Fictitious flow
        for i in nodes:
            saa_model += f_Vir[i, s] <= N_nodes * z_Vir[i, s]
            saa_model += (
                f_Vir[i, s]
                + pulp.lpSum([f_Line[n, s] for n in lines_into_bus[i]])
                + pulp.lpSum([f_Tx[n, s]   for n in tx_into_bus[i]])
                - pulp.lpSum([f_Line[n, s] for n in lines_out_of_bus[i]])
                - pulp.lpSum([f_Tx[n, s]   for n in tx_out_of_bus[i]])
                == s_N[i, s],
                f"Fictitious_Flow_Balance_{i}_{s}"
            )
        for line in all_lines:
            saa_model += f_Line[line['name'], s] <=  N_nodes * x_Line[line['name'], s]
            saa_model += f_Line[line['name'], s] >= -N_nodes * x_Line[line['name'], s]
        for tx in transformers:
            saa_model += f_Tx[tx['name'], s] <=  N_nodes * s_N[tx['bus1'], s]
            saa_model += f_Tx[tx['name'], s] >= -N_nodes * s_N[tx['bus1'], s]

        # -- CONNECTIVITY PROPAGATION --
        # Hardwired (non-switch, non-faulted) lines and transformers bond their endpoints.
        for line in all_lines:
            name = line['name']
            if name in current_faults or name in switch_names:
                continue
            saa_model += s_N[line['bus1'], s] == s_N[line['bus2'], s], f"BusProp_{name}_{s}"
        for tx in transformers:
            saa_model += s_N[tx['bus1'], s] == s_N[tx['bus2'], s], f"TxProp_{tx['name']}_{s}"

        # Voltage bounds
        for i in nodes:
            for p in phases:
                saa_model += U_N[i, p, s] <= s_N[i, s] * U_max
                saa_model += U_N[i, p, s] >= s_N[i, s] * U_min

        # Switch connectivity
        for sw in switchable_lines:
            saa_model += x_BR[sw['name'], s] <= s_N[sw['bus1'], s]
            saa_model += x_BR[sw['name'], s] <= s_N[sw['bus2'], s]

        # Capacitor status
        for cap in capacitors:
            saa_model += x_Cap[cap['name'], s] <= s_N[cap['bus'], s]

        # -- POWER BALANCE & POWER FLOW --
        for p in phases:
            for i in nodes:
                p_in_sum  = pulp.lpSum([P_Line[ln, p, s] for ln in lines_into_bus[i]])
                p_out_sum = pulp.lpSum([P_Line[ln, p, s] for ln in lines_out_of_bus[i]])
                q_in_sum  = pulp.lpSum([Q_Line[ln, p, s] for ln in lines_into_bus[i]])
                q_out_sum = pulp.lpSum([Q_Line[ln, p, s] for ln in lines_out_of_bus[i]])
                p_tx_in   = pulp.lpSum([P_Tx[n, p, s] for n in tx_into_bus[i]])
                p_tx_out  = pulp.lpSum([P_Tx[n, p, s] for n in tx_out_of_bus[i]])
                q_tx_in   = pulp.lpSum([Q_Tx[n, p, s] for n in tx_into_bus[i]])
                q_tx_out  = pulp.lpSum([Q_Tx[n, p, s] for n in tx_out_of_bus[i]])

                base_p = sum(load_weights[l]['kW_phase']   for l in loads_at_node[i] if p in load_weights[l]['phases'])
                base_q = sum(load_weights[l]['kvar_phase'] for l in loads_at_node[i] if p in load_weights[l]['phases'])
                p_served = base_p * s_N[i, s]
                q_served = base_q * s_N[i, s]

                p_dg = P_G['MAIN_SUBSTATION', p, s] if i == SUB_BUS else (P_G[i, p, s] if i in candidate_buses else 0)
                q_dg = Q_G['MAIN_SUBSTATION', p, s] if i == SUB_BUS else (Q_G[i, p, s] if i in candidate_buses else 0)
                q_cap = pulp.lpSum([
                    x_Cap[cap['name'], s] * cap['kvar_phase']
                    for cap in capacitors if cap['bus'] == i and p in cap['phases']
                ])

                # M-gated balance: trivially satisfied when bus is dead
                saa_model += p_in_sum + p_dg + p_tx_in - p_out_sum - p_tx_out - p_served <=  M_Power * (1 - s_N[i, s]), f"PBal_hi_{i}_{p}_{s}"
                saa_model += p_in_sum + p_dg + p_tx_in - p_out_sum - p_tx_out - p_served >= -M_Power * (1 - s_N[i, s]), f"PBal_lo_{i}_{p}_{s}"
                saa_model += q_in_sum + q_dg + q_cap + q_tx_in - q_out_sum - q_tx_out - q_served <=  M_Power * (1 - s_N[i, s]), f"QBal_hi_{i}_{p}_{s}"
                saa_model += q_in_sum + q_dg + q_cap + q_tx_in - q_out_sum - q_tx_out - q_served >= -M_Power * (1 - s_N[i, s]), f"QBal_lo_{i}_{p}_{s}"

            for line in all_lines:
                name = line['name']
                b1, b2 = line['bus1'], line['bus2']
                if p not in line['phases']:
                    saa_model += P_Line[name, p, s] == 0, f"PLine_NoPhase_{name}_{p}_{s}"
                    saa_model += Q_Line[name, p, s] == 0, f"QLine_NoPhase_{name}_{p}_{s}"
                    continue

                v_drop = 0
                for m in line['phases']:
                    r_t = alpha_real[p][m] * line['r_matrix'][p][m] - alpha_imag[p][m] * line['x_matrix'][p][m]
                    x_t = alpha_real[p][m] * line['x_matrix'][p][m] + alpha_imag[p][m] * line['r_matrix'][p][m]
                    v_drop += 2 * (r_t * P_Line[name, m, s] + x_t * Q_Line[name, m, s]) / VDROP_DENOM

                if name in current_faults:
                    saa_model += P_Line[name, p, s] == 0
                    saa_model += Q_Line[name, p, s] == 0
                elif name in switch_names:
                    saa_model += U_N[b1,p,s] - U_N[b2,p,s] <=  v_drop + M_Volt * (1 - x_BR[name, s])
                    saa_model += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop  - M_Volt * (1 - x_BR[name, s])
                    saa_model += P_Line[name, p, s] <=  M_Power * x_BR[name, s]
                    saa_model += P_Line[name, p, s] >= -M_Power * x_BR[name, s]
                    saa_model += Q_Line[name, p, s] <=  M_Power * x_BR[name, s]
                    saa_model += Q_Line[name, p, s] >= -M_Power * x_BR[name, s]
                else:
                    # M-gated for hardwired lines -- prevents infeasibility when line is dead
                    saa_model += U_N[b1,p,s] - U_N[b2,p,s] <=  v_drop + M_Volt * (1 - x_Line[name, s])
                    saa_model += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop  - M_Volt * (1 - x_Line[name, s])
                    saa_model += P_Line[name, p, s] <=  M_Power * x_Line[name, s]
                    saa_model += P_Line[name, p, s] >= -M_Power * x_Line[name, s]
                    saa_model += Q_Line[name, p, s] <=  M_Power * x_Line[name, s]
                    saa_model += Q_Line[name, p, s] >= -M_Power * x_Line[name, s]

            for tx in transformers:
                name = tx['name']
                b1, b2 = tx['bus1'], tx['bus2']
                if p not in tx['phases']:
                    continue
                a_sq = tx['a_ratio'][p] ** 2
                if name in switch_names:
                    saa_model += (a_sq * U_N[b2,p,s]) - U_N[b1,p,s] <=  M_Volt * (1 - x_BR[name, s])
                    saa_model += (a_sq * U_N[b2,p,s]) - U_N[b1,p,s] >= -M_Volt * (1 - x_BR[name, s])
                else:
                    # Gate on BOTH primary and secondary bus status
                    t_live = s_N[b1, s] + s_N[b2, s]
                    saa_model += (a_sq * U_N[b2,p,s]) - U_N[b1,p,s] <=  M_Volt * (2 - t_live)
                    saa_model += (a_sq * U_N[b2,p,s]) - U_N[b1,p,s] >= -M_Volt * (2 - t_live)

        # Line thermal capacity (phase-independent constraints — added once per
        # line/phase, outside the p-loop where they were previously triplicated)
        for line in all_lines:
            name = line['name']
            for ph in line['phases']:
                saa_model += ( sqrt3 * P_Line[name, ph, s] + Q_Line[name, ph, s]) <=  2 * LINE_CAPACITY
                saa_model += ( sqrt3 * P_Line[name, ph, s] + Q_Line[name, ph, s]) >= -2 * LINE_CAPACITY
                saa_model += ( sqrt3 * P_Line[name, ph, s] - Q_Line[name, ph, s]) <=  2 * LINE_CAPACITY
                saa_model += ( sqrt3 * P_Line[name, ph, s] - Q_Line[name, ph, s]) >= -2 * LINE_CAPACITY
                saa_model += P_Line[name, ph, s] <=  LINE_CAPACITY
                saa_model += P_Line[name, ph, s] >= -LINE_CAPACITY
                saa_model += Q_Line[name, ph, s] <=  LINE_CAPACITY
                saa_model += Q_Line[name, ph, s] >= -LINE_CAPACITY

        # Transformer power flow: M-gate + octagonal thermal capacity
        for tx in transformers:
            name, b1 = tx['name'], tx['bus1']
            for ph in tx['phases']:
                saa_model += P_Tx[name, ph, s] <=  M_Power * s_N[b1, s]
                saa_model += P_Tx[name, ph, s] >= -M_Power * s_N[b1, s]
                saa_model += Q_Tx[name, ph, s] <=  M_Power * s_N[b1, s]
                saa_model += Q_Tx[name, ph, s] >= -M_Power * s_N[b1, s]
                saa_model += ( sqrt3*P_Tx[name, ph, s] + Q_Tx[name, ph, s]) <=  2*TX_CAPACITY
                saa_model += ( sqrt3*P_Tx[name, ph, s] + Q_Tx[name, ph, s]) >= -2*TX_CAPACITY
                saa_model += ( sqrt3*P_Tx[name, ph, s] - Q_Tx[name, ph, s]) <=  2*TX_CAPACITY
                saa_model += ( sqrt3*P_Tx[name, ph, s] - Q_Tx[name, ph, s]) >= -2*TX_CAPACITY
                saa_model += P_Tx[name, ph, s] <=  TX_CAPACITY
                saa_model += P_Tx[name, ph, s] >= -TX_CAPACITY
                saa_model += Q_Tx[name, ph, s] <=  TX_CAPACITY
                saa_model += Q_Tx[name, ph, s] >= -TX_CAPACITY

    print("\n🚀 Solving Master SAA Matrix (May take 5-20 mins)...")

    # Derive a per-run Gurobi log filename from the output JSON name so the
    # best bound can be extracted after the solve without changing the function
    # signature (export_filename is already in scope here).
    log_filename = export_filename.replace("master_dg_placements_", "gurobi_log_").replace(".json", ".txt")

    # Gurobi opens LogFile in APPEND mode, so this file may already hold the
    # summaries of many previous runs (including stale solves with negative
    # objectives). Record the current end-of-file so we parse ONLY the bytes
    # this solve appends — otherwise reversed-readlines can latch onto an old
    # run's "best bound" line and report a wrong (even negative) bound.
    log_offset = os.path.getsize(log_filename) if os.path.exists(log_filename) else 0

    # --- Start Timer ---
    start_time = time.time()

    saa_model.solve(pulp.GUROBI_CMD(msg=True, options=[
        ("MIPGap",       mip_gap),   # primary stopping criterion (e.g. 0.05 = 5%)
        ("TimeLimit",    1200),      # 20-min hard cap; MIPFocus usually hits 5% well before
        ("MIPFocus",     3),         # bound-focused: directly attacks the stagnation phases
        ("NumericFocus", 2),         # compensates for large coefficient range in unbalanced grid
        ("ScaleFlag",    2),         # aggressive scaling for numerically difficult models
        ("Cuts", 2),                 # aggressive cuts to tighten LP relaxation during stagnation
        ("LogFile",      log_filename),  # written so we can extract the best bound below
    ]))

    # --- End Timer ---
    solve_duration = time.time() - start_time
    print(f"⏱️  Solver finished in {solve_duration:.2f} seconds.")

    # --- Parse Gurobi log for best bound and achieved MIP gap ---
    # The summary line looks like:
    #   Best objective 1.23e+05, best bound 1.17e+05, gap 5.00%
    best_bound      = None
    mip_gap_achieved = None
    try:
        with open(log_filename, "r") as _lf:
            _lf.seek(log_offset)                 # skip prior runs' appended output
            this_run_lines = _lf.readlines()
        for _line in reversed(this_run_lines):
            if "Best objective" in _line and "best bound" in _line:
                for _part in _line.split(','):
                    _part = _part.strip()
                    if _part.startswith("best bound"):
                        best_bound = float(_part.split()[2])
                    elif _part.startswith("gap"):
                        mip_gap_achieved = float(_part.split()[1].replace('%', '')) / 100
                break
    except Exception as _e:
        print(f"  [Warning] Could not parse Gurobi log for best bound: {_e}")
        best_bound = pulp.value(saa_model.objective)  # fall back to incumbent

    # Sanity check: for a minimization the bound can never exceed the incumbent.
    # If it does (stale/garbled parse), discard it rather than report nonsense.
    _incumbent = pulp.value(saa_model.objective)
    if best_bound is not None and _incumbent is not None and best_bound > _incumbent + 1.0:
        print(f"  [Warning] Parsed bound ${best_bound:,.2f} exceeds incumbent "
              f"${_incumbent:,.2f} — discarding as unreliable.")
        best_bound = None
        mip_gap_achieved = None

    # PuLP maps Gurobi status 9 (time-limit with incumbent) to LpStatusNotSolved
    # in some versions — check objective value directly so we never discard a
    # valid incumbent when the safety TimeLimit fires.
    if pulp.value(saa_model.objective) is not None:
        print("\n🏆 OPTIMIZATION COMPLETE 🏆")
        _obj      = pulp.value(saa_model.objective)
        _inv      = pulp.value(cost_dg)
        _maint_pv = pulp.value(maintenance_pv)
        _exp_dis  = pulp.value(expected_disaster_cost)
        _cvar_dis = pulp.value(cvar_disaster_cost)

        print(f"  Objective value (CVaR + Expected)        : ${_obj:>14,.2f}")
        if best_bound is not None:
            print(f"  Theoretical Best Bound (floor)           : ${best_bound:>14,.2f}")
        if mip_gap_achieved is not None:
            print(f"  Achieved MIP Gap                         :  {mip_gap_achieved*100:>13.2f}%")
        print(f"  CapEx (lump sum, present value)          : ${_inv:>14,.2f}")
        print(f"  O&M present value (PWF discounted)       : ${_maint_pv:>14,.2f}  [PWF={PWF:.4f}, Y={PLANNING_YEARS}yr, r={INTEREST_RATE*100:.0f}%]")
        print(f"  Expected disaster cost (Average)         : ${_exp_dis:>14,.2f}")
        print(f"  CVaR disaster cost (Worst 10%)           : ${_cvar_dis:>14,.2f}")

        purchased_dgs = []
        dg_sizes = {}
        for c in candidate_buses:
            if (pulp.value(x_G[c]) or 0) > 0.9:
                purchased_dgs.append(c)
                dg_sizes[c] = math.ceil((pulp.value(S_Cap[c]) or 0) * 10) / 10
                print(f"✅ Bus {c} | Diesel: {dg_sizes[c]:.1f} kVA (P_max={0.8*dg_sizes[c]:.1f} kW)")

        print("\n🌪️ INITIATING STORM ANALYSIS (MILP vs. OpenDSS Physics)...")
        scenario_results = {}

        # ── Auto-detect feeder L-L base voltage from bus_phase_info (cached above) ──
        # bus_phase_info was populated from dss.Bus.kVBase() on the compiled network.
        # kVBase() returns L-N kV in most opendssdirect builds (IEEE 37-bus ≈ 2.77 kV).
        # Vsource.basekv expects L-L kV, so convert when the raw value is < 3.5 kV.
        # Threshold: < 3.5 → L-N → × √3 (→ ~4.8 kV);  ≥ 3.5 → already L-L.
        _raw_kv = bus_phase_info.get('701', {}).get('kv_base', 2.771)
        if _raw_kv < 3.5:                           # L-N → convert to L-L
            system_kv_LL = round(_raw_kv * math.sqrt(3), 4)
        else:                                        # already L-L
            system_kv_LL = round(_raw_kv, 4)
        if not (1.0 <= system_kv_LL <= 15.0):       # sanity-check; IEEE 37-bus = 4.8 kV
            system_kv_LL = 4.8
            print(f"   ⚠ kVBase out of expected range — falling back to 4.8 kV")
        print(f"   Feeder kV (auto, L-L): {system_kv_LL:.4f} kV  "
              f"[Bus 701 kVBase raw = {_raw_kv:.4f} kV]")

        for s in scenarios:
            # --- Clean slate: recompile each scenario so `New` works without duplicates ---
            dss.Command('Clear')
            dss.Command(f'Compile "{dss_file}"')

            # Kill the main grid Vsource(s)
            dss.Vsources.First()
            for _ in range(dss.Vsources.Count()):
                dss.Command(f'Edit Vsource.{dss.Vsources.Name()} enabled=no')
                dss.Vsources.Next()

            # Disable all loads, capacitors, transformers
            dss.Loads.First()
            for _ in range(dss.Loads.Count()):
                dss.Command(f'Load.{dss.Loads.Name()}.enabled=no')
                dss.Loads.Next()
            dss.Capacitors.First()
            for _ in range(dss.Capacitors.Count()):
                dss.Command(f'Capacitor.{dss.Capacitors.Name()}.enabled=no')
                dss.Capacitors.Next()
            dss.Transformers.First()
            for _ in range(dss.Transformers.Count()):
                dss.Command(f'Transformer.{dss.Transformers.Name()}.enabled=no')
                dss.Transformers.Next()

            # Open ALL line terminals
            dss.Lines.First()
            for _ in range(dss.Lines.Count()):
                dss.Command(f'Open Line.{dss.Lines.Name()} terminal=1')
                dss.Command(f'Open Line.{dss.Lines.Name()} terminal=2')
                dss.Lines.Next()

            # --- Build up: enable only what MILP says is alive ---

            # 1. Determine island membership from MILP live topology, assign DG roles:
            #    - Slack DG per island (largest capacity): Wye Vsource — grounds the island
            #    - Co-island DGs: Delta Vsource with subtransient impedance — injects power
            #      without creating a second ground reference (no ground loop)
            G_live_s = nx.Graph()
            current_faults = {f.lower() for f in fault_scenarios_dict[s]["faults"]}
            for line in all_lines:
                name = line['name']
                if name in current_faults:
                    continue
                if name in switch_names:
                    active = (pulp.value(x_BR[name, s]) or 0) > 0.5
                else:
                    active = (pulp.value(x_Line[name, s]) or 0) > 0.5
                if active:
                    G_live_s.add_edge(line['bus1'], line['bus2'])
            for tx in transformers:
                b1, b2 = tx['bus1'], tx['bus2']
                if ((pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5 and
                        (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5):
                    G_live_s.add_edge(b1, b2)
            for bus in nodes:
                if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5:
                    G_live_s.add_node(bus)

            live_dgs_s = [g for g in purchased_dgs if (pulp.value(s_N.get((g, s), 0)) or 0) > 0.5]
            bus_island_s = {}
            for comp in nx.connected_components(G_live_s):
                root = min(comp)
                for b in comp:
                    bus_island_s[b] = root
            island_to_dgs_s: dict = {}
            for g in live_dgs_s:
                if g in bus_island_s:
                    island_to_dgs_s.setdefault(bus_island_s[g], []).append(g)

            slack_dgs_s: set = set()
            nonslack_dgs_s: set = set()
            for dg_list in island_to_dgs_s.values():
                slack = max(dg_list, key=lambda g: pulp.value(S_Cap[g]) or 0)
                slack_dgs_s.add(slack)
                for g in dg_list:
                    if g != slack:
                        nonslack_dgs_s.add(g)

            # One GFM Vsource per island (slack DG) + GFL Generator per co-island DG.
            # Generator model=1 (constant PQ) injects the MILP-dispatched P/Q at each
            # non-slack bus without creating a second voltage reference or ground loop.
            for g in slack_dgs_s:
                _g_raw  = bus_phase_info.get(g, {}).get('kv_base', 2.771)
                if _g_raw < 3.5:
                    g_kv_LL = round(_g_raw * math.sqrt(3), 4)
                else:
                    g_kv_LL = round(_g_raw, 4)
                if not (0.05 <= g_kv_LL <= 15.0):
                    g_kv_LL = system_kv_LL
                dss.Command(
                    f'New Vsource.DG_{g} bus1={g} basekv={g_kv_LL:.4f} pu=1.0 phases=3 '
                    f'R1=1e-4 X1=1e-4 R0=1e-4 X0=1e-4 enabled=yes'
                )

            for g in nonslack_dgs_s:
                _g_raw  = bus_phase_info.get(g, {}).get('kv_base', 2.771)
                if _g_raw < 3.5:
                    g_kv_LL = round(_g_raw * math.sqrt(3), 4)
                else:
                    g_kv_LL = round(_g_raw, 4)
                if not (0.05 <= g_kv_LL <= 15.0):
                    g_kv_LL = system_kv_LL
                p_milp = sum((pulp.value(P_G[g, p, s]) or 0.0) for p in phases)
                q_milp = sum((pulp.value(Q_G[g, p, s]) or 0.0) for p in phases)
                dss.Command(
                    f'New Generator.DG_{g} bus1={g} phases=3 kv={g_kv_LL:.4f} '
                    f'kw={p_milp:.4f} kvar={q_milp:.4f} '
                    f'model=1 vminpu=0.0 vmaxpu=2.0 enabled=yes'
                )

            # 2. Close both terminals of lines MILP has energised (x_Line=1 or x_BR=1).
            #    Faulted lines stay open. Both terminals must be closed for current to flow.
            for line in all_lines:
                name = line['name']
                if name in current_faults:
                    continue
                if name in switch_names:
                    if (pulp.value(x_BR[name, s]) or 0) > 0.5:
                        dss.Command(f'Close Line.{name} terminal=1')
                        dss.Command(f'Close Line.{name} terminal=2')
                else:
                    if (pulp.value(x_Line[name, s]) or 0) > 0.5:
                        dss.Command(f'Close Line.{name} terminal=1')
                        dss.Command(f'Close Line.{name} terminal=2')

            # 3. Transformers where both endpoint buses are live
            for tx in transformers:
                b1, b2 = tx['bus1'], tx['bus2']
                if ((pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5 and
                        (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5):
                    dss.Command(f'Transformer.{tx["name"]}.enabled=yes')

            # 4. Loads on live buses
            # NOTE: iterate the Python `loads` list, NOT dss.Loads.First()/Next().
            # By this point every load has been disabled, and OpenDSS's Loads
            # iterator only traverses ENABLED elements — so the cursor freezes on
            # the last load and no loads ever get re-enabled (silent zero-load
            # validation). Enabling by name from our own list is robust.
            for lname in loads:
                bus = load_weights[lname]['bus']
                if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5:
                    dss.Command(f'Load.{lname}.enabled=yes')

            # 5. Capacitors per MILP decision
            for cap in capacitors:
                if (pulp.value(x_Cap[cap['name'], s]) or 0) > 0.5:
                    dss.Command(f'Capacitor.{cap["name"]}.enabled=yes')

            dss.Command('set controlmode=off')
            dss.Solution.Solve()

            # Compute shedding from bus status
            milp_shed_kw = sum(
                load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
                * (1 - (pulp.value(s_N.get((load_weights[l]['bus'], s), 0)) or 0))
                for l in loads
            )

            milp_u_vals = [pulp.value(U_N[i, 'a', s]) for i in nodes
                           if (pulp.value(s_N.get((i, s), 0)) or 0) > 0.9
                           and pulp.value(U_N[i, 'a', s]) is not None]
            milp_min_v = math.sqrt(min(milp_u_vals)) if milp_u_vals else 0.0

            if dss.Solution.Converged():
                total_losses_kw = dss.Circuit.Losses()[0] / 1000.0
                live_bus_names = {i for i in nodes if (pulp.value(s_N.get((i, s), 0)) or 0) > 0.5}
                live_mags = []
                n_energized = 0
                for bus_name in dss.Circuit.AllBusNames():
                    if bus_name.lower().strip() not in live_bus_names:
                        continue
                    dss.Circuit.SetActiveBus(bus_name)
                    pu_v = dss.Bus.puVmagAngle()
                    mags = [pu_v[i] for i in range(0, len(pu_v), 2) if 0.1 < pu_v[i] < 2.0]
                    if mags:
                        n_energized += 1
                        live_mags.extend(mags)
                opendss_min_v = min(live_mags) if live_mags else 0.0
                opendss_avg_v = (sum(live_mags) / len(live_mags)) if live_mags else 0.0
                opendss_max_v = max(live_mags) if live_mags else 0.0

                # Branch loading: max apparent power across all active lines (phases only).
                max_branch_loading_pct = 0.0
                max_loaded_branch = None
                dss.Lines.First()
                while True:
                    ln = dss.Lines.Name()
                    dss.Circuit.SetActiveElement(f'Line.{ln}')
                    pows  = dss.CktElement.Powers()
                    nc    = dss.CktElement.NumConductors()
                    n_ph  = min(3, nc)
                    s_ph  = [math.sqrt(pows[2*k]**2 + pows[2*k+1]**2) for k in range(n_ph)]
                    s_max = max(s_ph) if s_ph else 0.0
                    loading_pct = s_max / LINE_CAPACITY * 100.0
                    if loading_pct > max_branch_loading_pct:
                        max_branch_loading_pct = loading_pct
                        max_loaded_branch = ln
                    if not dss.Lines.Next():
                        break

                print(f"[{s}] Shedding: {milp_shed_kw:7.1f} kW, {milp_shed_kw*100/total_grid_kw:.2f}% | "
                      f"OpenDSS Loss: {total_losses_kw:6.2f} kW | MILP Min V: {milp_min_v:.4f} pu | "
                      f"OpenDSS Min V: {opendss_min_v:.4f} pu | OpenDSS Max V: {opendss_max_v:.4f} pu | "
                      f"Max branch: {max_branch_loading_pct:.1f}%")

                # --- Power-balance cross-check: does OpenDSS deliver what the MILP promised? ---
                # TotalPower() returns Vsource (GFM) power only; GFL Generators are PC elements
                # and must be read separately. P_DG = GFM + GFL; P_load from iterated loads.
                p_vsource_kw = -dss.Circuit.TotalPower()[0]
                p_gen_kw     = 0.0
                for g in nonslack_dgs_s:
                    dss.Circuit.SetActiveElement(f'Generator.DG_{g}')
                    pows      = dss.CktElement.Powers()
                    nc        = dss.CktElement.NumConductors()
                    p_gen_kw += -sum(pows[2*k] for k in range(min(3, nc)))
                opendss_gen_kw = p_vsource_kw + p_gen_kw

                p_load_kw = 0.0
                if dss.Loads.First():
                    while True:
                        dss.Circuit.SetActiveElement(f'Load.{dss.Loads.Name()}')
                        pows      = dss.CktElement.Powers()
                        nc        = dss.CktElement.NumConductors()
                        p_load_kw += sum(pows[2*k] for k in range(min(3, nc)))
                        if not dss.Loads.Next():
                            break
                opendss_served_kw = p_load_kw
                milp_restored_kw  = total_grid_kw - milp_shed_kw
                bal_rel = (abs(opendss_served_kw - milp_restored_kw) / milp_restored_kw
                           if milp_restored_kw > 1.0 else 0.0)
                buses_ok = (n_energized == len(live_bus_names))
                balance_ok = (bal_rel <= 0.10) and buses_ok
                if not balance_ok:
                    print(f"   ⚠ BALANCE CHECK [{s}]: OpenDSS served {opendss_served_kw:7.1f} kW vs "
                          f"MILP restored {milp_restored_kw:7.1f} kW (Δ={bal_rel*100:5.1f}%) | "
                          f"energized {n_energized}/{len(live_bus_names)} live buses")
                else:
                    print(f"   ✓ balance [{s}]: served {opendss_served_kw:7.1f} kW ≈ MILP "
                          f"{milp_restored_kw:7.1f} kW (Δ={bal_rel*100:4.1f}%), "
                          f"{n_energized}/{len(live_bus_names)} buses energized")

                # --- Reactive-headroom check: does the slack Vsource exceed the machine Q limit? ---
                # MILP caps Q_total <= 0.6 * S_Cap (per-phase 0.6*S_cap/3, summed over 3 phases).
                # The OpenDSS Vsource has NO Q limit, so we read its actual Q and compare. In a
                # multi-DG island the slack also covers the omitted non-slack DGs, so this is where
                # an exceedance would surface.
                dg_dispatch = {}
                for g in slack_dgs_s:
                    dss.Circuit.SetActiveElement(f'Vsource.DG_{g}')
                    powers = dss.CktElement.Powers()   # [P1,Q1, P2,Q2, ...] kW/kvar per phase
                    # Power flows INTO the element from the bus, so a delivering source reads
                    # negative — negate to get power injected into the island.
                    q_delivered = -sum(powers[k] for k in range(1, len(powers), 2))
                    p_delivered = -sum(powers[k] for k in range(0, len(powers), 2))
                    q_limit = 0.6 * (pulp.value(S_Cap[g]) or 0)   # MILP per-machine reactive cap (kvar) = 0.6 × S_cap
                    within = abs(q_delivered) <= q_limit * (1.0 + Q_LIMIT_TOL_FRAC)
                    flag = "✓ within limit" if within else "⚠ EXCEEDS LIMIT"
                    print(f"   [{s}] Slack DG {g}: P={p_delivered:7.1f} kW  "
                          f"Q={q_delivered:7.1f} kvar  (limit ±{q_limit:.1f} kvar)  {flag}")
                    # Island-level Q limit: sum over all DGs in the same island
                    island_root = bus_island_s.get(g, g)
                    island_dgs  = island_to_dgs_s.get(island_root, [g])
                    island_q_limit = sum(0.6 * float(dg_sizes.get(d, dg_sizes.get(str(d), 0)))
                                         for d in island_dgs)
                    s_delivered = math.sqrt(p_delivered**2 + q_delivered**2)
                    s_cap = float(dg_sizes.get(g, dg_sizes.get(str(g), 0)))
                    dg_dispatch[g] = {
                        'p_delivered_kw':      round(p_delivered, 2),
                        'q_delivered_kvar':    round(q_delivered, 2),
                        's_delivered_kva':     round(s_delivered, 2),
                        's_cap_kva':           round(s_cap, 2),
                        's_loading_pct':       round(s_delivered / s_cap * 100.0, 2) if s_cap > 0 else None,
                        'q_limit_kvar':        round(q_limit, 2),
                        'within_limit':        within,
                        'island_q_limit_kvar': round(island_q_limit, 2),
                        'island_dgs':          sorted(island_dgs),
                    }

                scenario_results[s] = {
                    'shed': milp_shed_kw,
                    'opendss_loss_kw': total_losses_kw,
                    'opendss_min_v': opendss_min_v,
                    'opendss_avg_v': opendss_avg_v,
                    'opendss_max_v': opendss_max_v,
                    'milp_min_v': milp_min_v,
                    'dg_dispatch': dg_dispatch,
                    'opendss_served_kw':  round(opendss_served_kw, 2),
                    'milp_restored_kw':   round(milp_restored_kw, 2),
                    'balance_rel_error':  round(bal_rel, 4),
                    'buses_energized':    f"{n_energized}/{len(live_bus_names)}",
                    'balance_ok':         balance_ok,
                    'max_branch_loading_pct': round(max_branch_loading_pct, 2),
                    'max_loaded_branch':       max_loaded_branch,
                }
            else:
                print(f"[{s}] ❌ OpenDSS failed to converge.")
                scenario_results[s] = {'shed': milp_shed_kw}

            # (No teardown needed — next iteration starts with Clear + Compile)

        # plot_saa_results(purchased_dgs, dg_sizes, scenario_results, pos, SUB_BUS)

        print("\n💾 EXPORTING MASTER RESULTS...")
        c_inv = sum(COST_DG_FIXED + COST_DG_PER_KW * 0.8 * dg_sizes[g] for g in purchased_dgs)

        output_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scenarios_analyzed": len(scenarios),
                "solve_time_seconds": round(solve_duration, 2)  # <-- ADD THIS
            },
            "financials": {
                "objective_value_lifetime": pulp.value(saa_model.objective),
                "best_bound_theoretical":   best_bound,
                "mip_gap_achieved":         mip_gap_achieved,
                "total_investment_cost_CINV": c_inv,
                "capex_present_value": pulp.value(cost_dg),
                "om_present_value": pulp.value(maintenance_pv),
                "expected_disaster_cost_raw": pulp.value(expected_disaster_cost)
            },
            "constants": {
                "COST_DG_FIXED":               COST_DG_FIXED,
                "COST_DG_PER_KW":              COST_DG_PER_KW,
                "COST_DG_OM_PER_KW_YR":        COST_DG_OM_PER_KW_YR,
                "COST_GEN_PER_KWH":            COST_GEN_PER_KWH,
                "HOURS_NORMAL":                HOURS_NORMAL,
                "MAX_INVESTMENT_BUDGET":       budget_max,
                "PLANNING_YEARS":              PLANNING_YEARS,
                "INTEREST_RATE":               INTEREST_RATE,
                "PWF":                         PWF,
                "MIP_GAP":                     mip_gap,
                "LAMBDA_RISK":                 LAMBDA_RISK,
                "ALPHA":                       ALPHA,
                "VOLL_TIER_1":    VOLL_TIER_1,
                "VOLL_TIER_2":    VOLL_TIER_2,
                "VOLL_CRITICAL":  VOLL_CRITICAL,
                "CRITICAL_BUSES": sorted(CRITICAL_BUSES),
                "CONDITIONAL_EVENT_OCCURRENCE": conditional_occ,
            },
            "dg_placement": {
                "purchased_buses": purchased_dgs,
                "sizes_kva": dg_sizes
            },
            "metrics_storm_stress_test": scenario_results
        }

        with open(export_filename, "w") as outfile:
            json.dump(output_data, outfile, indent=4)

        print(f"✅ Data successfully saved to {export_filename}")

    else:
        print("❌ Failed to converge within time limit.")


if __name__ == "__main__":
    import sys

    # -----------------------------------------------------------------------
    # Orchestrator (saa_convergence_study.py) calls:
    #   python main_initial.py <iter> <LABEL> <MIPGap> <Lambda> <Alpha> [Budget] [CondOcc]
    # Standalone: all values fall back to the _CFG_* block / module constants.
    # The optional [Budget] (#7 Pareto) and [CondOcc] (#6 annualized-vs-conditional)
    # args default to MAX_INVESTMENT_BUDGET / CONDITIONAL_EVENT_OCCURRENCE, so
    # omitting them reproduces today's behaviour exactly.
    # -----------------------------------------------------------------------
    iteration   = sys.argv[1]        if len(sys.argv) > 1 else _CFG_ITERATION
    LABEL       = sys.argv[2]        if len(sys.argv) > 2 else _CFG_LABEL
    mip_gap     = float(sys.argv[3]) if len(sys.argv) > 3 else _CFG_MIP_GAP
    lambda_risk = float(sys.argv[4]) if len(sys.argv) > 4 else _CFG_LAMBDA
    alpha       = float(sys.argv[5]) if len(sys.argv) > 5 else _CFG_ALPHA
    budget_max  = float(sys.argv[6]) if len(sys.argv) > 6 else MAX_INVESTMENT_BUDGET
    conditional_occ = float(sys.argv[7]) if len(sys.argv) > 7 else CONDITIONAL_EVENT_OCCURRENCE

    if LABEL:
        json_filename   = f"saa_scenario_{LABEL}_{iteration}.json"
        export_filename = f"master_dg_placements_{LABEL}_{iteration}.json"
    else:
        json_filename   = f"saa_scenario_pga030N30_{iteration}.json"
        export_filename = f"master_dg_placements_test_{iteration}.json"

    if not os.path.exists(json_filename):
        print(f"❌ ERROR: Cannot find {json_filename}. Run generatefault.py first!")
        exit()

    with open(json_filename, "r") as infile:
        dynamic_fault_scenarios = json.load(infile)

    case_input = {k: v for k, v in dynamic_fault_scenarios.items() if k not in ('Normal_Operation', 'metadata')}
    print(f"📥 Loaded {len(case_input)} disaster scenarios for iteration {iteration}.")
    print(f"   lambda={lambda_risk} | alpha={alpha} | MIPGap={mip_gap*100:.1f}% | "
          f"budget=${budget_max:,.0f} | cond_occ={conditional_occ}")

    solve_stochastic_saa(case_input, export_filename, mip_gap, lambda_risk, alpha,
                         budget_max, conditional_occ)