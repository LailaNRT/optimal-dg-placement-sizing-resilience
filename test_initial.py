import opendssdirect as dss
import os
import pulp
import math
import networkx as nx
import json
from datetime import datetime
from collections import Counter
import sys

# ==========================================
# 1. LOAD MASTER PLAN DATA & CONSTANTS
# ==========================================

# If a number is passed, use it. Otherwise default to "15".
iteration = sys.argv[1] if len(sys.argv) > 1 else "1"
LABEL     = sys.argv[2] if len(sys.argv) > 2 else ""

# Dynamic master filename — uses LABEL when provided by orchestrator
if LABEL:
    json_filename_master = f"master_dg_placements_{LABEL}_{iteration}.json"
else:
    # Original hardcoded filename (kept for standalone backward-compatibility)
    # json_filename_master = f"master_dg_placements_initial.json"
    json_filename_master = f"master_dg_placements_pga030N30L000Y15R010_rho00_sig50_{iteration}.json"

if not os.path.exists(json_filename_master):
    print(f"❌ ERROR: Cannot find {json_filename_master}.")
    exit()

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

dss_file = "ieee37.dss" # Ensure this matches your target grid

with open(json_filename_master, "r") as infile:
    master_data = json.load(infile)

# Extract Constants
C = master_data.get("constants", {})
COST_GEN_PER_KWH     = C.get("COST_GEN_PER_KWH", 0.45)
COST_DG_OM_PER_KW_YR = C.get("COST_DG_OM_PER_KW_YR", 9.33)
VOLL_TIER_1    = C.get("VOLL_TIER_1",   3.3)
VOLL_TIER_2    = C.get("VOLL_TIER_2",   21.8)
VOLL_CRITICAL  = C.get("VOLL_CRITICAL", 100.0)
CRITICAL_BUSES = set(C.get("CRITICAL_BUSES", ["712", "729", "738", "720"]))
PLANNING_YEARS = C.get("PLANNING_YEARS", 15)
INTEREST_RATE  = C.get("INTEREST_RATE", 0.1)
CRF = INTEREST_RATE * (1 + INTEREST_RATE)**PLANNING_YEARS / ((1 + INTEREST_RATE)**PLANNING_YEARS - 1)
# MIP_GAP = C.get("MIP_GAP", 0.001)   # mirrors whatever main_initial.py used
MIP_GAP = 0.001 
EXPECTED_EVENTS_IN_LIFESPAN = C.get("CONDITIONAL_EVENT_OCCURRENCE", C.get("EXPECTED_EVENTS_IN_LIFESPAN", 1))
LINE_CAPACITY = 500.0
TX_CAPACITY   = 500.0 / 3.0   # XFM1 rated 500 kVA total 3-phase → per-phase limit

# Q-dispatch within_limit tolerance -- matches Q_TOL_FRAC in checkPQbranch.py
# so the "within limit" flags written here don't flag trivial solver noise
# that checkPQbranch.py's own Q_LIMIT_ODS check wouldn't.
Q_LIMIT_TOL_FRAC = 0.010   # 1.0% of q_limit

# Feeder base voltage for the LinDistFlow pu² normalization. IEEE 37-bus is
# 4.8 /sqrt(3) kV L-N.
V_BASE_LN_KV = 4.8 / math.sqrt(3)
VDROP_DENOM  = (V_BASE_LN_KV ** 2) * 1000.0   

# Extract Placements
C_INV = master_data["financials"].get("total_investment_cost_CINV", 0)
purchased_dgs = master_data["dg_placement"]["purchased_buses"]
dg_sizes = master_data["dg_placement"].get("sizes_kw") or master_data["dg_placement"]["sizes_kva"]

# ==========================================
# 2. BASE GRID EXTRACTION
# ==========================================
def refresh_base_grid_state():
    """Extracts the base-grid parameters ONCE without any faults applied."""
    global nodes, loads, load_weights, switchable_lines, switch_names
    global all_lines, transformers, capacitors, total_grid_kw
    global SUB_BUS, bus_phase_info, candidate_buses
    
    dss.Command('Clear')
    dss.Command(f'Compile "{dss_file}"')
    
    dss.Vsources.First()
    SUB_BUS = dss.CktElement.BusNames()[0].split('.')[0].lower().strip()
    
    bus_phase_info = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        bus_phase_info[bus_name.lower().strip()] = {
            'num_phases': dss.Bus.NumNodes(),
            'nodes': list(dss.Bus.Nodes()),
            'kv_base': dss.Bus.kVBase()
        }

    nodes = [bus.lower() for bus in dss.Circuit.AllBusNames()]
    
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
        
        kw_per_phase = dss.Loads.kW() / len(present_phases) if present_phases else 0
        kvar_per_phase = dss.Loads.kvar() / len(present_phases) if present_phases else 0
        
        loads.append(lname)
        seen_loads.add(lname)
        load_weights[lname] = {
            'bus': bname, 'phases': present_phases,
            'kW_phase': kw_per_phase, 'kvar_phase': kvar_per_phase
        }
        dss.Loads.Next()

    total_grid_kw = sum([load_weights[l]['kW_phase'] * len(load_weights[l]['phases']) for l in loads])
        
    switchable_lines = []
    seen_switches = set()
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        line_name = dss.Lines.Name().lower()
        is_switch = (dss.Lines.IsSwitch() if hasattr(dss.Lines, 'IsSwitch') else False) or line_name.startswith('sw')
            
        if is_switch and line_name not in seen_switches:
            b1 = dss.Lines.Bus1().split('.')[0].lower().strip()
            b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
            switchable_lines.append({'name': line_name, 'bus1': b1, 'bus2': b2})
            seen_switches.add(line_name)
        dss.Lines.Next()
    switch_names = [sw['name'] for sw in switchable_lines]

    all_lines = []
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        name = dss.Lines.Name().lower()
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
                        p_i = present_phases[i]
                        p_j = present_phases[j]
                        r_3x3[p_i][p_j] = rmat_flat[idx]
                        x_3x3[p_i][p_j] = xmat_flat[idx]
                    idx += 1
                    
        all_lines.append({'name': name, 'bus1': b1, 'bus2': b2, 'r_matrix': r_3x3, 'x_matrix': x_3x3, 'phases': present_phases})
        dss.Lines.Next()

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
        transformers.append({'name': t_name, 'bus1': b1, 'bus2': b2, 'a_ratio': a_ratio, 'phases': sec_phases})
        dss.Transformers.Next()

    capacitors = []
    dss.Capacitors.First()
    for _ in range(dss.Capacitors.Count()):
        c_name = dss.Capacitors.Name().lower().strip()
        bname_full = dss.CktElement.BusNames()[0].lower()
        bname = bname_full.split('.')[0].strip()
        cap_nodes = bname_full.split('.')[1:] if '.' in bname_full else ['1','2','3']
        phase_map = {'1': 'a', '2': 'b', '3': 'c'}
        cap_phases = [phase_map.get(n) for n in cap_nodes if n in phase_map]
        kvar_ph = dss.Capacitors.kvar() / len(cap_phases) if cap_phases else 0
        capacitors.append({'name': c_name, 'bus': bname, 'phases': cap_phases, 'kvar_phase': kvar_ph})
        dss.Capacitors.Next()

    regulator_buses = set()
    dss.RegControls.First()
    for _ in range(dss.RegControls.Count()):
        tx_name = dss.RegControls.Transformer()
        dss.Transformers.Name(tx_name)
        reg_buses = dss.CktElement.BusNames()
        if len(reg_buses) > 1: regulator_buses.add(reg_buses[1].split('.')[0].lower().strip())
        dss.RegControls.Next()

    _degree = Counter()
    for _l in all_lines:
        _degree[_l['bus1']] += 1
        _degree[_l['bus2']] += 1
    for _t in transformers:
        _degree[_t['bus1']] += 1
        _degree[_t['bus2']] += 1

    candidate_buses = [
        b for b in nodes
        if b != SUB_BUS and b != '1'
        and bus_phase_info.get(b, {}).get('num_phases', 0) == 3
        and _degree.get(b, 0) > 1
        and b not in regulator_buses
    ]

    # Bus adjacency lookups -- built once; the line/transformer topology is
    # static across all 500 scenarios (faults are imposed via constraints,
    # not by removing elements), so these are reused by every evaluation.
    global lines_into_bus, lines_out_of_bus, tx_into_bus, tx_out_of_bus
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

# ==========================================
# 3. SINGLE SCENARIO EVALUATOR
# ==========================================
def evaluate_single_scenario(scenario_name, current_faults, scenario_duration, purchased_dgs, dg_sizes, MIP_GAP):
    """Runs a lightning-fast MILP for a single scenario locking DGs to master sizes, and validates with OpenDSS."""
    s = 'eval' 
    phases = ['a', 'b', 'c']
    N_nodes = len(nodes)
    current_faults = {f.lower() for f in current_faults}

    saa_model = pulp.LpProblem(f"Eval_{scenario_name}", pulp.LpMinimize)

    # First-stage DG placement is FIXED by the master plan, so x_G / P_Cap are
    # plain constants here, not variables pinned by equality constraints. This
    # also makes the P_Cap * s_N product exactly linear (no big-M linearization).
    x_G   = {c: (1 if c in purchased_dgs else 0) for c in candidate_buses}
    P_Cap = {c: (float(dg_sizes.get(c, dg_sizes.get(str(c), 0))) if c in purchased_dgs else 0.0)
             for c in candidate_buses}

    s_N = pulp.LpVariable.dicts("s_N", ((i, s) for i in nodes), cat='Binary')
    x_BR = pulp.LpVariable.dicts("x_BR", ((sw, s) for sw in switch_names), cat='Binary')
    x_Cap = pulp.LpVariable.dicts("x_Cap", ((cap['name'], s) for cap in capacitors), cat='Binary')

    M_Power = 1000.0
    M_Volt = 1.5

    P_G = pulp.LpVariable.dicts("P_G", ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases), lowBound=0, cat='Continuous')
    Q_G = pulp.LpVariable.dicts("Q_G", ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases), lowBound=-300, upBound=300, cat='Continuous')
    U_N = pulp.LpVariable.dicts("U_N", ((i, p, s) for i in nodes for p in phases), cat='Continuous')
    P_Line = pulp.LpVariable.dicts("P_Line", ((ln['name'], p, s) for ln in all_lines for p in phases), lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')
    Q_Line = pulp.LpVariable.dicts("Q_Line", ((ln['name'], p, s) for ln in all_lines for p in phases), lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')

    # x_Line is declared Continuous [0,1] but is integral in every feasible
    # solution: faulted lines are fixed to 0, switch lines are forced equal to
    # the Binary x_BR, and hardwired lines equal the AND of s_N[bus1]/s_N[bus2]
    # (which the BusProp propagation constraint forces to be equal Binary
    # values). It always resolves to an exact 0/1 "line conducting" status.
    x_Line = pulp.LpVariable.dicts("x_Line", ((ln['name'], s) for ln in all_lines), lowBound=0, upBound=1, cat='Continuous')
    f_Line = pulp.LpVariable.dicts("f_Line", ((ln['name'], s) for ln in all_lines), cat='Continuous')
    z_Vir  = pulp.LpVariable.dicts("z_Vir",  ((i, s) for i in nodes), cat='Binary')
    f_Vir  = pulp.LpVariable.dicts("f_Vir",  ((i, s) for i in nodes), lowBound=0,  cat='Continuous')
    f_Tx   = pulp.LpVariable.dicts("f_Tx",   ((tx['name'], s) for tx in transformers), cat='Continuous')
    P_Tx   = pulp.LpVariable.dicts("P_Tx",   ((tx['name'], p, s) for tx in transformers for p in phases), cat='Continuous')
    Q_Tx   = pulp.LpVariable.dicts("Q_Tx",   ((tx['name'], p, s) for tx in transformers for p in phases), cat='Continuous')

    loads_at_node = {i: [] for i in nodes}
    for l in loads: loads_at_node[load_weights[l]['bus']].append(l)

    f_priority = {}
    for l in loads:
        bus = load_weights[l]['bus']
        total_kw = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
        if bus in CRITICAL_BUSES:
            f_priority[l] = VOLL_CRITICAL       # hospital / emergency services
        elif total_kw >= 50.0:
            f_priority[l] = VOLL_TIER_2         # commercial / industrial
        else:
            f_priority[l] = VOLL_TIER_1         # residential

    fuel_cost = pulp.lpSum([P_G[c, p, s] * COST_GEN_PER_KWH * scenario_duration for c in candidate_buses for p in phases])
    # Explicit shed variable (see main_initial.py): keeps the objective free of
    # the decision-independent shed-everything constant that PuLP drops from the
    # solver's view. shed is pinned to P_dem*(1 - s_N) by the positive penalty,
    # so opt_cost is unchanged.
    shed = pulp.LpVariable.dicts("shed", ((l, s) for l in loads), lowBound=0, cat='Continuous')
    for l in loads:
        nph = len(load_weights[l]['phases'])
        saa_model += shed[l, s] >= load_weights[l]['kW_phase'] * nph * (1 - s_N[load_weights[l]['bus'], s])
    shedding_penalty = pulp.lpSum([f_priority[l] * shed[l, s] * scenario_duration for l in loads])
    saa_model += shedding_penalty + fuel_cost

    sqrt3_2 = math.sqrt(3) / 2.0
    alpha_real = {'a': {'a': 1.0, 'b': -0.5, 'c': -0.5}, 'b': {'a': -0.5, 'b': 1.0, 'c': -0.5}, 'c': {'a': -0.5, 'b': -0.5, 'c': 1.0}}
    alpha_imag = {'a': {'a': 0.0, 'b': -sqrt3_2, 'c': sqrt3_2}, 'b': {'a': sqrt3_2, 'b': 0.0, 'c': -sqrt3_2}, 'c': {'a': -sqrt3_2, 'b': sqrt3_2, 'c': 0.0}}
    U_max, U_min = 1.05 ** 2, 0.95 ** 2

    saa_model += s_N[SUB_BUS, s] == 0
    for p in phases:
        saa_model += U_N[SUB_BUS, p, s] == 0
        saa_model += P_G['MAIN_SUBSTATION', p, s] == 0
        saa_model += Q_G['MAIN_SUBSTATION', p, s] == 0

    for c in candidate_buses:
        for p in phases:
            if x_G[c]:
                # Installed DG anchors island voltage when its bus is alive
                saa_model += U_N[c,p,s] >= 1.0 - M_Volt * (1 - s_N[c,s])
                saa_model += U_N[c,p,s] <= 1.0 + M_Volt * (1 - s_N[c,s])
            saa_model += P_G[c,p,s] <= 0.8 * P_Cap[c] / 3.0
            saa_model += P_G[c,p,s] <= M_Power * s_N[c,s]
            saa_model += Q_G[c,p,s] <= 0.6 * (P_Cap[c] / 3.0)
            saa_model += Q_G[c,p,s] >= -0.6 * (P_Cap[c] / 3.0)
            saa_model += Q_G[c,p,s] <= M_Power * s_N[c,s]
            saa_model += Q_G[c,p,s] >= -M_Power * s_N[c,s]
        # Exact product (P_Cap is constant): capacity counts only when bus is alive
        saa_model += pulp.lpSum([P_G[c,p,s] for p in phases]) <= P_Cap[c] * s_N[c,s]

    for line in all_lines:
        name, b1, b2 = line['name'], line['bus1'], line['bus2']
        if name in current_faults:
            saa_model += x_Line[name,s] == 0
        elif name in switch_names:
            saa_model += x_Line[name,s] == x_BR[name,s]
        else:
            saa_model += x_Line[name,s] <= s_N[b1,s]
            saa_model += x_Line[name,s] <= s_N[b2,s]
            saa_model += x_Line[name,s] >= s_N[b1,s] + s_N[b2,s] - 1

    # Virtual root only allowed at buses where a DG is actually installed
    for i in nodes:
        if i != SUB_BUS and i in candidate_buses and x_G[i]:
            saa_model += z_Vir[i,s] <= s_N[i,s]
        else:
            saa_model += z_Vir[i,s] == 0

    saa_model += (
        pulp.lpSum([x_Line[ln['name'],s] for ln in all_lines]) +
        pulp.lpSum([s_N[tx['bus1'],s] for tx in transformers]) +
        pulp.lpSum([z_Vir[i,s] for i in nodes])
        == pulp.lpSum([s_N[i,s] for i in nodes]),
        f"SpanningForest_{s}"
    )

    for i in nodes:
        saa_model += f_Vir[i,s] <= N_nodes * z_Vir[i,s]
        saa_model += (f_Vir[i,s]
                      + pulp.lpSum([f_Line[n,s] for n in lines_into_bus[i]])
                      + pulp.lpSum([f_Tx[n,s]   for n in tx_into_bus[i]])
                      - pulp.lpSum([f_Line[n,s] for n in lines_out_of_bus[i]])
                      - pulp.lpSum([f_Tx[n,s]   for n in tx_out_of_bus[i]])
                      == s_N[i,s])

    for line in all_lines:
        saa_model += f_Line[line['name'],s] <= N_nodes * x_Line[line['name'],s]
        saa_model += f_Line[line['name'],s] >= -N_nodes * x_Line[line['name'],s]

    for tx in transformers:
        saa_model += f_Tx[tx['name'],s] <=  N_nodes * s_N[tx['bus1'],s]
        saa_model += f_Tx[tx['name'],s] >= -N_nodes * s_N[tx['bus1'],s]

    for line in all_lines:
        name = line['name']
        if name in current_faults or name in switch_names:
            continue
        saa_model += s_N[line['bus1'], s] == s_N[line['bus2'], s], f"BusProp_{name}_{s}"

    for tx in transformers:
        saa_model += s_N[tx['bus1'], s] == s_N[tx['bus2'], s], f"TxProp_{tx['name']}_{s}"

    for i in nodes:
        for p in phases:
            saa_model += U_N[i,p,s] <= s_N[i,s] * U_max
            saa_model += U_N[i,p,s] >= s_N[i,s] * U_min

    for sw in switchable_lines:
        saa_model += x_BR[sw['name'],s] <= s_N[sw['bus1'],s]
        saa_model += x_BR[sw['name'],s] <= s_N[sw['bus2'],s]

    for cap in capacitors: saa_model += x_Cap[cap['name'],s] <= s_N[cap['bus'],s]

    for p in phases:
        for i in nodes:
            p_in, p_out = pulp.lpSum([P_Line[ln,p,s] for ln in lines_into_bus[i]]), pulp.lpSum([P_Line[ln,p,s] for ln in lines_out_of_bus[i]])
            q_in, q_out = pulp.lpSum([Q_Line[ln,p,s] for ln in lines_into_bus[i]]), pulp.lpSum([Q_Line[ln,p,s] for ln in lines_out_of_bus[i]])
            p_tx_in  = pulp.lpSum([P_Tx[n,p,s] for n in tx_into_bus[i]])
            p_tx_out = pulp.lpSum([P_Tx[n,p,s] for n in tx_out_of_bus[i]])
            q_tx_in  = pulp.lpSum([Q_Tx[n,p,s] for n in tx_into_bus[i]])
            q_tx_out = pulp.lpSum([Q_Tx[n,p,s] for n in tx_out_of_bus[i]])

            p_served = sum(load_weights[l]['kW_phase'] for l in loads_at_node[i] if p in load_weights[l]['phases']) * s_N[i,s]
            q_served = sum(load_weights[l]['kvar_phase'] for l in loads_at_node[i] if p in load_weights[l]['phases']) * s_N[i,s]

            p_dg = P_G['MAIN_SUBSTATION',p,s] if i == SUB_BUS else (P_G[i,p,s] if i in candidate_buses else 0)
            q_dg = Q_G['MAIN_SUBSTATION',p,s] if i == SUB_BUS else (Q_G[i,p,s] if i in candidate_buses else 0)
            q_cap = pulp.lpSum([x_Cap[cap['name'],s] * cap['kvar_phase'] for cap in capacitors if cap['bus'] == i and p in cap['phases']])

            saa_model += p_in + p_dg + p_tx_in - p_out - p_tx_out - p_served <= M_Power*(1-s_N[i,s])
            saa_model += p_in + p_dg + p_tx_in - p_out - p_tx_out - p_served >= -M_Power*(1-s_N[i,s])
            saa_model += q_in + q_dg + q_cap + q_tx_in - q_out - q_tx_out - q_served <= M_Power*(1-s_N[i,s])
            saa_model += q_in + q_dg + q_cap + q_tx_in - q_out - q_tx_out - q_served >= -M_Power*(1-s_N[i,s])

        for line in all_lines:
            name, b1, b2 = line['name'], line['bus1'], line['bus2']
            if p not in line['phases']:
                saa_model += P_Line[name,p,s] == 0
                saa_model += Q_Line[name,p,s] == 0
                continue

            v_drop = 0
            for m in line['phases']:
                r_t = alpha_real[p][m] * line['r_matrix'][p][m] - alpha_imag[p][m] * line['x_matrix'][p][m]
                x_t = alpha_real[p][m] * line['x_matrix'][p][m] + alpha_imag[p][m] * line['r_matrix'][p][m]
                v_drop += 2 * (r_t * P_Line[name,m,s] + x_t * Q_Line[name,m,s]) / VDROP_DENOM

            if name in current_faults:
                saa_model += P_Line[name,p,s] == 0
                saa_model += Q_Line[name,p,s] == 0
            elif name in switch_names:
                saa_model += U_N[b1,p,s] - U_N[b2,p,s] <= v_drop + M_Volt*(1-x_BR[name,s])
                saa_model += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop - M_Volt*(1-x_BR[name,s])
                saa_model += P_Line[name,p,s] <= M_Power*x_BR[name,s]
                saa_model += P_Line[name,p,s] >= -M_Power*x_BR[name,s]
                saa_model += Q_Line[name,p,s] <= M_Power*x_BR[name,s]
                saa_model += Q_Line[name,p,s] >= -M_Power*x_BR[name,s]
            else:
                saa_model += U_N[b1,p,s] - U_N[b2,p,s] <= v_drop + M_Volt*(1-x_Line[name,s])
                saa_model += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop - M_Volt*(1-x_Line[name,s])
                saa_model += P_Line[name,p,s] <= M_Power*x_Line[name,s]
                saa_model += P_Line[name,p,s] >= -M_Power*x_Line[name,s]
                saa_model += Q_Line[name,p,s] <= M_Power*x_Line[name,s]
                saa_model += Q_Line[name,p,s] >= -M_Power*x_Line[name,s]

        for tx in transformers:
            b1, b2 = tx['bus1'], tx['bus2']
            if p not in tx['phases']: continue
            a_sq = tx['a_ratio'][p] ** 2
            t_live = s_N[b1,s] + s_N[b2,s]
            saa_model += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] <= M_Volt*(2-t_live)
            saa_model += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] >= -M_Volt*(2-t_live)

    # Line thermal capacity (phase-independent constraints — added once per
    # line/phase, outside the p-loop where they were previously triplicated)
    sqrt3 = math.sqrt(3)
    for line in all_lines:
        name = line['name']
        for ph in line['phases']:
            saa_model += (sqrt3*P_Line[name,ph,s] + Q_Line[name,ph,s]) <= 2*LINE_CAPACITY
            saa_model += (sqrt3*P_Line[name,ph,s] + Q_Line[name,ph,s]) >= -2*LINE_CAPACITY
            saa_model += (sqrt3*P_Line[name,ph,s] - Q_Line[name,ph,s]) <= 2*LINE_CAPACITY
            saa_model += (sqrt3*P_Line[name,ph,s] - Q_Line[name,ph,s]) >= -2*LINE_CAPACITY
            saa_model += P_Line[name, ph, s] <=  LINE_CAPACITY
            saa_model += P_Line[name, ph, s] >= -LINE_CAPACITY
            saa_model += Q_Line[name, ph, s] <=  LINE_CAPACITY
            saa_model += Q_Line[name, ph, s] >= -LINE_CAPACITY
            
    # Transformer power flow: M-gate + octagonal thermal capacity
    for tx in transformers:
        name, b1 = tx['name'], tx['bus1']
        for ph in tx['phases']:
            saa_model += P_Tx[name,ph,s] <=  M_Power * s_N[b1,s]
            saa_model += P_Tx[name,ph,s] >= -M_Power * s_N[b1,s]
            saa_model += Q_Tx[name,ph,s] <=  M_Power * s_N[b1,s]
            saa_model += Q_Tx[name,ph,s] >= -M_Power * s_N[b1,s]
            saa_model += (sqrt3*P_Tx[name,ph,s] + Q_Tx[name,ph,s]) <=  2*TX_CAPACITY
            saa_model += (sqrt3*P_Tx[name,ph,s] + Q_Tx[name,ph,s]) >= -2*TX_CAPACITY
            saa_model += (sqrt3*P_Tx[name,ph,s] - Q_Tx[name,ph,s]) <=  2*TX_CAPACITY
            saa_model += (sqrt3*P_Tx[name,ph,s] - Q_Tx[name,ph,s]) >= -2*TX_CAPACITY
            saa_model += P_Tx[name,ph,s] <=  TX_CAPACITY
            saa_model += P_Tx[name,ph,s] >= -TX_CAPACITY
            saa_model += Q_Tx[name,ph,s] <=  TX_CAPACITY
            saa_model += Q_Tx[name,ph,s] >= -TX_CAPACITY

    try:
        saa_model.solve(pulp.GUROBI_CMD(msg=False, options=[
            ("MIPGap",       MIP_GAP),  # mirrors whatever main_initial used (read from master JSON)
            ("TimeLimit",    60),       # 60-s cap per sub-scenario — sufficient for smaller problems
        ]))
    except TypeError:
        # Catches the PuLP bug when Gurobi times out without finding an incumbent solution
        return None
    except Exception:
        # Catches any other random solver crashes
        return None

    if pulp.LpStatus[saa_model.status] == 'Optimal':
        opt_cost = pulp.value(saa_model.objective)
        milp_shed_kw = sum(load_weights[l]['kW_phase'] * len(load_weights[l]['phases']) * (1 - (pulp.value(s_N.get((load_weights[l]['bus'], s), 0)) or 0)) for l in loads)

        # Track which loads the MILP restored (bus alive = load served)
        restored_loads = {
            l: 1 if (pulp.value(s_N.get((load_weights[l]['bus'], s), 0)) or 0) > 0.5 else 0
            for l in loads
        }

        # ==========================================
        # OPENDSS VALIDATION (Robust Islanding)
        # ==========================================
        dss.Command('Clear')
        dss.Command(f'Compile "{dss_file}"')
        
        dss.Vsources.First()
        for _ in range(dss.Vsources.Count()):
            dss.Command(f'Edit Vsource.{dss.Vsources.Name()} enabled=no')
            dss.Vsources.Next()

        dss.Loads.First()
        for _ in range(dss.Loads.Count()):
            dss.Command(f'Edit Load.{dss.Loads.Name()} enabled=no')
            dss.Loads.Next()
        dss.Capacitors.First()
        for _ in range(dss.Capacitors.Count()):
            dss.Command(f'Edit Capacitor.{dss.Capacitors.Name()} enabled=no')
            dss.Capacitors.Next()
        dss.Transformers.First()
        for _ in range(dss.Transformers.Count()):
            dss.Command(f'Edit Transformer.{dss.Transformers.Name()} enabled=no')
            dss.Transformers.Next()

        dss.Lines.First()
        for _ in range(dss.Lines.Count()):
            lname = dss.Lines.Name()
            dss.Command(f'Open Line.{lname} terminal=1')
            dss.Command(f'Open Line.{lname} terminal=2')
            dss.Lines.Next()

        G_live = nx.Graph()
        for line in all_lines:
            name = line['name']
            if name in current_faults: continue
            if name in switch_names:
                if (pulp.value(x_BR[name, s]) or 0) > 0.5: G_live.add_edge(line['bus1'], line['bus2'])
            else:
                if (pulp.value(x_Line[name, s]) or 0) > 0.5: G_live.add_edge(line['bus1'], line['bus2'])
        for tx in transformers:
            b1, b2 = tx['bus1'], tx['bus2']
            if ((pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5 and (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5):
                G_live.add_edge(b1, b2)
        for bus in nodes:
            if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5: G_live.add_node(bus)

        live_dgs = [g for g in purchased_dgs if (pulp.value(s_N.get((g, s), 0)) or 0) > 0.5]
        bus_island = {}
        for comp in nx.connected_components(G_live):
            root = min(comp)
            for b in comp:
                bus_island[b] = root
        island_to_dgs = {}
        for g in live_dgs:
            if g in bus_island:
                island_to_dgs.setdefault(bus_island[g], []).append(g)

        slack_dgs    = set()
        nonslack_dgs = set()
        for dg_list in island_to_dgs.values():
            slack = max(dg_list, key=lambda g: float(dg_sizes.get(g, dg_sizes.get(str(g), 0))))
            slack_dgs.add(slack)
            for g in dg_list:
                if g != slack:
                    nonslack_dgs.add(g)

        for g in slack_dgs:
            info   = bus_phase_info.get(g, {})
            kv_raw = info.get('kv_base', 2.7713)
            kv_LL  = round(kv_raw * math.sqrt(3), 4) if kv_raw < 3.5 else round(kv_raw, 4)
            if not (0.05 <= kv_LL <= 15.0):
                kv_LL = 4.8
            dss.Command(f'New Vsource.DG_{g} bus1={g} basekv={kv_LL:.4f} pu=1.0 phases=3 R1=1e-4 X1=1e-4 R0=1e-4 X0=1e-4 enabled=yes')

        for g in nonslack_dgs:
            info   = bus_phase_info.get(g, {})
            kv_raw = info.get('kv_base', 2.7713)
            kv_LL  = round(kv_raw * math.sqrt(3), 4) if kv_raw < 3.5 else round(kv_raw, 4)
            if not (0.05 <= kv_LL <= 15.0):
                kv_LL = 4.8
            p_milp = sum((pulp.value(P_G[g, p, s]) or 0.0) for p in phases)
            q_milp = sum((pulp.value(Q_G[g, p, s]) or 0.0) for p in phases)
            dss.Command(
                f'New Generator.DG_{g} bus1={g} phases=3 kv={kv_LL:.4f} '
                f'kw={p_milp:.4f} kvar={q_milp:.4f} '
                f'model=1 vminpu=0.0 vmaxpu=2.0 enabled=yes'
            )

        for line in all_lines:
            name = line['name']
            if name in current_faults: continue
            active = (pulp.value(x_BR[name, s]) or 0) > 0.5 if name in switch_names else (pulp.value(x_Line[name, s]) or 0) > 0.5
            if active:
                dss.Command(f'Close Line.{name} terminal=1')
                dss.Command(f'Close Line.{name} terminal=2')

        for tx in transformers:
            if ((pulp.value(s_N.get((tx['bus1'], s), 0)) or 0) > 0.5 and
                    (pulp.value(s_N.get((tx['bus2'], s), 0)) or 0) > 0.5):
                dss.Command(f'Edit Transformer.{tx["name"]} enabled=yes')

        # Iterate the Python `loads` list, NOT dss.Loads.First()/Next(): all loads
        # have been disabled by this point, and OpenDSS's Loads iterator only
        # traverses ENABLED elements, so the cursor freezes on the last load and
        # no loads ever get re-enabled (silent zero-load validation).
        for lname in loads:
            bus = load_weights[lname]['bus']
            if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5:
                dss.Command(f'Edit Load.{lname} enabled=yes')

        for cap in capacitors:
            if (pulp.value(x_Cap[cap['name'], s]) or 0) > 0.5:
                dss.Command(f'Edit Capacitor.{cap["name"]} enabled=yes')

        dss.Command('set controlmode=off')
        dss.Solution.Solve()

        # ── OpenDSS validation ────────────────────────────────────────────────
        ods = {}   # dict populated below; always returned so call site is uniform
        if not dss.Solution.Converged():
            ods = None
        else:
            # 1. Losses
            total_losses_kw = dss.Circuit.Losses()[0]    / 1000.0
            line_losses_kw  = dss.Circuit.LineLosses()[0] / 1000.0
            tx_losses_kw    = total_losses_kw - line_losses_kw

            # 2. Per-bus phase voltages
            live_set  = {i for i in nodes if (pulp.value(s_N.get((i, s), 0)) or 0) > 0.5}
            live_mags = []
            phase_vpu = {}
            for bus_name in dss.Circuit.AllBusNames():
                bkey = bus_name.lower().strip()
                if bkey not in live_set:
                    continue
                dss.Circuit.SetActiveBus(bus_name)
                pu_v      = dss.Bus.puVmagAngle()
                bus_nodes = dss.Bus.Nodes()
                ph_map    = {}
                for idx, nd in enumerate(bus_nodes):
                    if nd in (1, 2, 3):
                        mag = pu_v[idx * 2]
                        if 0.05 < mag < 2.0:
                            ph_map[nd] = mag
                if ph_map:
                    phase_vpu[bkey] = ph_map
                    live_mags.extend(ph_map.values())
            opendss_min_v = min(live_mags) if live_mags else 0.0
            opendss_max_v = max(live_mags) if live_mags else 0.0
            opendss_avg_v = sum(live_mags) / len(live_mags) if live_mags else 0.0

            # 3. NEMA voltage unbalance
            vub_list = []
            for bkey, ph_map in phase_vpu.items():
                if len(ph_map) < 3:
                    continue
                v_vals = list(ph_map.values())
                v_avg  = sum(v_vals) / 3.0
                if v_avg > 1e-3:
                    vub_list.append((bkey, max(abs(v - v_avg) for v in v_vals) / v_avg * 100.0))
            vub_list.sort(key=lambda x: x[1], reverse=True)
            max_vub_pct = vub_list[0][1] if vub_list else 0.0
            max_vub_bus = vub_list[0][0] if vub_list else 'N/A'

            # 4. DG power readback (Vsource = GFM, Generator = GFL)
            p_vsource_kw = -dss.Circuit.TotalPower()[0]
            p_gen_kw     = 0.0
            dg_dispatch  = {}
            for g in live_dgs:
                p_milp = sum((pulp.value(P_G[g, p, s]) or 0.0) for p in phases)
                q_milp = sum((pulp.value(Q_G[g, p, s]) or 0.0) for p in phases)
                s_cap  = float(dg_sizes.get(g, dg_sizes.get(str(g), 0)))
                if g in slack_dgs:
                    dss.Circuit.SetActiveElement(f'Vsource.DG_{g}')
                    pows  = dss.CktElement.Powers()
                    nc    = dss.CktElement.NumConductors()
                    p_act = -sum(pows[2*k]   for k in range(min(3, nc)))
                    q_act = -sum(pows[2*k+1] for k in range(min(3, nc)))
                    tag   = 'GFM'
                else:
                    dss.Circuit.SetActiveElement(f'Generator.DG_{g}')
                    pows  = dss.CktElement.Powers()
                    nc    = dss.CktElement.NumConductors()
                    p_act = -sum(pows[2*k]   for k in range(min(3, nc)))
                    q_act = -sum(pows[2*k+1] for k in range(min(3, nc)))
                    p_gen_kw += p_act
                    tag   = 'GFL'
                s_act = math.sqrt(p_act**2 + q_act**2)
                island_root    = bus_island.get(g)
                island_dgs_lst = island_to_dgs.get(island_root, [g])
                island_q_lim   = sum(0.6 * float(dg_sizes.get(d, dg_sizes.get(str(d), 0)))
                                     for d in island_dgs_lst)
                dg_dispatch[g] = {
                    # New fields
                    'type':               tag,
                    'p_milp_kw':          round(p_milp, 2),
                    'q_milp_kvar':        round(q_milp, 2),
                    'p_ods_kw':           round(p_act,  2),
                    'q_ods_kvar':         round(q_act,  2),
                    's_ods_kva':          round(s_act,  2),
                    's_cap_kva':          round(s_cap,  2),
                    'island_q_limit_kvar':round(island_q_lim, 2),
                    'island_dgs':         sorted(island_dgs_lst),
                    # Aliases for resultspreadsheet.py backward compatibility
                    'p_delivered_kw':     round(p_act,  2),
                    'q_delivered_kvar':   round(q_act,  2),
                    's_delivered_kva':    round(s_act,  2),
                    's_loading_pct':      round(s_act / s_cap * 100.0, 2) if s_cap > 0 else None,
                    'q_limit_kvar':       round(0.6 * s_cap, 2),
                    'within_limit':       abs(q_act) <= 0.6 * s_cap * (1.0 + Q_LIMIT_TOL_FRAC),
                    'within_island_limit':abs(q_act) <= island_q_lim * (1.0 + Q_LIMIT_TOL_FRAC),
                }
            p_dg_total = p_vsource_kw + p_gen_kw

            # 5. Served load (iterated Load elements) — P, Q, S per load and totals
            p_load_kw   = 0.0
            q_load_kvar = 0.0
            served_loads = {}
            if dss.Loads.First():
                while True:
                    lname_el = dss.Loads.Name()
                    dss.Circuit.SetActiveElement(f'Load.{lname_el}')
                    pows = dss.CktElement.Powers()
                    nc   = dss.CktElement.NumConductors()
                    n_ph = min(3, nc)
                    lp   = sum(pows[2*k]   for k in range(n_ph))
                    lq   = sum(pows[2*k+1] for k in range(n_ph))
                    ls   = math.sqrt(lp**2 + lq**2)
                    p_load_kw   += lp
                    q_load_kvar += lq
                    if lp > 0.01 or lq > 0.01:
                        served_loads[lname_el] = {
                            'kw':   round(lp, 3),
                            'kvar': round(lq, 3),
                            'kva':  round(ls, 3),
                        }
                    if not dss.Loads.Next():
                        break
            s_load_kva = math.sqrt(p_load_kw**2 + q_load_kvar**2)

            # MILP-side losses: LinDistFlow is lossless by formulation → should be ~0
            milp_restored_kw = total_grid_kw - milp_shed_kw
            milp_p_dg_kw     = sum((pulp.value(P_G[g, p, s]) or 0.0)
                                   for g in live_dgs for p in phases)
            milp_losses_kw   = milp_p_dg_kw - milp_restored_kw

            # 6. Branch loading
            branch_loading = {}
            dss.Lines.First()
            while True:
                lname = dss.Lines.Name()
                dss.Circuit.SetActiveElement(f'Line.{lname}')
                pows  = dss.CktElement.Powers()
                nc    = dss.CktElement.NumConductors()
                n_ph  = min(3, nc)
                s_ph  = [math.sqrt(pows[2*k]**2 + pows[2*k+1]**2) for k in range(n_ph)]
                s_tot = sum(s_ph)
                s_max = max(s_ph) if s_ph else 0.0
                if s_tot > 0.5:
                    cur      = dss.CktElement.CurrentsMagAng()
                    i_max    = max(cur[2*k] for k in range(n_ph)) if n_ph else 0.0
                    load_pct = s_max / LINE_CAPACITY * 100.0
                    branch_loading[lname] = {
                        's_3ph_kva':    round(s_tot,    2),
                        's_max_ph_kva': round(s_max,    2),
                        'i_max_a':      round(i_max,    2),
                        'load_pct':     round(load_pct, 2),
                    }
                if not dss.Lines.Next():
                    break
            max_branch_pct  = max((v['load_pct'] for v in branch_loading.values()), default=0.0)
            max_branch_name = max(branch_loading, key=lambda k: branch_loading[k]['load_pct'],
                                  default=None)

            # 7. Transformer loading
            tx_alive = (
                (pulp.value(s_N.get(('709', s), 0)) or 0) > 0.5 and
                (pulp.value(s_N.get(('775', s), 0)) or 0) > 0.5
            )
            tx_s_kva       = 0.0
            tx_load_pct    = 0.0
            tx_phase_detail = {}   # per-phase MILP vs ODS comparison
            if tx_alive:
                dss.Circuit.SetActiveElement('Transformer.XFM1')
                pows        = dss.CktElement.Powers()
                p_tx        = sum(pows[2*k]   for k in range(3))
                q_tx        = sum(pows[2*k+1] for k in range(3))
                tx_s_kva    = math.sqrt(p_tx**2 + q_tx**2)
                tx_load_pct = tx_s_kva / 500.0 * 100.0
                # per-phase comparison: MILP variable vs OpenDSS terminal-1 (bus 709 side)
                ph_idx = {'a': 0, 'b': 1, 'c': 2}
                for ph in phases:
                    k = ph_idx[ph]
                    milp_p = round(pulp.value(P_Tx.get(('xfm1', ph, s), 0)) or 0.0, 3)
                    milp_q = round(pulp.value(Q_Tx.get(('xfm1', ph, s), 0)) or 0.0, 3)
                    ods_p  = round(pows[2*k],   3)
                    ods_q  = round(pows[2*k+1], 3)
                    tx_phase_detail[ph] = {
                        'milp_p_kw':   milp_p,
                        'milp_q_kvar': milp_q,
                        'ods_p_kw':    ods_p,
                        'ods_q_kvar':  ods_q,
                        'diff_p_kw':   round(milp_p - ods_p, 3),
                    }

            ods = {
                'total_loss_kw':          round(total_losses_kw, 4),
                'line_loss_kw':           round(line_losses_kw,  4),
                'trafo_loss_kw':          round(tx_losses_kw,    4) if tx_alive else 0.0,
                'p_dg_kw':                round(p_dg_total, 2),
                'p_load_kw':              round(p_load_kw,  2),
                'q_load_kvar':            round(q_load_kvar, 2),
                's_load_kva':             round(s_load_kva,  2),
                'milp_p_dg_kw':           round(milp_p_dg_kw,   2),
                'milp_losses_kw':         round(milp_losses_kw,  4),
                'min_v':                  round(opendss_min_v, 4),
                'avg_v':                  round(opendss_avg_v, 4),
                'max_v':                  round(opendss_max_v, 4),
                'bus_voltages':           {b: {str(ph): round(v, 4) for ph, v in pm.items()}
                                           for b, pm in phase_vpu.items()},
                'nema_vub_max_pct':       round(max_vub_pct, 4),
                'nema_vub_bus':           max_vub_bus,
                'vub_per_bus':            [{'bus': b, 'vub_pct': round(vub, 4)}
                                           for b, vub in vub_list],
                'max_branch_loading_pct': round(max_branch_pct, 2),
                'max_loaded_branch':      max_branch_name,
                'branch_loading':         branch_loading,
                'trafo_alive':            tx_alive,
                'trafo_s_kva':            round(tx_s_kva,    2),
                'trafo_load_pct':         round(tx_load_pct, 2),
                'trafo_phase_detail':     tx_phase_detail,
                'served_loads':           served_loads,
            }

        return {
            'opt_cost':      opt_cost,
            'milp_shed_kw':  milp_shed_kw,
            'restored_loads':restored_loads,
            'dg_dispatch':   dg_dispatch if ods is not None else {},
            'ods':           ods,
        }
    else:
        return None
    
# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    if LABEL:
        json_filename_test = f"500_scenarios_{LABEL}.json"
    else:
        json_filename_test = "500_scenarios_pga030N30L000Y15R010_rho00_sig50.json"   # original hardcoded name (standalone)
    
    if not os.path.exists(json_filename_test):
        print(f"❌ ERROR: Missing {json_filename_test}. Ensure it has 'faults' and 'duration_hours'.")
        exit()
        
    with open(json_filename_test, "r") as infile:
        fault_scenarios = json.load(infile)
        
    print(f"📥 Loaded Master Plan: {len(purchased_dgs)} DGs installed. C_INV = ${C_INV:,.2f}")
    print(f"🚀 Initiating Rigorous SAA Upper Bound Test ({len(fault_scenarios)} Scenarios)...")
    
    refresh_base_grid_state()

    # --- Static load priority list (bus → kW → VoLL tier), sorted highest-priority first ---
    tier_label = lambda v: "CRITICAL" if v == VOLL_CRITICAL else ("COMMERCIAL/INDUSTRIAL" if v == VOLL_TIER_2 else "RESIDENTIAL")
    f_priority_static = {}
    for l in loads:
        bus = load_weights[l]['bus']
        total_kw = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
        if bus in CRITICAL_BUSES:
            f_priority_static[l] = VOLL_CRITICAL
        elif total_kw >= 50.0:
            f_priority_static[l] = VOLL_TIER_2
        else:
            f_priority_static[l] = VOLL_TIER_1

    static_load_priority_list = sorted([
        {
            "load": l,
            "bus": load_weights[l]['bus'],
            "phases": load_weights[l]['phases'],
            "total_kW": round(load_weights[l]['kW_phase'] * len(load_weights[l]['phases']), 4),
            "voll_tier": f_priority_static[l],
            "tier_label": tier_label(f_priority_static[l])
        }
        for l in loads
    ], key=lambda x: (-x['voll_tier'], -x['total_kW']))

    total_storm_penalty_cost = 0.0
    valid_scenarios_count = 0
    results_log = {}
    restoration_count = {l: 0 for l in loads}   # tallied across all valid scenarios

    for i, (s_name, details) in enumerate(fault_scenarios.items()):
        if s_name in ('Normal_Operation', 'metadata'):
            continue
            
        current_faults = details["faults"]
        scenario_duration = details["duration_hours"]
            
        print(f"[{i}/{len(fault_scenarios)-1}] Simulating {s_name} ({len(current_faults)} faults, {scenario_duration:.1f} hrs)...", end=" ")
        
        result = evaluate_single_scenario(
            s_name, current_faults, scenario_duration, purchased_dgs, dg_sizes, MIP_GAP)

        if result is not None:
            cost  = result['opt_cost']
            shed  = result['milp_shed_kw']
            ods   = result['ods']   # None if OpenDSS did not converge
            total_storm_penalty_cost += cost
            valid_scenarios_count    += 1

            _loss    = ods['total_loss_kw']           if ods else None
            _min_v   = ods['min_v']                   if ods else None
            _max_bpct= ods['max_branch_loading_pct']  if ods else None
            print(f"Cost: ${cost:9,.2f} | Shed: {shed:6.1f} kW | "
                  f"ODS Loss: {_loss if _loss is not None else 0:5.2f} kW | "
                  f"Min V: {_min_v if _min_v else 0:.3f} | "
                  f"Max branch: {_max_bpct if _max_bpct else 0:.1f}%")

            for l, was_restored in result['restored_loads'].items():
                restoration_count[l] += was_restored

            results_log[s_name] = {
                "penalty_cost":  cost,
                "shed_kw":       shed,
                "ens_kwh":       shed * scenario_duration,
                "dg_dispatch":   result['dg_dispatch'],
                # OpenDSS fields — all None when ODS did not converge
                "dss_total_loss_kw":          ods['total_loss_kw']           if ods else None,
                "dss_line_loss_kw":           ods['line_loss_kw']            if ods else None,
                "dss_trafo_loss_kw":          ods['trafo_loss_kw']           if ods else None,
                "dss_p_dg_kw":                ods['p_dg_kw']                 if ods else None,
                "dss_p_load_kw":              ods['p_load_kw']               if ods else None,
                "dss_q_load_kvar":            ods['q_load_kvar']             if ods else None,
                "dss_s_load_kva":             ods['s_load_kva']              if ods else None,
                "served_loads":               ods['served_loads']            if ods else {},
                "milp_p_dg_kw":               ods['milp_p_dg_kw']            if ods else None,
                "milp_losses_kw":             ods['milp_losses_kw']          if ods else None,
                "dss_min_v":                  ods['min_v']                   if ods else None,
                "dss_avg_v":                  ods['avg_v']                   if ods else None,
                "dss_max_v":                  ods['max_v']                   if ods else None,
                "bus_voltages":               ods['bus_voltages']            if ods else {},
                "nema_vub_max_pct":           ods['nema_vub_max_pct']        if ods else None,
                "nema_vub_bus":               ods['nema_vub_bus']            if ods else None,
                "vub_per_bus":                ods['vub_per_bus']             if ods else [],
                "max_branch_loading_pct":     ods['max_branch_loading_pct']  if ods else None,
                "max_loaded_branch":          ods['max_loaded_branch']        if ods else None,
                "branch_loading":             ods['branch_loading']           if ods else {},
                "trafo_alive":                ods['trafo_alive']              if ods else False,
                "trafo_s_kva":                ods['trafo_s_kva']              if ods else None,
                "trafo_load_pct":             ods['trafo_load_pct']           if ods else None,
                "trafo_phase_detail":         ods['trafo_phase_detail']       if ods else {},
                # Aliases for resultspreadsheet.py backward compatibility
                "dss_loss_kw":                ods['total_loss_kw']            if ods else None,
                "dss_served_kw":              ods['p_load_kw']                if ods else None,
            }
        else:
            print("❌ MILP Failed to Converge.")

    # --- SAA UPPER BOUND CALCULATION ---
    if valid_scenarios_count > 0:
        average_storm_cost = total_storm_penalty_cost / valid_scenarios_count
        expected_lifetime_storm_cost = average_storm_cost * EXPECTED_EVENTS_IN_LIFESPAN

        # Present-value capital + O&M — mirrors main_initial.py's objective and the
        # paper's Eq (C_Inv + C_OM^PV): CAPEX is a one-time outlay at t=0 (= C_INV),
        # O&M is the PWF-discounted annual stream. (Previously CRF*Y*C_INV and OM*Y*kw,
        # an undiscounted-lifetime convention that over-counted capital ~2x and did not
        # match the objective the MILP actually minimized.)
        PWF                 = 1.0 / CRF   # present-worth factor (reciprocal of CRF)
        capex_present_value = C_INV
        # Reuse main_initial.py's own PV(O&M), which already applies the 0.8 kVA->kW
        # power-factor conversion (Eq. C_OM^PV = C_OM_yr * PWF * 0.8 * S_Cap); dg_sizes
        # here holds raw nameplate kVA, so recomputing from it without the 0.8 factor
        # would overstate O&M by 1/0.8.
        om_present_value    = master_data["financials"]["om_present_value"]

        # Risk-averse storm cost = (1-λ)·mean + λ·CVaRα over the per-scenario costs,
        # so the saved upper_bound_total_cost matches the paper's risk-averse objective
        # for λ>0 (not just the risk-neutral mean). expected_lifetime_storm_cost above
        # stays mean-based on purpose — resultspreadsheet.py uses it only as the
        # lifetime multiplier. CVaR via Rockafellar–Uryasev with a linear-interpolation
        # VaR quantile (matches numpy's default, so it agrees with resultspreadsheet).
        lam        = float(C.get("LAMBDA_RISK", 0.0) or 0.0)
        alpha_cvar = float(C.get("ALPHA", 0.90) or 0.90)
        _costs = sorted(v["penalty_cost"] for v in results_log.values()
                        if v.get("penalty_cost") is not None)
        if _costs:
            _n    = len(_costs)
            _mean = sum(_costs) / _n
            _h    = (_n - 1) * alpha_cvar
            _lo   = int(math.floor(_h))
            _hi   = min(_lo + 1, _n - 1)
            _var  = _costs[_lo] + (_h - _lo) * (_costs[_hi] - _costs[_lo])   # VaRα (linear interp)
            _cvar = _var + (1.0 / (1.0 - alpha_cvar)) * (sum(max(c - _var, 0.0) for c in _costs) / _n)
            risk_averse_storm = ((1.0 - lam) * _mean + lam * _cvar) * EXPECTED_EVENTS_IN_LIFESPAN
        else:
            risk_averse_storm = expected_lifetime_storm_cost
        upper_bound_total_cost = capex_present_value + om_present_value + risk_averse_storm

        print("\n==========================================")
        print("🏆 RIGOROUS TESTING COMPLETE 🏆")
        print("==========================================")
        print(f"   Planning Horizon (Y):                {PLANNING_YEARS} years")
        print(f"   Interest Rate (r):                   {INTEREST_RATE*100:.1f}%")
        print(f"   Capital Recovery Factor (CRF):       {CRF:.4f}")
        print(f"   Total Investment Cost (C_INV):       ${C_INV:,.2f}")
        print(f"   Present-Value Capital (C_INV @ t=0): ${capex_present_value:,.2f}")
        print(f"   Present-Value O&M (PWF-discounted):  ${om_present_value:,.2f}")
        print(f"   Average Cost per Storm Event:        ${average_storm_cost:,.2f}")
        print(f"   Expected Lifetime Storm Cost:        ${expected_lifetime_storm_cost:,.2f}")
        print(f"   ---------------------------------------")
        print(f"   SAA UPPER BOUND (risk-averse, λ={lam}): ${upper_bound_total_cost:,.2f}")
        print("==========================================\n")

        # Load restoration frequency — sorted by restoration rate desc, then kW desc
        load_restoration_summary = sorted([
            {
                "load": l,
                "bus": load_weights[l]['bus'],
                "phases": load_weights[l]['phases'],
                "total_kW": round(load_weights[l]['kW_phase'] * len(load_weights[l]['phases']), 4),
                "voll_tier": f_priority_static[l],
                "tier_label": tier_label(f_priority_static[l]),
                "restored_count": restoration_count[l],
                "restoration_rate_pct": round(restoration_count[l] / valid_scenarios_count * 100, 2)
            }
            for l in loads
        ], key=lambda x: (-x['restoration_rate_pct'], -x['total_kW']))

        output_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scenarios_tested": valid_scenarios_count
            },
            "saa_calculations": {
                "C_INV": C_INV,
                "capex_present_value": capex_present_value,
                "om_present_value": om_present_value,
                "average_storm_cost": average_storm_cost,
                "expected_lifetime_storm_cost": expected_lifetime_storm_cost,
                "upper_bound_total_cost": upper_bound_total_cost
            },
            "load_priority_static": static_load_priority_list,
            "load_restoration_frequency": load_restoration_summary,
            "scenario_details": results_log
        }
        
        # Dynamic export filename — uses LABEL when provided by orchestrator
        if LABEL:
            export_filename = f"evaluation_results_500_{LABEL}_{iteration}.json"
        else:
            # Original hardcoded filename (kept for standalone backward-compatibility)
            # export_filename = f"evaluation_results_500_initial.json"
            export_filename = f"evaluation_results_500_test_{iteration}.json"
        with open(export_filename, "w") as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"✅ Data successfully saved to {export_filename}")
