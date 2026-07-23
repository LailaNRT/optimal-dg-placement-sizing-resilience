import opendssdirect as dss
import os
import pulp
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import json
from collections import Counter

# =============================================================================
# CONFIGURATION — change these three lines to test any scenario
# =============================================================================
# # JSON_SCENARIO_FILE  = "500_scenarios.json"
JSON_SCENARIO_FILE = "500_scenarios_pga030N30L000Y15R010_rho00_sig50.json"
SCENARIO_KEY        = "Earthquake_Scenario_480" # Change this to whichever storm you want to view
DG_PLACEMENTS_FILE  = "master_dg_placements_pga030N30L000Y15R010_rho00_sig50_1.json" # UPDATED to match your main output
DSS_FILE            = "ieee37.dss"
#coba 49, 62, 68, 360, 383, 402, 499
# JSON_SCENARIO_FILE  = "500_scenarios.json"
# JSON_SCENARIO_FILE = "saa_scenario_pga030N30_3.json"
# SCENARIO_KEY        = "Earthquake_Scenario_29" # Change this to whichever storm you want to view
# DG_PLACEMENTS_FILE  = "master_dg_placements_test_3.json" # UPDATED to match your main output
# DSS_FILE            = "ieee37.dss"
#coba 49, 62, 68, 360, 383, 402, 499
# =============================================================================
# CONSTANTS  (mirror main_initial.py / test_initial.py)
# =============================================================================
COST_GEN_PER_KWH  = 0.45
VOLL_TIER_1       = 3.3        # Residential (< 50 kW)
VOLL_TIER_2       = 21.8       # Commercial / Industrial (>= 50 kW)
VOLL_CRITICAL     = 100.0      # Critical infrastructure (bus-specific override)
CRITICAL_BUSES    = {'712', '729', '738', '720'}
LINE_CAPACITY     = 500.0
TX_CAPACITY       = 500.0 / 3.0   # XFM1 rated 500 kVA total 3-phase → per-phase limit
M_Power           = 1000.0
M_Volt            = 1.5
sqrt3             = math.sqrt(3)

# Feeder base voltage for the LinDistFlow pu² normalization. IEEE 37-bus is
# 4.8 kV L-L; LinDistFlow voltage drop uses L-N base (4.8/√3).
V_BASE_LN_KV      = 4.8 / math.sqrt(3)
VDROP_DENOM       = (V_BASE_LN_KV ** 2) * 1000.0   # = 7680.0
U_max, U_min      = 1.05 ** 2, 0.95 ** 2

# =============================================================================
# STEP 1 — Load the fault scenario and locked DG/BESS placements
# =============================================================================
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

print(f"\n📥 Loading scenario '{SCENARIO_KEY}' from {JSON_SCENARIO_FILE}...")
with open(JSON_SCENARIO_FILE, "r") as f:
    all_scenarios = json.load(f)

if SCENARIO_KEY not in all_scenarios:
    raise KeyError(f"'{SCENARIO_KEY}' not found in {JSON_SCENARIO_FILE}.\n"
                   f"Available: {list(all_scenarios.keys())}")

scenario_data     = all_scenarios[SCENARIO_KEY]
faulted_lines     = {ln.lower() for ln in scenario_data["faults"]}
scenario_duration = float(scenario_data["duration_hours"])
print(f"   Faults ({len(faulted_lines)}): {sorted(faulted_lines)}")
print(f"   Duration: {scenario_duration} h")

print(f"\n📥 Loading DG placements from {DG_PLACEMENTS_FILE}...")
with open(DG_PLACEMENTS_FILE, "r") as f:
    master_data = json.load(f)

placement      = master_data["dg_placement"]
purchased_dgs  = placement["purchased_buses"]
dg_sizes       = placement.get("sizes_kva", placement.get("sizes_kw", {}))

print(f"   Diesel DGs locked at: {purchased_dgs}")
for dg in purchased_dgs:
    sz = float(dg_sizes.get(dg, dg_sizes.get(str(dg), 0)))
    print(f"     Bus {dg:>6s}: Diesel {sz:.0f} kVA")

s = SCENARIO_KEY   # scenario label used as index key throughout

# =============================================================================
# STEP 2 — Compile OpenDSS and extract full topology
# =============================================================================
print("\n🔌 Initialising OpenDSS topology engine...")
dss.Command('Clear')
dss.Command(f'Compile "{DSS_FILE}"')

dss.Vsources.First()
SUB_BUS = dss.CktElement.BusNames()[0].split('.')[0].lower().strip()
print(f"   Substation bus: '{SUB_BUS}'")

bus_phase_info = {}
for bus_name in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(bus_name)
    bus_phase_info[bus_name.lower().strip()] = {
        'num_phases': dss.Bus.NumNodes(),
        'nodes':      list(dss.Bus.Nodes()),
        'kv_base':    dss.Bus.kVBase()
    }

nodes = [b.lower() for b in dss.Circuit.AllBusNames()]

# ── Bus coordinates ──────────────────────────────────────────────────────────
pos = {}
for coord_file in ["IEEE37_BusXY.csv", "BusCoords.csv", "BusCoords.dat"]:
    if not os.path.exists(coord_file):
        continue
    loaded_now = 0
    with open(coord_file, "r") as cf:
        for raw_line in cf:
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith('!') or raw_line.startswith('#'):
                continue
            parts = raw_line.replace(',', ' ').split()
            if len(parts) >= 3:
                try:
                    key = parts[0].strip().lower()
                    if key not in pos:
                        pos[key] = (float(parts[1]), float(parts[2]))
                        loaded_now += 1
                except ValueError:
                    continue
    if loaded_now:
        print(f"   +{loaded_now} coords from {coord_file}")
        break

covered = sum(1 for n in nodes if n in pos)
print(f"   Coordinate coverage: {covered}/{len(nodes)} buses ({100*covered//len(nodes) if nodes else 0}%)")

# ── Loads ─────────────────────────────────────────────────────────────────────
loads, load_weights, seen_loads = [], {}, set()
dss.Loads.First()
for _ in range(dss.Loads.Count()):
    lname = dss.Loads.Name().lower().strip()
    if lname not in seen_loads:
        bname_full     = dss.CktElement.BusNames()[0].lower()
        bname          = bname_full.split('.')[0].strip()
        load_nodes     = (bname_full.split('.')[1:] if '.' in bname_full
                          else [str(i+1) for i in range(dss.Loads.Phases())])
        phase_map      = {'1': 'a', '2': 'b', '3': 'c'}
        present_phases = [phase_map[n] for n in load_nodes if n in phase_map]
        n_ph           = len(present_phases) if present_phases else 1
        loads.append(lname)
        seen_loads.add(lname)
        load_weights[lname] = {
            'bus':        bname,
            'phases':     present_phases,
            'kW_phase':   dss.Loads.kW()   / n_ph,
            'kvar_phase': dss.Loads.kvar() / n_ph,
        }
    dss.Loads.Next()

loads_at_node = {i: [] for i in nodes}
for l in loads:
    bus = load_weights[l]['bus']
    if bus in loads_at_node:
        loads_at_node[bus].append(l)

total_grid_kw = sum(
    load_weights[l]['kW_phase'] * len(load_weights[l]['phases']) for l in loads
)

# ── Lines ─────────────────────────────────────────────────────────────────────
switchable_lines, switch_names, all_lines = [], [], []
seen_switches = set()
dss.Lines.First()
for _ in range(dss.Lines.Count()):
    name      = dss.Lines.Name().lower()
    bus1_full = dss.Lines.Bus1().lower()
    b1        = bus1_full.split('.')[0].strip()
    b2        = dss.Lines.Bus2().split('.')[0].lower().strip()
    nodes1    = (bus1_full.split('.')[1:] if '.' in bus1_full
                 else [str(i+1) for i in range(dss.Lines.Phases())])
    phase_map      = {'1': 'a', '2': 'b', '3': 'c'}
    present_phases = [phase_map[n] for n in nodes1 if n in phase_map]

    rmat_flat = dss.Lines.RMatrix()
    xmat_flat = dss.Lines.XMatrix()
    r_3x3 = {p1: {p2: 0.0 for p2 in ['a','b','c']} for p1 in ['a','b','c']}
    x_3x3 = {p1: {p2: 0.0 for p2 in ['a','b','c']} for p1 in ['a','b','c']}
    num_ph = dss.Lines.Phases()
    if len(rmat_flat) == num_ph ** 2:
        idx = 0
        for i in range(num_ph):
            for j in range(num_ph):
                if i < len(present_phases) and j < len(present_phases):
                    r_3x3[present_phases[i]][present_phases[j]] = rmat_flat[idx]
                    x_3x3[present_phases[i]][present_phases[j]] = xmat_flat[idx]
                idx += 1

    all_lines.append({
        'name': name, 'bus1': b1, 'bus2': b2,
        'r_matrix': r_3x3, 'x_matrix': x_3x3, 'phases': present_phases
    })
    is_sw = (dss.Lines.IsSwitch() if hasattr(dss.Lines, 'IsSwitch') else False) \
            or name.startswith('sw')
    if is_sw and name not in seen_switches:
        switchable_lines.append({'name': name, 'bus1': b1, 'bus2': b2})
        switch_names.append(name)
        seen_switches.add(name)
    dss.Lines.Next()

# ── Transformers ──────────────────────────────────────────────────────────────
transformers = []
dss.Transformers.First()
for _ in range(dss.Transformers.Count()):
    t_name = dss.Transformers.Name().lower()
    buses  = dss.CktElement.BusNames()
    tb1    = buses[0].split('.')[0].lower().strip()
    tb2    = (buses[1].split('.')[0].lower().strip() if len(buses) > 1 else tb1)
    phase_map  = {'1': 'a', '2': 'b', '3': 'c'}
    nodes2     = (buses[1].split('.')[1:] if len(buses) > 1 and '.' in buses[1]
                  else [str(i+1) for i in range(dss.CktElement.NumPhases())])
    sec_phases = [phase_map[n] for n in nodes2 if n in phase_map]
    dss.Transformers.Wdg(2)
    tap_ratio = dss.Transformers.Tap()
    a_ratio   = {p: (tap_ratio if p in sec_phases else 1.0) for p in ['a','b','c']}
    transformers.append({
        'name': t_name, 'bus1': tb1, 'bus2': tb2,
        'a_ratio': a_ratio, 'phases': sec_phases
    })
    dss.Transformers.Next()

# ── Capacitors ────────────────────────────────────────────────────────────────
capacitors = []
dss.Capacitors.First()
for _ in range(dss.Capacitors.Count()):
    c_name     = dss.Capacitors.Name().lower().strip()
    bname_full = dss.CktElement.BusNames()[0].lower()
    bname      = bname_full.split('.')[0].strip()
    cap_nodes  = bname_full.split('.')[1:] if '.' in bname_full else ['1','2','3']
    phase_map  = {'1': 'a', '2': 'b', '3': 'c'}
    cap_phases = [phase_map[n] for n in cap_nodes if n in phase_map]
    kvar_ph    = dss.Capacitors.kvar() / len(cap_phases) if cap_phases else 0
    capacitors.append({
        'name': c_name, 'bus': bname, 'phases': cap_phases, 'kvar_phase': kvar_ph
    })
    dss.Capacitors.Next()

# ── Regulator secondary buses (excluded from candidate list) ──────────────────
regulator_buses = set()
dss.RegControls.First()
for _ in range(dss.RegControls.Count()):
    tx_name   = dss.RegControls.Transformer()
    dss.Transformers.Name(tx_name)
    reg_buses = dss.CktElement.BusNames()
    if len(reg_buses) > 1:
        regulator_buses.add(reg_buses[1].split('.')[0].lower().strip())
    dss.RegControls.Next()

# ── Candidate buses ───────────────────────────────────────────────────────────
phases  = ['a', 'b', 'c']
N_nodes = len(nodes)

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

print(f"   {len(candidate_buses)} candidate buses | "
      f"{len(regulator_buses)} regulator bus(es) excluded")

# Bus adjacency lookups -- built once, reused by the fictitious-flow and
# power-balance constraint loops instead of scanning all_lines per node.
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

# =============================================================================
# STEP 3 — Build the MILP  (mirrors main_v1.py constraint-for-constraint)
# =============================================================================
print(f"\n🚀 Building single-event MILP for '{SCENARIO_KEY}'...")
milp = pulp.LpProblem("SingleEvent_CSH", pulp.LpMinimize)

# ── First-stage: DG placement is FIXED by the master plan, so x_G / S_Cap are
# plain constants here, not variables pinned by equality constraints. This
# also makes the S_Cap * s_N product exactly linear (no big-M linearization).
x_G   = {c: (1 if c in purchased_dgs else 0) for c in candidate_buses}
S_Cap = {c: (float(dg_sizes.get(c, dg_sizes.get(str(c), 0))) if c in purchased_dgs else 0.0)
         for c in candidate_buses}

# ── Second-stage operational variables ───────────────────────────────────────
s_N   = pulp.LpVariable.dicts("s_N",   ((i,  s) for i   in nodes),              cat='Binary')
x_BR  = pulp.LpVariable.dicts("x_BR",  ((sw, s) for sw  in switch_names),       cat='Binary')
x_Cap = pulp.LpVariable.dicts("x_Cap", ((cap['name'], s) for cap in capacitors), cat='Binary')

P_G = pulp.LpVariable.dicts(
    "P_G",
    ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases),
    lowBound=0, cat='Continuous'
)
Q_G = pulp.LpVariable.dicts(
    "Q_G",
    ((c, p, s) for c in ['MAIN_SUBSTATION'] + candidate_buses for p in phases),
    lowBound=-300, upBound=300, cat='Continuous'
)
U_N    = pulp.LpVariable.dicts("U_N",    ((i, p, s) for i in nodes for p in phases),
                                cat='Continuous')
P_Line = pulp.LpVariable.dicts("P_Line", ((ln['name'], p, s) for ln in all_lines for p in phases),
                                lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')
Q_Line = pulp.LpVariable.dicts("Q_Line", ((ln['name'], p, s) for ln in all_lines for p in phases),
                                lowBound=-LINE_CAPACITY, upBound=LINE_CAPACITY, cat='Continuous')

# ── Radiality: spanning forest via fictitious flow ────────────────────────────
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

# ── Load priority weights ────────────────────────────────────────────────────
# Bus-specific critical override takes precedence over kW-threshold tiers,
# matching the formulation in main_initial.py and test_initial.py.
f_priority = {}
for l in loads:
    bus = load_weights[l]['bus']
    total_kw = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
    if bus in CRITICAL_BUSES:
        f_priority[l] = VOLL_CRITICAL
    elif total_kw >= 50.0:
        f_priority[l] = VOLL_TIER_2
    else:
        f_priority[l] = VOLL_TIER_1

# # ── Objective: minimise weighted load shedding + DG fuel cost ────────────────
fuel_cost = pulp.lpSum([
    P_G[c, p, s] * COST_GEN_PER_KWH * scenario_duration
    for c in candidate_buses for p in phases
])

# OF2_shedding = pulp.lpSum([
#     f_priority[l] * load_weights[l]['kW_phase']
#     * (1 - s_N[load_weights[l]['bus'], s]) * scenario_duration
#     for l in loads for _ in load_weights[l]['phases']
# ])

# # Direct dollar minimization, identical to main_initial.py
# milp += OF2_shedding + fuel_cost, "Minimize_SingleEvent_Cost"

# 1. Define the shed variable
shed = pulp.LpVariable.dicts("shed", ((l, s) for l in loads), lowBound=0, cat='Continuous')

# 2. Tie it to the bus status
for l in loads:
    nph = len(load_weights[l]['phases'])
    milp += shed[l, s] >= load_weights[l]['kW_phase'] * nph * (1 - s_N[load_weights[l]['bus'], s])

# 3. Create the penalty
shedding_penalty = pulp.lpSum([f_priority[l] * shed[l, s] * scenario_duration for l in loads])

# 4. Minimize
milp += shedding_penalty + fuel_cost, "Minimize_SingleEvent_Cost"

# ── alpha matrices for 3-phase LinDistFlow ────────────────────────────────────
sqrt3_2    = sqrt3 / 2.0
alpha_real = {
    'a': {'a': 1.0, 'b': -0.5,    'c': -0.5},
    'b': {'a': -0.5,   'b': 1.0,  'c': -0.5},
    'c': {'a': -0.5,   'b': -0.5, 'c':  1.0}
}
alpha_imag = {
    'a': {'a': 0.0,    'b': -sqrt3_2, 'c':  sqrt3_2},
    'b': {'a':  sqrt3_2, 'b': 0.0,   'c': -sqrt3_2},
    'c': {'a': -sqrt3_2, 'b':  sqrt3_2, 'c': 0.0}
}

# ── Substation is dead in every storm scenario ────────────────────────────────
milp += s_N[SUB_BUS, s] == 0, "Sub_Dead"
for p in phases:
    milp += U_N[SUB_BUS, p, s] == 0,                   f"Sub_U_Dead_{p}"
    milp += P_G['MAIN_SUBSTATION', p, s] == 0,         f"Sub_P_Dead_{p}"
    milp += Q_G['MAIN_SUBSTATION', p, s] == 0,         f"Sub_Q_Dead_{p}"

# ── DG bus voltage reference (grid-forming synchronous generator anchors island at 1.0 pu) ──
for c in candidate_buses:
    if not x_G[c]:
        continue   # constraint is vacuous for non-installed candidates
    for p in phases:
        milp += U_N[c,p,s] >= 1.0 - M_Volt * (1 - s_N[c,s]), f"DG_V_lo_{c}_{p}"
        milp += U_N[c,p,s] <= 1.0 + M_Volt * (1 - s_N[c,s]), f"DG_V_hi_{c}_{p}"

# --- FIXED DG OUTPUT CONSTRAINTS (Unbalanced synchronous-generator dispatch) ---
for c in candidate_buses:
    for p in phases:
        # 1. Individual Phase Limits: Max 1/3 of total capacity per phase
        milp += P_G[c,p,s] <= 0.8 * S_Cap[c] / 3.0,      f"DG_Pmax_{c}_{p}"
        milp += P_G[c,p,s] <= M_Power * s_N[c,s],        f"DG_Pgate_{c}_{p}"

        # 2. Reactive power limit (standard 0.8 PF → Q max ≈ 60 % of P_rated)
        milp += Q_G[c,p,s] <=  0.6 * (S_Cap[c] / 3.0),   f"DG_Qmax_{c}_{p}"
        milp += Q_G[c,p,s] >= -0.6 * (S_Cap[c] / 3.0),   f"DG_Qmin_{c}_{p}"
        milp += Q_G[c,p,s] <=  M_Power * s_N[c,s],        f"DG_Qgate_hi_{c}_{p}"
        milp += Q_G[c,p,s] >= -M_Power * s_N[c,s],        f"DG_Qgate_lo_{c}_{p}"

    # 3. ASYMMETRIC INJECTION:
    # Total P across phases ≤ P_max = 0.8 × S_cap (S_Cap holds kVA from sizes_kva).
    p_tot = pulp.lpSum([P_G[c,p,s] for p in phases])
    milp += p_tot <= 0.8 * S_Cap[c] * s_N[c,s], f"DG_Cap3ph_Total_{c}"

# ── Radiality: line status ────────────────────────────────────────────────────
for line in all_lines:
    name, b1, b2 = line['name'], line['bus1'], line['bus2']
    if name in faulted_lines:
        milp += x_Line[name,s] == 0,                      f"Fault_xLine_{name}"
    elif name in switch_names:
        milp += x_Line[name,s] == x_BR[name,s],           f"SW_xLine_{name}"
    else:
        milp += x_Line[name,s] <= s_N[b1,s],              f"HW_xLine_b1_{name}"
        milp += x_Line[name,s] <= s_N[b2,s],              f"HW_xLine_b2_{name}"
        milp += x_Line[name,s] >= s_N[b1,s] + s_N[b2,s] - 1, f"HW_xLine_lo_{name}"

# ── Radiality: virtual root can only connect to buses with an installed DG ───
for i in nodes:
    if i != SUB_BUS and i in candidate_buses and x_G[i]:
        milp += z_Vir[i,s] <= s_N[i,s],  f"zVir_DG_sN_{i}"
    else:
        milp += z_Vir[i,s] == 0,   f"zVir_off_{i}"

# Spanning forest: physical edges + transformer edges + virtual roots = live nodes
milp += (
    pulp.lpSum([x_Line[ln['name'],s] for ln in all_lines]) +
    pulp.lpSum([s_N[tx['bus1'],s] for tx in transformers]) +
    pulp.lpSum([z_Vir[i,s] for i in nodes])
    == pulp.lpSum([s_N[i,s] for i in nodes]),
    "SpanningForest"
)

# ── Fictitious flow (proves connectivity, no cycles) ─────────────────────────
for i in nodes:
    milp += f_Vir[i,s] <= N_nodes * z_Vir[i,s],    f"fVir_gate_{i}"
    f_in     = pulp.lpSum([f_Line[n,s] for n in lines_into_bus[i]])
    f_out    = pulp.lpSum([f_Line[n,s] for n in lines_out_of_bus[i]])
    f_tx_in  = pulp.lpSum([f_Tx[n,s]   for n in tx_into_bus[i]])
    f_tx_out = pulp.lpSum([f_Tx[n,s]   for n in tx_out_of_bus[i]])
    milp += f_Vir[i,s] + f_in + f_tx_in - f_out - f_tx_out == s_N[i,s], f"fFlow_bal_{i}"

for line in all_lines:
    milp += f_Line[line['name'],s] <=  N_nodes * x_Line[line['name'],s]
    milp += f_Line[line['name'],s] >= -N_nodes * x_Line[line['name'],s]

for tx in transformers:
    milp += f_Tx[tx['name'],s] <=  N_nodes * s_N[tx['bus1'],s]
    milp += f_Tx[tx['name'],s] >= -N_nodes * s_N[tx['bus1'],s]

# ── Connectivity propagation: hardwired lines + transformers ──────────────────
for line in all_lines:
    name = line['name']
    if name in faulted_lines or name in switch_names:
        continue
    milp += s_N[line['bus1'], s] == s_N[line['bus2'], s], f"BusProp_{name}"

for tx in transformers:
    milp += s_N[tx['bus1'], s] == s_N[tx['bus2'], s], f"TxProp_{tx['name']}"

# ── Voltage bounds ────────────────────────────────────────────────────────────
for i in nodes:
    for p in phases:
        milp += U_N[i,p,s] <= s_N[i,s] * U_max,  f"Vmax_{i}_{p}"
        milp += U_N[i,p,s] >= s_N[i,s] * U_min,  f"Vmin_{i}_{p}"

# ── Switch connectivity ───────────────────────────────────────────────────────
for sw in switchable_lines:
    milp += x_BR[sw['name'],s] <= s_N[sw['bus1'],s],  f"SW_conn_b1_{sw['name']}"
    milp += x_BR[sw['name'],s] <= s_N[sw['bus2'],s],  f"SW_conn_b2_{sw['name']}"

# ── Capacitor status ──────────────────────────────────────────────────────────
for cap in capacitors:
    milp += x_Cap[cap['name'],s] <= s_N[cap['bus'],s],  f"Cap_gate_{cap['name']}"

# ── Power balance and LinDistFlow voltage-drop constraints ────────────────────
for p in phases:
    for i in nodes:
        p_in  = pulp.lpSum([P_Line[ln,p,s] for ln in lines_into_bus[i]])
        p_out = pulp.lpSum([P_Line[ln,p,s] for ln in lines_out_of_bus[i]])
        q_in  = pulp.lpSum([Q_Line[ln,p,s] for ln in lines_into_bus[i]])
        q_out = pulp.lpSum([Q_Line[ln,p,s] for ln in lines_out_of_bus[i]])
        p_tx_in  = pulp.lpSum([P_Tx[n,p,s] for n in tx_into_bus[i]])
        p_tx_out = pulp.lpSum([P_Tx[n,p,s] for n in tx_out_of_bus[i]])
        q_tx_in  = pulp.lpSum([Q_Tx[n,p,s] for n in tx_into_bus[i]])
        q_tx_out = pulp.lpSum([Q_Tx[n,p,s] for n in tx_out_of_bus[i]])

        base_p = sum(load_weights[l]['kW_phase']   for l in loads_at_node[i] if p in load_weights[l]['phases'])
        base_q = sum(load_weights[l]['kvar_phase'] for l in loads_at_node[i] if p in load_weights[l]['phases'])
        p_served = base_p * s_N[i,s]
        q_served = base_q * s_N[i,s]

        p_dg = (P_G['MAIN_SUBSTATION',p,s] if i == SUB_BUS
                else (P_G[i,p,s] if i in candidate_buses else 0))
        q_dg = (Q_G['MAIN_SUBSTATION',p,s] if i == SUB_BUS
                else (Q_G[i,p,s] if i in candidate_buses else 0))

        q_cap = pulp.lpSum([
            x_Cap[cap['name'],s] * cap['kvar_phase']
            for cap in capacitors if cap['bus'] == i and p in cap['phases']
        ])

        milp += p_in + p_dg + p_tx_in - p_out - p_tx_out - p_served <=  M_Power*(1-s_N[i,s]), f"PBal_hi_{i}_{p}"
        milp += p_in + p_dg + p_tx_in - p_out - p_tx_out - p_served >= -M_Power*(1-s_N[i,s]), f"PBal_lo_{i}_{p}"
        milp += q_in + q_dg + q_cap + q_tx_in - q_out - q_tx_out - q_served <=  M_Power*(1-s_N[i,s]), f"QBal_hi_{i}_{p}"
        milp += q_in + q_dg + q_cap + q_tx_in - q_out - q_tx_out - q_served >= -M_Power*(1-s_N[i,s]), f"QBal_lo_{i}_{p}"

    for line in all_lines:
        name, b1, b2 = line['name'], line['bus1'], line['bus2']
        if p not in line['phases']:
            milp += P_Line[name,p,s] == 0, f"PLine_NoPhase_{name}_{p}"
            milp += Q_Line[name,p,s] == 0, f"QLine_NoPhase_{name}_{p}"
            continue

        v_drop = 0
        for m in line['phases']:
            r_t = (alpha_real[p][m] * line['r_matrix'][p][m]
                   - alpha_imag[p][m] * line['x_matrix'][p][m])
            x_t = (alpha_real[p][m] * line['x_matrix'][p][m]
                   + alpha_imag[p][m] * line['r_matrix'][p][m])
            v_drop += 2 * (r_t * P_Line[name,m,s] + x_t * Q_Line[name,m,s]) / VDROP_DENOM

        if name in faulted_lines:
            milp += P_Line[name,p,s] == 0, f"Fault_PLine_{name}_{p}"
            milp += Q_Line[name,p,s] == 0, f"Fault_QLine_{name}_{p}"
        elif name in switch_names:
            milp += U_N[b1,p,s] - U_N[b2,p,s] <=  v_drop + M_Volt*(1-x_BR[name,s])
            milp += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop  - M_Volt*(1-x_BR[name,s])
            milp += P_Line[name,p,s] <=  M_Power*x_BR[name,s]
            milp += P_Line[name,p,s] >= -M_Power*x_BR[name,s]
            milp += Q_Line[name,p,s] <=  M_Power*x_BR[name,s]
            milp += Q_Line[name,p,s] >= -M_Power*x_BR[name,s]
        else:
            milp += U_N[b1,p,s] - U_N[b2,p,s] <=  v_drop + M_Volt*(1-x_Line[name,s])
            milp += U_N[b1,p,s] - U_N[b2,p,s] >= v_drop  - M_Volt*(1-x_Line[name,s])
            milp += P_Line[name,p,s] <=  M_Power*x_Line[name,s]
            milp += P_Line[name,p,s] >= -M_Power*x_Line[name,s]
            milp += Q_Line[name,p,s] <=  M_Power*x_Line[name,s]
            milp += Q_Line[name,p,s] >= -M_Power*x_Line[name,s]

    # ── Transformer voltage tie (gate on BOTH buses) ──────────────────────────
    for tx in transformers:
        name, b1, b2 = tx['name'], tx['bus1'], tx['bus2']
        if p not in tx['phases']:
            continue
        a_sq = tx['a_ratio'][p] ** 2
        if name in switch_names:
            milp += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] <=  M_Volt*(1 - x_BR[name,s])
            milp += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] >= -M_Volt*(1 - x_BR[name,s])
        else:
            t_live = s_N[b1,s] + s_N[b2,s]   # = 2 iff both alive
            milp += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] <=  M_Volt*(2-t_live)
            milp += (a_sq*U_N[b2,p,s]) - U_N[b1,p,s] >= -M_Volt*(2-t_live)

# ── Line thermal capacity (phase-independent constraints — added once per
# line/phase, outside the p-loop where they were previously triplicated) ──────
for line in all_lines:
    name = line['name']
    for ph in line['phases']:
        milp += ( sqrt3*P_Line[name,ph,s] + Q_Line[name,ph,s]) <=  2*LINE_CAPACITY
        milp += ( sqrt3*P_Line[name,ph,s] + Q_Line[name,ph,s]) >= -2*LINE_CAPACITY
        milp += ( sqrt3*P_Line[name,ph,s] - Q_Line[name,ph,s]) <=  2*LINE_CAPACITY
        milp += ( sqrt3*P_Line[name,ph,s] - Q_Line[name,ph,s]) >= -2*LINE_CAPACITY
        milp += P_Line[name,ph,s] <=  LINE_CAPACITY
        milp += P_Line[name,ph,s] >= -LINE_CAPACITY
        milp += Q_Line[name,ph,s] <=  LINE_CAPACITY
        milp += Q_Line[name,ph,s] >= -LINE_CAPACITY

# Transformer power flow: M-gate + octagonal thermal capacity
for tx in transformers:
    name, b1 = tx['name'], tx['bus1']
    for ph in tx['phases']:
        milp += P_Tx[name,ph,s] <=  M_Power * s_N[b1,s]
        milp += P_Tx[name,ph,s] >= -M_Power * s_N[b1,s]
        milp += Q_Tx[name,ph,s] <=  M_Power * s_N[b1,s]
        milp += Q_Tx[name,ph,s] >= -M_Power * s_N[b1,s]
        milp += ( sqrt3*P_Tx[name,ph,s] + Q_Tx[name,ph,s]) <=  2*TX_CAPACITY
        milp += ( sqrt3*P_Tx[name,ph,s] + Q_Tx[name,ph,s]) >= -2*TX_CAPACITY
        milp += ( sqrt3*P_Tx[name,ph,s] - Q_Tx[name,ph,s]) <=  2*TX_CAPACITY
        milp += ( sqrt3*P_Tx[name,ph,s] - Q_Tx[name,ph,s]) >= -2*TX_CAPACITY
        milp += P_Tx[name,ph,s] <=  TX_CAPACITY
        milp += P_Tx[name,ph,s] >= -TX_CAPACITY
        milp += Q_Tx[name,ph,s] <=  TX_CAPACITY
        milp += Q_Tx[name,ph,s] >= -TX_CAPACITY

# =============================================================================
# STEP 4 — Solve
# =============================================================================
print("\n⚙️  Solving MILP...")
milp.solve(pulp.GUROBI_CMD(msg=True, options=[("MIPGap", 0.001), ("TimeLimit", 300)]))

status_str = pulp.LpStatus[milp.status]
if status_str != 'Optimal':
    print(f"❌ MILP did not find an optimal solution (status: {status_str}).")
    exit()

print("✅ MILP solved successfully!")

# =============================================================================
# STEP 4b — MILP Solution Summary  (observe what the solver decided)
# =============================================================================
print(f"\n{'='*65}")
print(f"  MILP RESULTS — {SCENARIO_KEY}")
print(f"{'='*65}")

# ── Topology ─────────────────────────────────────────────────────────────────
live_buses = [i for i in nodes if (pulp.value(s_N[i,s]) or 0) > 0.5]
dead_buses = [i for i in nodes if (pulp.value(s_N[i,s]) or 0) < 0.5]
print(f"\n  Bus status:  {len(live_buses)} alive / {len(dead_buses)} dead")
print(f"  Dead buses:  {sorted(dead_buses)}")

# ── DG dispatch ──────────────────────────────────────────────────────────────
print(f"\n  DG Dispatch:")
for dg in purchased_dgs:
    alive     = (pulp.value(s_N.get((dg,s), 0)) or 0) > 0.5
    p_a       = pulp.value(P_G[dg,'a',s]) or 0
    p_b       = pulp.value(P_G[dg,'b',s]) or 0
    p_c       = pulp.value(P_G[dg,'c',s]) or 0
    p_total   = p_a + p_b + p_c
    q_a       = pulp.value(Q_G[dg,'a',s]) or 0
    cap       = float(dg_sizes.get(dg, dg_sizes.get(str(dg), 0)))
    status    = "🟢 ACTIVE" if alive else "🔴 DEAD"
    print(f"    Bus {dg:>6s}: {status}  |  P = {p_total:6.1f} kW / {cap:.0f} kVA rated  "
          f"|  Q_a = {q_a:5.1f} kVAr")

# ── Switch decisions ──────────────────────────────────────────────────────────
print(f"\n  Switch Decisions:")
for sw in switchable_lines:
    val    = (pulp.value(x_BR[sw['name'],s]) or 0)
    status = "CLOSED" if val > 0.5 else "OPEN"
    b1_live = (pulp.value(s_N.get((sw['bus1'],s),0)) or 0) > 0.5
    b2_live = (pulp.value(s_N.get((sw['bus2'],s),0)) or 0) > 0.5
    print(f"    {sw['name']:12s}: {status}  ({sw['bus1']}:{'+' if b1_live else '-'}  "
          f"↔  {sw['bus2']}:{'+' if b2_live else '-'})")

# ── Load shedding ─────────────────────────────────────────────────────────────
total_shed_kw = total_restored_kw = 0.0
for l in loads:
    bus    = load_weights[l]['bus']
    kw     = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
    alive  = (pulp.value(s_N.get((bus,s),0)) or 0) > 0.5
    if alive:
        total_restored_kw += kw
    else:
        total_shed_kw += kw

pct_restored = total_restored_kw / total_grid_kw * 100 if total_grid_kw > 0 else 0
pct_shed     = total_shed_kw     / total_grid_kw * 100 if total_grid_kw > 0 else 0

print(f"\n  Load Restoration:")
print(f"    Restored : {total_restored_kw:8.1f} kW  ({pct_restored:.1f}%)")
print(f"    Shed     : {total_shed_kw:8.1f} kW  ({pct_shed:.1f}%)")
print(f"    Total    : {total_grid_kw:8.1f} kW")

# ── Per-bus voltage table ─────────────────────────────────────────────────────
print(f"\n  Per-Bus Voltages (MILP √U_N, live buses only):")
print(f"  {'Bus':<10} {'Ph-A pu':>8} {'Ph-B pu':>8} {'Ph-C pu':>8}  Load kW")
for i in sorted(live_buses):
    va = pulp.value(U_N.get((i,'a',s), None))
    vb = pulp.value(U_N.get((i,'b',s), None))
    vc = pulp.value(U_N.get((i,'c',s), None))
    va_str = f"{math.sqrt(va):7.4f}" if va and va > 1e-6 else "   —   "
    vb_str = f"{math.sqrt(vb):7.4f}" if vb and vb > 1e-6 else "   —   "
    vc_str = f"{math.sqrt(vc):7.4f}" if vc and vc > 1e-6 else "   —   "
    kw = sum(load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
             for l in loads_at_node.get(i,[]))
    print(f"  {i:<10} {va_str} {vb_str} {vc_str}  {kw:7.1f}")

print(f"{'='*65}")

# =============================================================================
# STEP 5 — OpenDSS physics validation
# =============================================================================
print("\n🔬 Running OpenDSS physics validation...")
dss.Command('Clear')
dss.Command(f'Compile "{DSS_FILE}"')

_raw_kv = bus_phase_info.get('701', {}).get('kv_base', 2.771)
if _raw_kv < 3.5:                                   # L-N → convert to L-L
    system_kv_LL = round(_raw_kv * math.sqrt(3), 4)
else:                                               # already L-L
    system_kv_LL = round(_raw_kv, 4)
if not (1.0 <= system_kv_LL <= 15.0):
    system_kv_LL = 4.8
    print(f"   ⚠ kVBase out of expected range — falling back to 4.8 kV")
print(f"   Bus 701 kVBase raw = {_raw_kv:.4f} kV  →  system_kv_LL = {system_kv_LL:.4f} kV")
print(f"   DG bus kVBase values: " +
      ", ".join(f"{g}={bus_phase_info.get(g,{}).get('kv_base',0):.4f}" for g in purchased_dgs))

# ─── 1. Kill the main grid ───────────────────────────────────────────────────
dss.Vsources.First()
for _ in range(dss.Vsources.Count()):
    dss.Command(f'Edit Vsource.{dss.Vsources.Name()} enabled=no')
    dss.Vsources.Next()

# ─── 2. Disable all loads, capacitors, transformers ─────────────────────────
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

# ─── 3. Open ALL line terminals ─────────────────────────────────────────────
dss.Lines.First()
for _ in range(dss.Lines.Count()):
    lname = dss.Lines.Name()
    dss.Command(f'Open Line.{lname} terminal=1')
    dss.Command(f'Open Line.{lname} terminal=2')
    dss.Lines.Next()

# ─── 4. Island detection from MILP live topology ────────────────────────────
G_live = nx.Graph()
for line in all_lines:
    name = line['name']
    if name in faulted_lines: continue
    if name in switch_names:
        if (pulp.value(x_BR[name, s]) or 0) > 0.5: G_live.add_edge(line['bus1'], line['bus2'])
    else:
        if (pulp.value(x_Line[name, s]) or 0) > 0.5: G_live.add_edge(line['bus1'], line['bus2'])
for tx in transformers:
    b1, b2 = tx['bus1'], tx['bus2']
    if ((pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5 and
            (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5):
        G_live.add_edge(b1, b2)
for bus in nodes:
    if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5:
        G_live.add_node(bus)

live_dgs = [g for g in purchased_dgs if (pulp.value(s_N.get((g, s), 0)) or 0) > 0.5]
live_components = list(nx.connected_components(G_live))
bus_island = {}
for comp in live_components:
    root = min(comp)
    for b in comp:
        bus_island[b] = root
island_to_dgs: dict = {}
for g in live_dgs:
    if g in bus_island:
        island_to_dgs.setdefault(bus_island[g], []).append(g)

slack_dgs:    set = set()
nonslack_dgs: set = set()
for dg_list in island_to_dgs.values():
    slack = max(dg_list, key=lambda g: float(dg_sizes.get(g, dg_sizes.get(str(g), 0))))
    slack_dgs.add(slack)
    for g in dg_list:
        if g != slack:
            nonslack_dgs.add(g)

print(f"   Islands: {len(island_to_dgs)}  |  "
      f"Slack Vsources: {sorted(slack_dgs)}  |  "
      f"Non-slack (omitted): {sorted(nonslack_dgs)}")
for key, dg_list in island_to_dgs.items():
    comp_buses = sorted(next((c for c in live_components if key in c), set()))
    print(f"     [{key}] buses={comp_buses}  DGs={dg_list}")

# ─── 5. One Wye Vsource per island (slack DG only) ──────────────────────────
for g in slack_dgs:
    _g_raw = bus_phase_info.get(g, {}).get('kv_base', 2.771)
    if _g_raw < 3.5:                    # L-N reading → convert to L-L
        g_kv_LL = round(_g_raw * math.sqrt(3), 4)
    else:                               # already L-L
        g_kv_LL = round(_g_raw, 4)
    if not (0.05 <= g_kv_LL <= 15.0):  # sanity fallback
        g_kv_LL = system_kv_LL
    dss.Command(
        f'New Vsource.DG_{g} bus1={g} basekv={g_kv_LL:.4f} pu=1.0 phases=3 '
        f'R1=1e-4 X1=1e-4 R0=1e-4 X0=1e-4 enabled=yes'
    )
    print(f"   GFM Vsource @ bus {g}  basekv={g_kv_LL:.4f} kV L-L")

# ─── 5b. GFL Generator (constant PQ) for co-island non-slack DGs ─────────────
# One Vsource per island acts as GFM (voltage/frequency reference).
# Co-island DGs are modeled as Generator model=1 injecting the MILP-dispatched
# P and Q so OpenDSS can report per-DG power without inflating the slack Vsource.
for g in nonslack_dgs:
    _g_raw = bus_phase_info.get(g, {}).get('kv_base', 2.771)
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
    print(f"   GFL Generator @ bus {g}  basekv={g_kv_LL:.4f} kV L-L  "
          f"P_milp={p_milp:.1f} kW  Q_milp={q_milp:.1f} kvar")

# ─── 6. Close MILP-active lines (BOTH terminals — avoids half-open bug) ──────
n_closed = 0
for line in all_lines:
    name = line['name']
    if name in faulted_lines: continue
    if name in switch_names:
        active = (pulp.value(x_BR[name, s]) or 0) > 0.5
    else:
        active = (pulp.value(x_Line[name, s]) or 0) > 0.5
    if active:
        dss.Command(f'Close Line.{name} terminal=1')
        dss.Command(f'Close Line.{name} terminal=2')
        n_closed += 1
print(f"   Lines closed: {n_closed}")

# ─── 7. Transformers, loads, capacitors ─────────────────────────────────────
for tx in transformers:
    b1, b2 = tx['bus1'], tx['bus2']
    b1_alive = (pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5
    b2_alive = (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5
    # Both sides must be alive: enabling a transformer with one dead side
    # leaves a floating HV winding that OpenDSS tries to magnetize,
    # producing large circulating currents and convergence failures.
    if b1_alive and b2_alive:
        dss.Command(f'Edit Transformer.{tx["name"]} enabled=yes')
    else:
        dss.Command(f'Edit Transformer.{tx["name"]} enabled=no')

# Iterate the Python `loads` list, NOT dss.Loads.First()/Next(): all loads
# have been disabled by this point, and OpenDSS's Loads iterator only traverses
# ENABLED elements, so the cursor freezes on the last load and no loads ever get
# re-enabled (silent zero-load validation).
for lname in loads:
    bus = load_weights[lname]['bus']
    if (pulp.value(s_N.get((bus, s), 0)) or 0) > 0.5:
        dss.Command(f'Edit Load.{lname} enabled=yes')

for cap in capacitors:
    if (pulp.value(x_Cap[cap['name'], s]) or 0) > 0.5:
        dss.Command(f'Edit Capacitor.{cap["name"]} enabled=yes')

dss.Command('set controlmode=off')
dss.Solution.Solve()

total_losses_kw               = float('nan')
opendss_min_v, opendss_max_v  = 0.0, 0.0
opendss_avg_v                 = 0.0
opendss_vpu: dict[str, float] = {}
opendss_phase_vpu: dict[str, dict] = {}

if not dss.Solution.Converged():
    print("   ⚠ OpenDSS did not converge — skipping validation table.")
else:
    # ── 1. Losses via native API ──────────────────────────────────────────────
    total_losses_kw = dss.Circuit.Losses()[0]    / 1000.0
    line_losses_kw  = dss.Circuit.LineLosses()[0] / 1000.0
    tx_losses_kw    = total_losses_kw - line_losses_kw

    # ── 2. Collect per-bus phase voltages ─────────────────────────────────────
    live_set  = {i for i in nodes if (pulp.value(s_N.get((i, s), 0)) or 0) > 0.5}
    live_mags = []
    for bus_name in dss.Circuit.AllBusNames():
        bkey = bus_name.lower().strip()
        if bkey not in live_set:
            continue
        dss.Circuit.SetActiveBus(bus_name)
        pu_v      = dss.Bus.puVmagAngle()
        bus_nodes = dss.Bus.Nodes()
        phase_mag = {}
        for idx, nd in enumerate(bus_nodes):
            if nd in (1, 2, 3):
                mag = pu_v[idx * 2]
                if 0.05 < mag < 2.0:
                    phase_mag[nd] = mag
        if phase_mag:
            opendss_phase_vpu[bkey] = phase_mag
            opendss_vpu[bkey]       = sum(phase_mag.values()) / len(phase_mag)
            live_mags.extend(phase_mag.values())
    if live_mags:
        opendss_min_v = min(live_mags)
        opendss_max_v = max(live_mags)
        opendss_avg_v = sum(live_mags) / len(live_mags)

    # ── 3. Voltage unbalance: NEMA (max phase deviation / average × 100) ──────
    vub_list = []
    for bkey, ph_map in opendss_phase_vpu.items():
        if len(ph_map) < 3:
            continue
        v_vals = list(ph_map.values())
        v_avg  = sum(v_vals) / 3.0
        if v_avg > 1e-3:
            vub = max(abs(v - v_avg) for v in v_vals) / v_avg * 100.0
            vub_list.append((bkey, vub))
    vub_list.sort(key=lambda x: x[1], reverse=True)
    max_vub_pct = vub_list[0][1] if vub_list else 0.0
    max_vub_bus = vub_list[0][0] if vub_list else 'N/A'

    # ── 4. DG power readback ──────────────────────────────────────────────────
    # Circuit.TotalPower() returns aggregate power from Vsource elements only.
    # GFL Generators are PC elements (not Vsources) so must be read separately.
    p_vsource_kw = -dss.Circuit.TotalPower()[0]
    p_gen_kw     = 0.0
    dg_rows      = []
    for g in sorted(live_dgs):
        p_milp = sum((pulp.value(P_G[g, p, s]) or 0.0) for p in phases)
        q_milp = sum((pulp.value(Q_G[g, p, s]) or 0.0) for p in phases)
        s_cap  = float(dg_sizes.get(g, dg_sizes.get(str(g), 0)))
        p_max  = 0.8 * s_cap
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
        s_act   = math.sqrt(p_act**2 + q_act**2)
        exceed  = p_act > p_max * 1.05 or s_act > s_cap * 1.05
        dg_rows.append((tag, g, s_cap, p_max, p_milp, q_milp, p_act, q_act, s_act,
                        '⚠ EXCEED' if exceed else '✓'))
    p_dg_total = p_vsource_kw + p_gen_kw

    # ── 5. Served load: iterate all Load elements, sum terminal-1 real power ──
    # OpenDSS convention: positive = power entering the element from the bus,
    # so loads report positive P at terminal 1. Disabled loads return zeros.
    p_load_kw = 0.0
    if dss.Loads.First():
        while True:
            dss.Circuit.SetActiveElement(f'Load.{dss.Loads.Name()}')
            pows      = dss.CktElement.Powers()
            nc        = dss.CktElement.NumConductors()
            p_load_kw += sum(pows[2*k] for k in range(min(3, nc)))
            if not dss.Loads.Next():
                break

    # Energy identity: P_DG = P_load + P_loss  (OpenDSS enforces this internally)
    bal_err = p_dg_total - p_load_kw - total_losses_kw

    # MILP restored load (reference for comparison, not used in balance)
    milp_restored_kw = sum(
        load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
        * (1 if (pulp.value(s_N.get((load_weights[l]['bus'], s), 0)) or 0) > 0.5 else 0)
        for l in loads
    )

    # ── 6. Branch loading + phase currents ───────────────────────────────────
    branch_rows = []
    dss.Lines.First()
    while True:
        lname = dss.Lines.Name()
        dss.Circuit.SetActiveElement(f'Line.{lname}')
        pows  = dss.CktElement.Powers()
        nc    = dss.CktElement.NumConductors()
        n_ph  = min(3, nc)                          # phases only, skip neutral
        s_ph  = [math.sqrt(pows[2*k]**2 + pows[2*k+1]**2) for k in range(n_ph)]
        s_tot = sum(s_ph)
        s_max = max(s_ph) if s_ph else 0.0
        load_pct = s_max / LINE_CAPACITY * 100.0   # worst-phase vs per-phase limit
        cur   = dss.CktElement.CurrentsMagAng()
        i_ph  = [cur[2*k] for k in range(n_ph)]          # terminal-1 magnitudes (A)
        i_max = max(i_ph) if i_ph else 0.0
        branch_rows.append((lname, s_tot, s_max, i_max, load_pct))
        if not dss.Lines.Next():
            break
    branch_rows.sort(key=lambda x: x[1], reverse=True)
    active_branches = [(n, s, sm, im, p) for n, s, sm, im, p in branch_rows if s > 0.5]

    # ── 7. Transformer loading (XFM1: 709→775, rated 500 kVA) ────────────────
    tx_alive = (
        (pulp.value(s_N.get(('709', s), 0)) or 0) > 0.5 and
        (pulp.value(s_N.get(('775', s), 0)) or 0) > 0.5
    )
    tx_s_kva    = 0.0
    tx_load_pct = 0.0
    if tx_alive:
        dss.Circuit.SetActiveElement('Transformer.XFM1')
        pows        = dss.CktElement.Powers()
        p_tx        = sum(pows[2*k]   for k in range(3))
        q_tx        = sum(pows[2*k+1] for k in range(3))
        tx_s_kva    = math.sqrt(p_tx**2 + q_tx**2)
        tx_load_pct = tx_s_kva / 500.0 * 100.0

    # ── 8. Print summary table ────────────────────────────────────────────────
    W   = 68
    SEP = '  ' + '─' * W
    print()
    print(SEP)
    print(f"  OpenDSS Validation ─ {SCENARIO_KEY}")
    print(SEP)

    print("  POWER BALANCE")
    print(f"    P_DG   = GFM {p_vsource_kw:>7.1f} kW  +  GFL {p_gen_kw:>7.1f} kW"
          f"  =  {p_dg_total:>7.1f} kW")
    print(f"    P_load   (load elements, terminal-1 sum)      =  {p_load_kw:>7.1f} kW")
    if tx_alive:
        print(f"    P_loss   = line {line_losses_kw:>5.2f}  +  trafo {tx_losses_kw:>5.2f}"
              f"            =  {total_losses_kw:>6.2f} kW  [OpenDSS native]")
    else:
        print(f"    P_loss   = {total_losses_kw:>6.2f} kW  (line losses only, trafo not energized)"
              f"  [OpenDSS native]")
    print(f"    P_DG − P_load − P_loss  (should be ≈0)       =  {bal_err:>+7.2f} kW"
          f"  {'✓' if abs(bal_err) < 2.0 else '⚠'}")
    print(f"    MILP restored (reference)                     =  {milp_restored_kw:>7.1f} kW")
    print(SEP)

    print("  DG OUTPUT   [P_max = 0.8×S_cap  |  ⚠ if P_ODS or S_ODS > 1.05× limit]")
    print(f"  {'Typ':3}  {'Bus':>5}  {'S_cap kVA':>9}  {'P_max kW':>8}  "
          f"{'P_MILP':>7}  {'Q_MILP':>7}  {'P_ODS':>7}  {'Q_ODS':>7}  {'S_ODS kVA':>9}")
    print('  ' + '─' * W)
    for tag, g, s_cap, p_max, p_milp, q_milp, p_act, q_act, s_act, status in dg_rows:
        print(f"  {tag:3}  {g:>5}  {s_cap:>9.1f}  {p_max:>8.1f}  "
              f"{p_milp:>7.1f}  {q_milp:>7.1f}  {p_act:>7.1f}  {q_act:>7.1f}  "
              f"{s_act:>9.1f}  {status}")
    print(SEP)

    print("  VOLTAGE (pu, OpenDSS phase magnitudes)")
    print(f"    Min {opendss_min_v:.4f}   Max {opendss_max_v:.4f}   Avg {opendss_avg_v:.4f}")
    if vub_list:
        vub_flag = '✓ <2%' if max_vub_pct < 2.0 else '⚠ >2%'
        print(f"    Voltage unbalance NEMA: max {max_vub_pct:.2f}% @ bus {max_vub_bus}"
              f"   {vub_flag}")
    print(SEP)

    print("  BRANCH LOADING  [limit: 1500 kVA 3-phase | 500 kVA/phase | ⚠ if >100%]")
    print(f"  {'Line':<18}  {'S_3ph kVA':>9}  {'S_max_ph kVA':>12}  {'I_max A':>7}  {'Load%':>6}")
    print('  ' + '─' * W)
    for lname, s_tot, s_max, i_max, load_pct in active_branches[:12]:
        flag = '  ⚠' if load_pct > 100.0 else ''
        print(f"  {lname:<18}  {s_tot:>9.1f}  {s_max:>12.1f}  {i_max:>7.1f}  "
              f"{load_pct:>5.1f}%{flag}")
    print(SEP)

    if tx_alive:
        tx_flag = '  ⚠ OVERLOAD' if tx_load_pct > 100.0 else '  ✓'
        print(f"  TRANSFORMER XFM1 (709→775, rated 500 kVA): "
              f"{tx_s_kva:.1f} kVA  {tx_load_pct:.1f}%{tx_flag}")
        # Per-phase MILP vs OpenDSS comparison
        # pows[0,2,4] = Pa,Pb,Pc (kW)  pows[1,3,5] = Qa,Qb,Qc (kVAR) at terminal-1 (bus 709)
        print(f"  {'Ph':>2}  {'MILP P_Tx kW':>12}  {'ODS P kW':>9}  "
              f"{'MILP Q_Tx kVAR':>14}  {'ODS Q kVAR':>11}  {'diff P':>7}")
        ph_idx = {'a': 0, 'b': 1, 'c': 2}
        for ph in phases:
            k = ph_idx[ph]
            m_p = pulp.value(P_Tx.get(('xfm1', ph, s), 0)) or 0.0
            m_q = pulp.value(Q_Tx.get(('xfm1', ph, s), 0)) or 0.0
            o_p = pows[2*k]
            o_q = pows[2*k+1]
            diff = m_p - o_p
            flag = '  ⚠' if abs(diff) > 5.0 else ''
            print(f"  {ph:>2}  {m_p:>12.2f}  {o_p:>9.2f}  "
                  f"{m_q:>14.2f}  {o_q:>11.2f}  {diff:>+7.2f}{flag}")
    else:
        print("  TRANSFORMER XFM1 (709→775): not energized")
    print(SEP)


def plot_opendss_phase_debugger(scenario_name):
    """
    Per-phase voltage bar chart: OpenDSS (solid) vs MILP (hatched) side-by-side
    """
    live_set = sorted(i for i in nodes
                      if (pulp.value(s_N.get((i, scenario_name), 0)) or 0) > 0.5)
    n = len(live_set)
    if n == 0:
        print("   ⚠ No live buses to plot.")
        return

    ods = {'a': [], 'b': [], 'c': []}
    mlp = {'a': [], 'b': [], 'c': []}
    phase_node = {'a': 1, 'b': 2, 'c': 3}

    for b in live_set:
        ph_map = opendss_phase_vpu.get(b, {})
        for ph, nd in phase_node.items():
            ods[ph].append(ph_map.get(nd, 0.0))
            raw = pulp.value(U_N.get((b, ph, scenario_name), 0)) or 0
            mlp[ph].append(math.sqrt(raw) if raw > 1e-6 else 0.0)

    x     = np.arange(n)
    w     = 0.13   
    COLOR = {'a': '#e74c3c', 'b': '#2ecc71', 'c': '#3498db'}
    LABEL = {'a': 'Ph-A', 'b': 'Ph-B', 'c': 'Ph-C'}
    offsets = {'a': -2, 'b': 0, 'c': 2}   

    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))

    for ph in ['a', 'b', 'c']:
        off = offsets[ph] * w
        colors_ods = [COLOR[ph] if 0.95 <= v <= 1.05 else '#e74c3c'
                      if v < 0.95 else '#e67e22' for v in ods[ph]]
        ax.bar(x + off - w/2, ods[ph], w, color=colors_ods,
               edgecolor='black', linewidth=0.3,
               label=f'ODS {LABEL[ph]}')
        ax.bar(x + off + w/2, mlp[ph], w, color=COLOR[ph],
               edgecolor='black', linewidth=0.3, alpha=0.35,
               hatch='//', label=f'MLP {LABEL[ph]}')

    ax.axhline(1.05, color='red',    lw=1.0, ls='--', alpha=0.6, label='±5 % band')
    ax.axhline(0.95, color='red',    lw=1.0, ls='--', alpha=0.6)
    ax.axhline(1.00, color='black',  lw=0.7, ls=':',  alpha=0.4)

    for dg in purchased_dgs:
        if dg in live_set:
            idx = live_set.index(dg)
            role = 'Slack' if dg in slack_dgs else 'Non-slack'
            ax.annotate(f'⚡{role}', xy=(idx, 1.004),
                        fontsize=10, ha='center', color='darkgreen', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(live_set, rotation=0, fontsize=9)
    ax.set_ylabel('Voltage magnitude (pu)', fontsize=11)
    ax.set_title(f'OpenDSS vs MILP Phase Voltages — {scenario_name}\n'
                 f'Solid = OpenDSS  |  Hatched = MILP  |  Red bar = outside [0.95, 1.05]',
                 fontsize=11)
    ax.set_ylim(0.95, 1.01)
    ax.grid(axis='y', ls='--', alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(f'phase_debug_{scenario_name}.png', dpi=150, bbox_inches='tight')


# Always call the phase debugger so per-phase deviations are visible
plot_opendss_phase_debugger(s)

# =============================================================================
# STEP 6 — Build island graph for colouring
# =============================================================================
G_alive = nx.Graph()
for line in all_lines:
    if (pulp.value(x_Line[line['name'],s]) or 0) > 0.5:
        G_alive.add_edge(line['bus1'], line['bus2'])
for tx in transformers:
    b1, b2 = tx['bus1'], tx['bus2']
    if ((pulp.value(s_N.get((b1,s),0)) or 0) > 0.5 and
            (pulp.value(s_N.get((b2,s),0)) or 0) > 0.5):
        G_alive.add_edge(b1, b2)

islands = [c for c in nx.connected_components(G_alive)]

# =============================================================================
# STEP 7 — Visualisation
# =============================================================================

# ── MILP voltage lookup (√U_N) — kept for reference ─────────────────────────
bus_voltage_pu = {}
for i in nodes:
    if (pulp.value(s_N.get((i,s),0)) or 0) > 0.5:
        vals = [
            math.sqrt(pulp.value(U_N[i,p,s]))
            for p in phases
            if (pulp.value(U_N.get((i,p,s), 0)) or 0) > 1e-6
        ]
        if vals:
            bus_voltage_pu[i] = sum(vals) / len(vals)

# ── Restored kW per bus ───────────────────────────────────────────────────────
bus_restored_kw = {b: 0.0 for b in nodes}
for l in loads:
    bus = load_weights[l]['bus']
    if (pulp.value(s_N[bus,s]) or 0) > 0.5:
        bus_restored_kw[bus] += load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])

# ── Build full network graph ──────────────────────────────────────────────────
G_full = nx.Graph()
for line in all_lines:
    G_full.add_edge(line['bus1'], line['bus2'], ename=line['name'], etype='line')
for tx in transformers:
    G_full.add_edge(tx['bus1'], tx['bus2'], ename=tx['name'], etype='transformer')

# ── Draw positions: raw CSV coordinates — same layout as base_case.py ─────────
# Handle SUB_BUS alias (sourcebus / 799r / 799 all share one physical point)
if SUB_BUS not in pos:
    for alias in ['799r', '799', 'sourcebus']:
        if alias != SUB_BUS and alias in pos:
            pos[SUB_BUS] = pos[alias]
            break

draw_pos = {b: pos[b] for b in G_full.nodes() if b in pos}

# Fallback spring layout for any buses without CSV coordinates
missing = [b for b in G_full.nodes() if b not in draw_pos]
if missing:
    sp = nx.spring_layout(G_full, seed=42, k=2.5)
    all_x_t = [v[0] for v in draw_pos.values()] if draw_pos else [0.0]
    all_y_t = [v[1] for v in draw_pos.values()] if draw_pos else [0.0]
    cx = (max(all_x_t) + min(all_x_t)) / 2
    cy = (max(all_y_t) + min(all_y_t)) / 2
    scale = max(max(all_x_t) - min(all_x_t), max(all_y_t) - min(all_y_t), 1) * 0.2
    for b in missing:
        if b in sp:
            draw_pos[b] = (cx + sp[b][0] * scale, cy + sp[b][1] * scale)

# ── Battery icon sizing — proportional to coordinate span ────────────────────
all_x_vals = [v[0] for v in draw_pos.values()]
all_y_vals = [v[1] for v in draw_pos.values()]
coord_range = max(max(all_x_vals) - min(all_x_vals),
                  max(all_y_vals) - min(all_y_vals), 1.0)
bat_w = coord_range * 0.020
bat_h = bat_w * 1.80


# ── Battery icon sizing ──────────────────────────────────────────────────────
all_x_vals = [v[0] for v in draw_pos.values()]
all_y_vals = [v[1] for v in draw_pos.values()]
coord_range = max(max(all_x_vals) - min(all_x_vals),
                  max(all_y_vals) - min(all_y_vals), 1.0)

# Made it wider and shorter (landscape)
bat_w = coord_range * 0.035
bat_h = bat_w * 0.65 

def draw_dg_battery(ax, node_x, node_y, alive, p_out, cap_kw, offset=(0, 0)):
    # Apply your manual offset to the node's location
    x = node_x + offset[0]
    y = node_y + offset[1]

    line_color = '#282C34' # Dark charcoal
    fill_color = '#FFFFFF' if alive else '#E5E7EB'
    lw = 1.5

    # Main battery body
    body = mpatches.FancyBboxPatch(
        (x - bat_w / 2, y - bat_h / 2), bat_w, bat_h,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        facecolor=fill_color, edgecolor=line_color, linewidth=lw,
        transform=ax.transData, zorder=6
    )
    ax.add_patch(body)

    # Top nub
    top_nub = mpatches.Rectangle(
        (x - bat_w * 0.15, y + bat_h / 2), bat_w * 0.3, bat_h * 0.15,
        facecolor=line_color, edgecolor=line_color,
        transform=ax.transData, zorder=5
    )
    ax.add_patch(top_nub)

    # Bottom nubs (left and right)
    bot_nub_l = mpatches.Rectangle(
        (x - bat_w * 0.25, y - bat_h * 0.65), bat_w * 0.15, bat_h * 0.15,
        facecolor=line_color, edgecolor=line_color,
        transform=ax.transData, zorder=5
    )
    bot_nub_r = mpatches.Rectangle(
        (x + bat_w * 0.10, y - bat_h * 0.65), bat_w * 0.15, bat_h * 0.15,
        facecolor=line_color, edgecolor=line_color,
        transform=ax.transData, zorder=5
    )
    ax.add_patch(bot_nub_l)
    ax.add_patch(bot_nub_r)

    # Lightning bolt shape
    bolt_verts = [
        (x + bat_w*0.08, y + bat_h*0.3), (x - bat_w*0.12, y - bat_h*0.05),
        (x + bat_w*0.02, y - bat_h*0.05), (x - bat_w*0.08, y - bat_h*0.3),
        (x + bat_w*0.12, y + bat_h*0.05), (x - bat_w*0.02, y + bat_h*0.05)
    ]
    bolt = plt.Polygon(bolt_verts, color=line_color, zorder=7)
    ax.add_patch(bolt)

    # Text below the battery
    status = 'ON' if alive else 'OFF'
    ax.annotate(f"DG {status}\n{p_out:.0f}/{cap_kw:.0f} kW",
                xy=(x, y - bat_h * 0.75),
                xytext=(0, -2), textcoords='offset points',
                fontsize=6, color=line_color, fontweight='bold',
                ha='center', va='top', zorder=8)


# ── Edge classification ───────────────────────────────────────────────────────
active_edges, faulted_edges, opened_sw, dead_isolated = [], [], [], []
for line in all_lines:
    b1, b2, name = line['bus1'], line['bus2'], line['name']
    if b1 not in draw_pos or b2 not in draw_pos:
        continue
    status = pulp.value(x_Line[name, s]) or 0
    if name in faulted_lines:
        faulted_edges.append((b1, b2))
    elif name in switch_names and status < 0.5:
        opened_sw.append((b1, b2))
    elif status < 0.5:
        dead_isolated.append((b1, b2))
    else:
        active_edges.append((b1, b2))

for tx in transformers:
    b1, b2 = tx['bus1'], tx['bus2']
    if b1 not in draw_pos or b2 not in draw_pos:
        continue
    if ((pulp.value(s_N.get((b1, s), 0)) or 0) > 0.5 and
            (pulp.value(s_N.get((b2, s), 0)) or 0) > 0.5):
        active_edges.append((b1, b2))
    else:
        dead_isolated.append((b1, b2))

# ── Node classification: tier-based colours (mirrors base_case.py) ───────────
TIER_3_THRESHOLD_VIZ = 85.0
TIER_2_THRESHOLD_VIZ = 50.0

bus_total_kw = {}
for l in loads:
    bus = load_weights[l]['bus']
    kw = load_weights[l]['kW_phase'] * len(load_weights[l]['phases'])
    bus_total_kw[bus] = bus_total_kw.get(bus, 0.0) + kw

C_NODE = {
    'no_load': '#D1D5DB', 'tier1': '#22C55E',
    'tier2':   '#F59E0B', 'tier3': '#EF4444',
}
EC_NODE = {
    'no_load': '#9CA3AF', 'tier1': '#15803D',
    'tier2':   '#B45309', 'tier3': '#991B1B',
}
NODE_SIZES = {'no_load': 80, 'tier1': 160, 'tier2': 210, 'tier3': 260}


def tier_class(b):
    kw = bus_total_kw.get(b, 0.0)
    if kw == 0:                         return 'no_load'
    if kw > TIER_3_THRESHOLD_VIZ:       return 'tier3'
    if kw >= TIER_2_THRESHOLD_VIZ:      return 'tier2'
    return 'tier1'


# Dead buses — all non-substation buses with s_N = 0
dead_nodes = [b for b in nodes
              if (pulp.value(s_N[b, s]) or 0) < 0.5
              and b in draw_pos and b != SUB_BUS]

# Alive non-substation nodes — DG buses included (battery drawn beside them)
alive_nodes = [b for b in nodes
               if (pulp.value(s_N[b, s]) or 0) > 0.5
               and b in draw_pos and b != SUB_BUS]

# Bucket by tier so each group uses one draw call
tier_buckets: dict = {'no_load': [], 'tier1': [], 'tier2': [], 'tier3': []}
for b in alive_nodes:
    tier_buckets[tier_class(b)].append(b)

# ── Label offsets — identical to base_case.py for consistent layout ───────────
custom_offsets = {
    '701': (14, -10),  '702': (12, 10),   '720': (14, -10),
    '706': (14, -6),   '707': (14, -6),   '714': (14, -10),
    '704': (11, -11),  '718': (14, -10),  '703': (14, -6),
    '730': (14, -10),  '741': (14, -10),  '705': (0, -12),
    '728': (-14, -10), '732': (-14, -10), '736': (-14, -10),
    '710': (-14, -6),  '737': (-14, -10), '711': (11, 9),
    '727': (0, -14),   '744': (11, 11),   '735': (-14, -10),
    '733': (14, -10),  '713': (0, -14),   '734': (14, -10),
    '709': (0, -14),   '725': (14, -10),  '722': (0, -14),
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(17, 12))
fig1.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# ── Edges ─────────────────────────────────────────────────────────────────────
if dead_isolated:
    nx.draw_networkx_edges(G_full, draw_pos, ax=ax, edgelist=dead_isolated,
                           edge_color='#D1D5DB', width=1.2, style='dotted', alpha=0.7)
if faulted_edges:
    nx.draw_networkx_edges(G_full, draw_pos, ax=ax, edgelist=faulted_edges,
                           edge_color='#DC2626', width=3.5, style='solid')
if opened_sw:
    nx.draw_networkx_edges(G_full, draw_pos, ax=ax, edgelist=opened_sw,
                           edge_color='#F59E0B', width=2.5, style='dashed', alpha=0.9)
if active_edges:
    # Green marks all energised (island) connections
    nx.draw_networkx_edges(G_full, draw_pos, ax=ax, edgelist=active_edges,
                           edge_color='#22C55E', width=2.2, alpha=0.9)

# ── Dead nodes ────────────────────────────────────────────────────────────────
if dead_nodes:
    nx.draw_networkx_nodes(G_full, draw_pos, ax=ax, nodelist=dead_nodes,
                           node_size=60, node_color='#E5E7EB',
                           edgecolors='#9CA3AF', linewidths=0.5)

# ── Alive nodes: tier colour + tier size (same palette as base_case.py) ───────
for tname, tnodes in tier_buckets.items():
    if not tnodes:
        continue
    nx.draw_networkx_nodes(G_full, draw_pos, ax=ax, nodelist=tnodes,
                           node_color=C_NODE[tname], node_size=NODE_SIZES[tname],
                           edgecolors=EC_NODE[tname], linewidths=1.0)

# ── Substation (dead dark square) ─────────────────────────────────────────────
if SUB_BUS in draw_pos:
    nx.draw_networkx_nodes(G_full, draw_pos, ax=ax, nodelist=[SUB_BUS],
                           node_size=320, node_shape='s', node_color='#1F2937')

# ── DG battery symbols ────────────────────────────────────────────────────────
# Tweak these numbers to move specific batteries around!
# (X_offset, Y_offset). For example, a negative X moves it left.
dg_custom_offsets = {
    '701': (bat_w * -1, 0),          # Push right
    '722': (0, bat_h * 2.0),          # Push up
    '713': (-bat_w * 1.5, bat_h),     # Push left and up
    '738': (-bat_w * 1.5, bat_h),
    '737': (-bat_w * 1.5, bat_h),
    '711': (-0, bat_h* -1.5),
    # Add other DG nodes here as needed...
}

for dg in purchased_dgs:
    if dg not in draw_pos:
        continue
    alive = (pulp.value(s_N.get((dg, s), 0)) or 0) > 0.5
    p_out = sum((pulp.value(P_G[dg, p, s]) or 0) for p in phases)
    cap   = float(dg_sizes.get(dg, dg_sizes.get(str(dg), 0)))
    x, y  = draw_pos[dg]
    
    # Grab the custom offset, or default to putting it on the right
    offset = dg_custom_offsets.get(dg, (bat_w * 1.5, 0))

    draw_dg_battery(ax, x, y, alive, p_out, cap, offset)

# ── Bus labels: name + restored kW for live nodes only ───────────────────────
for node, (x, y) in draw_pos.items():
    if node == SUB_BUS:
        ax.annotate('SUB\n(dead)', xy=(x, y),
                    xytext=(14, -10), textcoords='offset points',
                    fontsize=9, ha='left', va='top', color='#1F2937',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='#CBD5E1',
                              boxstyle='round,pad=0.2', lw=0.5))
        continue

    if node in ['799', '799r']:
        continue

    alive    = (pulp.value(s_N.get((node, s), 0)) or 0) > 0.5
    restored = bus_restored_kw.get(node, 0.0)

    label = node.upper()
    if alive and restored > 0.1:
        label += f"\n+{restored:.0f} kW"

    offset  = custom_offsets.get(node, (0, 13))
    align_h = 'center'
    align_v = 'bottom'
    if offset[0] < -10: align_h = 'right'
    elif offset[0] > 10: align_h = 'left'
    if offset[1] < -10: align_v = 'top'

    text_color = '#1E293B' if alive else '#9CA3AF'

    ax.annotate(label,
                xy=(x, y),
                xytext=offset,
                textcoords='offset points',
                fontsize=9, ha=align_h, va=align_v, color=text_color,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='#CBD5E1',
                          boxstyle='round,pad=0.2', lw=0.5))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    f"Self-Healing Microgrid  |  {SCENARIO_KEY}\n"
    f"Restored: {total_restored_kw:.0f} kW ({pct_restored:.1f}%)  "
    f"|  Shed: {total_shed_kw:.0f} kW  "
    f"|  OpenDSS loss: {total_losses_kw:.1f} kW",
    fontsize=13, fontweight='bold', color='#0F172A', pad=12,
    loc='left', x=0.01
)
ax.axis('off')

# ── Legend ────────────────────────────────────────────────────────────────────
node_patches = [
    mpatches.Patch(facecolor='#1F2937',         edgecolor='#374151', linewidth=0.8,
                   label='Substation (dead)'),
    mpatches.Patch(facecolor='#E5E7EB',         edgecolor='#9CA3AF', linewidth=0.8,
                   label='Dead bus'),
    mpatches.Patch(facecolor=C_NODE['no_load'], edgecolor=EC_NODE['no_load'],
                   linewidth=0.8, label='No load'),
    mpatches.Patch(facecolor=C_NODE['tier1'],   edgecolor=EC_NODE['tier1'],
                   linewidth=0.8, label='Tier 1  (<50 kW)'),
    mpatches.Patch(facecolor=C_NODE['tier2'],   edgecolor=EC_NODE['tier2'],
                   linewidth=0.8, label='Tier 2  (50–85 kW)'),
    mpatches.Patch(facecolor=C_NODE['tier3'],   edgecolor=EC_NODE['tier3'],
                   linewidth=0.8, label='Tier 3  (>85 kW)'),
    mpatches.Patch(facecolor='#FFD700',         edgecolor='#1a5c2e', linewidth=1.0,
                   label='DG (battery symbol)'),
]

edge_proxies = [
    mlines.Line2D([], [], color='#22C55E', linewidth=2.2,
                  label='Energized line (island)'),
    mlines.Line2D([], [], color='#DC2626', linewidth=3.0,
                  label='Physical fault'),
    mlines.Line2D([], [], color='#F59E0B', linewidth=2.0, linestyle='dashed',
                  label='Switch opened (MILP)'),
    mlines.Line2D([], [], color='#D1D5DB', linewidth=1.5, linestyle='dotted',
                  label='De-energized line'),
]

ax.legend(
    handles=node_patches + edge_proxies,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.10),
    ncol=4,
    frameon=True,
    framealpha=0.96,
    edgecolor='#CBD5E1',
    facecolor='#FFFFFF',
    prop={'size': 10},
    title='Legend',
    title_fontproperties={'size': 10},
    borderpad=0.9,
    handlelength=1.8,
    columnspacing=1.1,
    handletextpad=0.6,
)

# ── Axis limits with symmetric padding ───────────────────────────────────────
x_pad = (max(all_x_vals) - min(all_x_vals)) * 0.08
y_pad = (max(all_y_vals) - min(all_y_vals)) * 0.08
ax.set_xlim(min(all_x_vals) - x_pad, max(all_x_vals) + x_pad)
ax.set_ylim(min(all_y_vals) - y_pad, max(all_y_vals) + y_pad)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()
