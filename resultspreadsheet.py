import os
import sys
import json
import math
import pandas as pd
import numpy as np
import scipy.stats as st

# Set working directory to script location
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

NUM_ITERATIONS = 30

# -----------------------------------------------------------------------
# RUN LABEL — change this to switch between runs, OR pass via sys.argv[1]:
#   ""          → original lambda 0.75 files (no label prefix)
#   "pga030N15" → convergence study N=15 files
#   "pga030N30" → convergence study N=30 files
#   "pga030N60" → convergence study N=60 files
# -----------------------------------------------------------------------
RUN_LABEL = "pga030N30L000Y15R010_rho60_sig50"   # hardcoded default (overridden by sys.argv below)
if len(sys.argv) > 1:
    RUN_LABEL = sys.argv[1]   # orchestrator passes LABEL here

compiled_data = []
# Keyed by load name → dict of {iter_i: rate, meta: {...}}
restoration_by_load = {}
# Flat list of per-(iteration, scenario, DG) OpenDSS dispatch records (P/Q/S).
dg_dispatch_records = []
# Flat lists of per-scenario voltage values across ALL iterations — used to
# compute the true global min/max (not averages of per-iteration averages).
_all_min_v   = []   # dss_min_v  for every valid scenario in every iteration
_all_max_v   = []   # dss_max_v  for every valid scenario in every iteration
_all_avg_v   = []   # dss_avg_v  for every valid scenario in every iteration
_all_served  = []   # dss_served_kw
_all_branch  = []   # max_branch_loading_pct
_all_nema    = []   # nema_vub_max_pct
_all_s_load  = []   # dss_s_load_kva

print("Compiling Advanced SAA Simulation Results...")

for i in range(1, NUM_ITERATIONS + 1):
    if RUN_LABEL:
        # Convergence study / labeled runs: clean names without "iter_" prefix
        master_file = f"master_dg_placements_{RUN_LABEL}_{i}.json"
        eval_file   = f"evaluation_results_500_{RUN_LABEL}_{i}.json"
    else:
        # Original unlabeled runs (standalone backward-compatibility)
        master_file = f"master_dg_placements_iter_{i}.json"
        eval_file   = f"evaluation_results_500_iter_{i}.json"

    if not os.path.exists(master_file) or not os.path.exists(eval_file):
        continue

    # --- Load Data ---
    with open(master_file, "r") as f:
        master_data = json.load(f)
    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    # --- Extract Metadata ---
    metadata = master_data.get("metadata", {})
    solve_time_s = metadata.get("solve_time_seconds", None)

    # --- Extract Financials ---
    financials = master_data.get("financials", {})
    c_inv = financials.get("total_investment_cost_CINV", 0)
    lifetime_om = financials.get("lifetime_om_maintenance", 0) or financials.get("lifetime_om", 0) or 0
    expected_disaster_lb = financials.get("expected_disaster_cost_raw", 0) or financials.get("expected_disaster_cost", 0) or 0
    objective_value = financials.get("objective_value_lifetime", 0)
    best_bound = financials.get("best_bound_theoretical", None)
    mip_gap_achieved_val = financials.get("mip_gap_achieved", None)

    # --- Extract Placements ---
    purchased_dgs = master_data.get("dg_placement", {}).get("purchased_buses", [])
    dg_sizes = (master_data.get("dg_placement", {}).get("sizes_kw")
                or master_data.get("dg_placement", {}).get("sizes_kva", {}))
    total_dg_capacity = sum(dg_sizes.values())

    dg_list = purchased_dgs[:3] + [None] * (3 - len(purchased_dgs[:3]))
    dg1_bus, dg2_bus, dg3_bus = dg_list[0], dg_list[1], dg_list[2]

    dg1_size = dg_sizes.get(str(dg1_bus), 0) if dg1_bus else 0
    dg2_size = dg_sizes.get(str(dg2_bus), 0) if dg2_bus else 0
    dg3_size = dg_sizes.get(str(dg3_bus), 0) if dg3_bus else 0

    # --- Extract 500-Scenario Testing Data ---
    saa_calcs = eval_data.get("saa_calculations", {})
    avg_storm_cost_ub = saa_calcs.get("average_storm_cost", 0)
    expected_lifetime_storm_cost = saa_calcs.get("expected_lifetime_storm_cost", avg_storm_cost_ub)

    # NPV-equivalent CAPEX: read from eval JSON first (saved by test_initial.py),
    # fall back to master JSON financials, last resort use raw c_inv.
    # Prefer the present-value convention (matches main_initial.py / paper Eq);
    # fall back to the older CRF×Y "capex_npv_equivalent" key for legacy files.
    capex_npv_equiv = (saa_calcs.get("capex_present_value")
                       or financials.get("capex_present_value")
                       or saa_calcs.get("capex_npv_equivalent", 0)
                       or financials.get("capex_npv_equivalent", 0)
                       or c_inv)

    scenario_details = eval_data.get("scenario_details", {})
    total_scenarios = len(scenario_details)

    # A scenario is valid only if OpenDSS converged (dss_loss_kw is not None).
    # Infeasible/failed scenarios are excluded from all averages so the
    # denominator reflects only the scenarios that actually produced a solution.
    valid_scenarios = [sd for sd in scenario_details.values() if sd.get("dss_loss_kw") is not None]
    valid_tests = len(valid_scenarios)

    avg_shed = 0
    avg_min_v = 0
    avg_losses = 0
    avg_ens = 0      # conditional EENS (kWh) = mean ENS over the scenario library
    worst_ens = 0    # worst-case ENS (kWh) across scenarios

    avg_avg_v = 0; avg_max_v = 0; avg_served = 0; avg_max_branch = 0
    avg_nema_vub = 0; max_nema_vub = 0; avg_s_load = 0
    if valid_tests > 0:
        avg_shed   = sum(sd.get("shed_kw", 0)    for sd in valid_scenarios) / valid_tests
        avg_min_v  = sum(sd.get("dss_min_v", 0)  for sd in valid_scenarios) / valid_tests
        avg_losses = sum(sd.get("dss_loss_kw", 0) for sd in valid_scenarios) / valid_tests
        _ens_vals  = [sd.get("ens_kwh", 0) or 0 for sd in valid_scenarios]
        avg_ens    = sum(_ens_vals) / valid_tests
        worst_ens  = max(_ens_vals) if _ens_vals else 0
        # New fields — tolerate older eval JSONs that predate the re-evaluation.
        _avg_v_vals    = [sd["dss_avg_v"]              for sd in valid_scenarios if sd.get("dss_avg_v")              is not None]
        _max_v_vals    = [sd["dss_max_v"]              for sd in valid_scenarios if sd.get("dss_max_v")              is not None]
        _served_vals   = [sd["dss_served_kw"]          for sd in valid_scenarios if sd.get("dss_served_kw")          is not None]
        _branch_vals   = [sd["max_branch_loading_pct"] for sd in valid_scenarios if sd.get("max_branch_loading_pct") is not None]
        _nema_vals     = [sd["nema_vub_max_pct"]       for sd in valid_scenarios if sd.get("nema_vub_max_pct")       is not None]
        _s_load_vals   = [sd["dss_s_load_kva"]         for sd in valid_scenarios if sd.get("dss_s_load_kva")         is not None]
        avg_avg_v      = sum(_avg_v_vals)  / len(_avg_v_vals)  if _avg_v_vals  else 0
        avg_max_v      = sum(_max_v_vals)  / len(_max_v_vals)  if _max_v_vals  else 0
        avg_served     = sum(_served_vals) / len(_served_vals) if _served_vals else 0
        avg_max_branch = sum(_branch_vals) / len(_branch_vals) if _branch_vals else 0
        avg_nema_vub   = sum(_nema_vals)   / len(_nema_vals)   if _nema_vals   else 0
        max_nema_vub   = max(_nema_vals)                        if _nema_vals   else 0
        avg_s_load     = sum(_s_load_vals) / len(_s_load_vals) if _s_load_vals else 0
        # Accumulate into global flat lists.
        _all_min_v.extend( sd.get("dss_min_v", None)              for sd in valid_scenarios if sd.get("dss_min_v")              is not None)
        _all_max_v.extend( sd.get("dss_max_v", None)              for sd in valid_scenarios if sd.get("dss_max_v")              is not None)
        _all_avg_v.extend( sd.get("dss_avg_v", None)              for sd in valid_scenarios if sd.get("dss_avg_v")              is not None)
        _all_served.extend(sd.get("dss_served_kw", None)          for sd in valid_scenarios if sd.get("dss_served_kw")          is not None)
        _all_branch.extend(sd.get("max_branch_loading_pct", None) for sd in valid_scenarios if sd.get("max_branch_loading_pct") is not None)
        _all_nema.extend(  sd.get("nema_vub_max_pct", None)       for sd in valid_scenarios if sd.get("nema_vub_max_pct")       is not None)
        _all_s_load.extend(sd.get("dss_s_load_kva", None)         for sd in valid_scenarios if sd.get("dss_s_load_kva")         is not None)

    # --- Per-iteration Bounds (risk-averse, CVaR-consistent) ----------------
    # Both bounds share ONE capital+O&M base (capex_npv_equiv + om_lifetime from
    # the eval JSON, which computes them uniformly), so capex/O&M cancels in the
    # gap and the bounds are directly comparable regardless of the master JSON's
    # capex convention.
    constants  = master_data.get("constants", {})
    lam        = float(constants.get("LAMBDA_RISK", 0.0) or 0.0)
    alpha_cvar = float(constants.get("ALPHA", 0.90) or 0.90)
    om_lifetime_base = (saa_calcs.get("om_present_value")
                        or financials.get("om_present_value")
                        or saa_calcs.get("om_lifetime")
                        or lifetime_om or 0)
    base_cost  = capex_npv_equiv + om_lifetime_base

    # LOWER BOUND: in-sample weighted disaster (1-λ)E_N + λ·CVaR_N is recovered
    # from the MILP objective by removing whatever capex/O&M it bundled in.
    # The `or` chain handles both key conventions: old
    # (capex_npv_equivalent / lifetime_om_maintenance) and new
    # (capex_present_value / om_present_value). Previously the LB used only the
    # raw E[Q] (expected_disaster_cost_raw), dropping the CVaR penalty for λ>0.
    capex_obj = (financials.get("capex_present_value")
                 or financials.get("capex_npv_equivalent") or 0)
    om_obj    = (financials.get("om_present_value")
                 or financials.get("lifetime_om_maintenance") or 0)
    weighted_disaster_lb = objective_value - capex_obj - om_obj
    lower_bound = base_cost + weighted_disaster_lb

    # UPPER BOUND: out-of-sample weighted disaster (1-λ)·mean₅₀₀ + λ·CVaRα,₅₀₀
    # over the saved per-scenario costs (Rockafellar–Uryasev empirical CVaR),
    # then scaled by the same lifetime multiplier the eval JSON used.
    costs = [sd.get("penalty_cost") for sd in scenario_details.values()
             if sd.get("penalty_cost") is not None]
    storm_mult = (expected_lifetime_storm_cost / avg_storm_cost_ub) if avg_storm_cost_ub else 1.0
    if costs:
        mean_c = sum(costs) / len(costs)
        var_q  = float(np.quantile(costs, alpha_cvar))
        cvar   = var_q + (1.0 / (1.0 - alpha_cvar)) * (sum(max(c - var_q, 0.0) for c in costs) / len(costs))
        storm_term = (1.0 - lam) * mean_c + lam * cvar
    else:
        storm_term = avg_storm_cost_ub
    upper_bound = base_cost + storm_mult * storm_term

    # Simple per-iteration gap (raw difference before statistical aggregation)
    gap_dollars = upper_bound - lower_bound
    gap_percentage = (gap_dollars / upper_bound) * 100 if upper_bound > 0 else 0

    # --- Harvest load restoration frequency for this iteration ---
    for entry in eval_data.get("load_restoration_frequency", []):
        lname = entry["load"]
        if lname not in restoration_by_load:
            restoration_by_load[lname] = {
                "Bus":        entry.get("bus", ""),
                "Total kW":   entry.get("total_kW", 0),
                "Tier":       entry.get("tier_label", ""),
                "VoLL ($/kWh)": entry.get("voll_tier", 0),
            }
        restoration_by_load[lname][f"Iter {i} (%)"] = entry.get("restoration_rate_pct", None)

    # --- Harvest per-DG OpenDSS dispatch (P/Q) and count reactive exceedances ---
    # Two bases: single-DG (documents the slack-aggregation artifact) and
    # island-aggregate (physically meaningful). Tolerant of older eval JSONs
    # that predate the dg_dispatch / island fields.
    iter_q_exceed = 0          # single-DG basis
    iter_q_exceed_island = 0   # island-aggregate basis
    for s_name, sd in scenario_details.items():
        for g, d in (sd.get("dg_dispatch") or {}).items():
            within = d.get("within_limit")
            within_island = d.get("within_island_limit")
            dg_dispatch_records.append({
                "Iteration":      i,
                "Scenario":       s_name,
                "DG Bus":         g,
                "P (kW)":         d.get("p_delivered_kw"),
                "Q (kvar)":       d.get("q_delivered_kvar"),
                "S Delivered (kVA)": d.get("s_delivered_kva"),
                "S Cap (kVA)":       d.get("s_cap_kva"),
                "S Loading (%)":     d.get("s_loading_pct"),
                "Q Limit (kvar)": d.get("q_limit_kvar"),
                "Within Limit":   within,
                "Island Q Limit (kvar)": d.get("island_q_limit_kvar"),
                "Within Island Limit":   within_island,
            })
            if within is False:
                iter_q_exceed += 1
            if within_island is False:
                iter_q_exceed_island += 1

    compiled_data.append({
        "Iteration": i,
        "Q Exceed single-DG (#)": iter_q_exceed,
        "Q Exceed island (#)": iter_q_exceed_island,
        "DG 1 Bus": dg1_bus, "DG 1 Size (kVA)": dg1_size,
        "DG 2 Bus": dg2_bus, "DG 2 Size (kVA)": dg2_size,
        "DG 3 Bus": dg3_bus, "DG 3 Size (kVA)": dg3_size,
        "Total DG Capacity (kVA)": total_dg_capacity,
        "Investment Cost ($)": c_inv,
        "O&M Cost ($)": lifetime_om,
        "Expected Disaster Cost - SAA ($)": expected_disaster_lb,
        "Risk-Averse Objective (CVaR $)": objective_value,
        "Lower Bound ($)": lower_bound,
        "Upper Bound ($)": upper_bound,
        "SAA Gap ($)": gap_dollars,
        "SAA Gap (%)": gap_percentage,
        # CI columns are aggregate statistics — populated only in the summary row
        "LB 95% CI Lower ($)": None,
        "LB 95% CI Upper ($)": None,
        "UB 95% CI Lower ($)": None,
        "UB 95% CI Upper ($)": None,
        "SAA Opt. Gap ($) [Eq.38]": None,
        "SAA Opt. Gap (%) [Eq.38]": None,
        "Scenarios Total": total_scenarios,
        "Scenarios Valid (OpenDSS OK)": valid_tests,
        "Scenarios Infeasible": total_scenarios - valid_tests,
        "Avg Storm Cost - Test ($)": avg_storm_cost_ub,
        "Avg Load Shedding (kW)": avg_shed,
        "Conditional EENS (kWh)": avg_ens,
        "Worst-Case ENS (kWh)": worst_ens,
        "Avg OpenDSS Min Volt (pu)": avg_min_v,
        "Avg OpenDSS Avg Volt (pu)": avg_avg_v,
        "Avg OpenDSS Max Volt (pu)": avg_max_v,
        "Avg OpenDSS Served (kW)": avg_served,
        "Avg Max Branch Loading (%)": avg_max_branch,
        "Avg NEMA VUB Max (%)": avg_nema_vub,
        "Max NEMA VUB Max (%)": max_nema_vub,
        "Avg OpenDSS S Load (kVA)": avg_s_load,
        "Avg OpenDSS Losses (kW)": avg_losses,
        "Best Bound - MILP ($)": best_bound,
        "MIP Gap Achieved (%)": round(mip_gap_achieved_val * 100, 2) if mip_gap_achieved_val is not None else None,
        "Solve Time (s)": solve_time_s,
        # Summary-only columns — None in per-iteration rows, populated in summary row
        "Global Min Volt (pu)": None,
        "Global Max Volt (pu)": None,
        "Global Max Branch Loading (%)": None,
        "Global Max NEMA VUB (%)": None,
        "Global Avg NEMA VUB (%)": None,
        "Global Avg S Load (kVA)": None,
    })

# --- Global voltage / loading statistics across ALL scenarios × iterations ---
global_min_v      = min(_all_min_v)    if _all_min_v   else None
global_max_v      = max(_all_max_v)    if _all_max_v   else None
global_avg_v      = (sum(_all_avg_v)   / len(_all_avg_v))   if _all_avg_v   else None
global_avg_served = (sum(_all_served)  / len(_all_served))  if _all_served  else None
global_max_branch = max(_all_branch)   if _all_branch  else None
global_avg_branch = (sum(_all_branch)  / len(_all_branch))  if _all_branch  else None
global_max_nema   = max(_all_nema)     if _all_nema    else None
global_avg_nema   = (sum(_all_nema)    / len(_all_nema))    if _all_nema    else None
global_avg_s_load = (sum(_all_s_load)  / len(_all_s_load))  if _all_s_load  else None

print(f"\nGlobal voltage range across ALL {len(_all_min_v)} valid scenario-iterations:")
print(f"  Absolute Min V (pu) : {global_min_v:.4f}" if global_min_v else "  Absolute Min V : N/A")
print(f"  Mean Avg V (pu)     : {global_avg_v:.4f}" if global_avg_v else "  Mean Avg V : N/A")
print(f"  Absolute Max V (pu) : {global_max_v:.4f}" if global_max_v else "  Absolute Max V : N/A")
print(f"  Max Branch Loading  : {global_max_branch:.1f}%" if global_max_branch else "  Max Branch Loading : N/A")

# --- Build DataFrame ---
if not compiled_data:
    print("\nERROR: No iterations loaded. Expected filenames:")
    if RUN_LABEL:
        print(f"  master_dg_placements_iter_{RUN_LABEL}_1.json ... _{RUN_LABEL}_30.json")
        print(f"  evaluation_results_500_{RUN_LABEL}_1.json   ... _{RUN_LABEL}_30.json")
    else:
        print("  master_dg_placements_iter_1.json ... iter_30.json")
        print("  evaluation_results_500_iter_1.json ... iter_30.json")
    raise SystemExit("Fix RUN_LABEL or check that JSON files are in the same folder as this script.")

df = pd.DataFrame(compiled_data)

# -----------------------------------------------------------------------
# SAA Statistical Calculations (Section III of paper, Eqs. 31-38)
# -----------------------------------------------------------------------
alpha = 0.05
M = len(df)

# t-value: kappa = t_{alpha/2, M-1}  (Eq. 33 / 37)
kappa = st.t.ppf(1 - alpha / 2, M - 1)

# Aggregate means (L and U in the paper)
avg_lb = df["Lower Bound ($)"].mean()   # L  (Eq. 31)
avg_ub = df["Upper Bound ($)"].mean()   # U  (Eq. 35)

# Standard deviations (s_L and s_U)  (Eq. 32 / 36)
std_lb = df["Lower Bound ($)"].std()    # s_L
std_ub = df["Upper Bound ($)"].std()    # s_U

sqrt_M = math.sqrt(M)

# Confidence intervals for the lower bound (Eq. 33)
lb_ci_lower = avg_lb - kappa * std_lb / sqrt_M
lb_ci_upper = avg_lb + kappa * std_lb / sqrt_M

# Confidence intervals for the upper bound (Eq. 37)
ub_ci_lower = avg_ub - kappa * std_ub / sqrt_M
ub_ci_upper = avg_ub + kappa * std_ub / sqrt_M

# SAA optimality gap (Eq. 38)
# gap = (U + kappa*s_U/sqrt(M)) - (L - kappa*s_L/sqrt(M))
#      = ub_ci_upper - lb_ci_lower
saa_opt_gap_dollars = ub_ci_upper - lb_ci_lower
saa_opt_gap_pct = (saa_opt_gap_dollars / ub_ci_upper) * 100 if ub_ci_upper > 0 else 0

print(f"\n95% Confidence Interval Results (M={M}, kappa={kappa:.4f}):")
print(f"  Lower Bound CI : [${lb_ci_lower:,.2f}  ,  ${lb_ci_upper:,.2f}]")
print(f"  Upper Bound CI : [${ub_ci_lower:,.2f}  ,  ${ub_ci_upper:,.2f}]")
print(f"  SAA Opt. Gap   : ${saa_opt_gap_dollars:,.2f}  ({saa_opt_gap_pct:.2f}%)")

# Solve-time aggregates — defined here so the summary print below can use them;
# also reused in the stats/std reference rows further down.
_solve_times = pd.to_numeric(df["Solve Time (s)"], errors="coerce")
total_solve_time_s = _solve_times.sum()
avg_solve_time_s   = _solve_times.mean()

print(f"\nSolve Time Summary:")
print(f"  Total (all iterations) : {total_solve_time_s/3600:.2f} h  ({total_solve_time_s:.0f} s)")
print(f"  Average per iteration  : {avg_solve_time_s:.0f} s")

# -----------------------------------------------------------------------
# Summary row
# -----------------------------------------------------------------------
summary = pd.DataFrame([{
    "Iteration": "AVERAGE / SAA STATS",
    "DG 1 Bus": "-", "DG 1 Size (kVA)": df["DG 1 Size (kVA)"].mean(),
    "DG 2 Bus": "-", "DG 2 Size (kVA)": df["DG 2 Size (kVA)"].mean(),
    "DG 3 Bus": "-", "DG 3 Size (kVA)": df["DG 3 Size (kVA)"].mean(),
    "Total DG Capacity (kVA)": df["Total DG Capacity (kVA)"].mean(),
    "Investment Cost ($)": df["Investment Cost ($)"].mean(),
    "O&M Cost ($)": df["O&M Cost ($)"].mean(),
    "Expected Disaster Cost - SAA ($)": df["Expected Disaster Cost - SAA ($)"].mean(),
    "Risk-Averse Objective (CVaR $)": df["Risk-Averse Objective (CVaR $)"].mean(),
    "Lower Bound ($)": avg_lb,
    "Upper Bound ($)": avg_ub,
    # Simple average gap (point estimate: U - L)
    "SAA Gap ($)": avg_ub - avg_lb,
    "SAA Gap (%)": ((avg_ub - avg_lb) / avg_ub) * 100 if avg_ub > 0 else 0,
    # Confidence interval bounds for LB and UB (Eqs. 33 & 37)
    "LB 95% CI Lower ($)": lb_ci_lower,
    "LB 95% CI Upper ($)": lb_ci_upper,
    "UB 95% CI Lower ($)": ub_ci_lower,
    "UB 95% CI Upper ($)": ub_ci_upper,
    # SAA optimality gap with kappa, stdev, sqrt(M) (Eq. 38)
    "SAA Opt. Gap ($) [Eq.38]": saa_opt_gap_dollars,
    "SAA Opt. Gap (%) [Eq.38]": saa_opt_gap_pct,
    "Scenarios Total": df["Scenarios Total"].sum(),
    "Scenarios Valid (OpenDSS OK)": df["Scenarios Valid (OpenDSS OK)"].sum(),
    "Scenarios Infeasible": df["Scenarios Infeasible"].sum(),
    "Avg Storm Cost - Test ($)": df["Avg Storm Cost - Test ($)"].mean(),
    "Avg Load Shedding (kW)": df["Avg Load Shedding (kW)"].mean(),
    "Conditional EENS (kWh)": df["Conditional EENS (kWh)"].mean(),
    "Worst-Case ENS (kWh)": df["Worst-Case ENS (kWh)"].max(),
    "Avg OpenDSS Min Volt (pu)": df["Avg OpenDSS Min Volt (pu)"].mean(),
    "Avg OpenDSS Avg Volt (pu)": df["Avg OpenDSS Avg Volt (pu)"].mean(),
    "Avg OpenDSS Max Volt (pu)": df["Avg OpenDSS Max Volt (pu)"].mean(),
    "Avg OpenDSS Served (kW)": df["Avg OpenDSS Served (kW)"].mean(),
    "Avg Max Branch Loading (%)": df["Avg Max Branch Loading (%)"].mean(),
    "Avg NEMA VUB Max (%)": df["Avg NEMA VUB Max (%)"].mean(),
    "Max NEMA VUB Max (%)": df["Max NEMA VUB Max (%)"].max(),
    "Avg OpenDSS S Load (kVA)": df["Avg OpenDSS S Load (kVA)"].mean(),
    "Global Min Volt (pu)": global_min_v,
    "Global Max Volt (pu)": global_max_v,
    "Global Max Branch Loading (%)": global_max_branch,
    "Global Max NEMA VUB (%)": global_max_nema,
    "Global Avg NEMA VUB (%)": global_avg_nema,
    "Global Avg S Load (kVA)": global_avg_s_load,
    "Avg OpenDSS Losses (kW)": df["Avg OpenDSS Losses (kW)"].mean(),
    "Best Bound - MILP ($)": pd.to_numeric(df["Best Bound - MILP ($)"], errors="coerce").mean(),
    "MIP Gap Achieved (%)": pd.to_numeric(df["MIP Gap Achieved (%)"], errors="coerce").mean(),
    "Solve Time (s)": pd.to_numeric(df["Solve Time (s)"], errors="coerce").mean(),
}])

# -----------------------------------------------------------------------
# Stats reference row (shows the components of the gap formula)
# -----------------------------------------------------------------------
# (_solve_times / total_solve_time_s / avg_solve_time_s computed above.)

stats_ref = pd.DataFrame([{
    "Iteration": "STATS COMPONENTS",
    "DG 1 Bus": f"kappa={kappa:.4f}",
    "DG 1 Size (kVA)": None,
    "DG 2 Bus": f"sqrt(M)={sqrt_M:.4f}",
    "DG 2 Size (kVA)": None,
    "DG 3 Bus": f"Total solve: {total_solve_time_s/3600:.2f} h",
    "DG 3 Size (kVA)": None,
    "Total DG Capacity (kVA)": None,
    "Investment Cost ($)": None,
    "O&M Cost ($)": None,
    "Expected Disaster Cost - SAA ($)": None,
    "Risk-Averse Objective (CVaR $)": None,
    "Lower Bound ($)": avg_lb,
    "Upper Bound ($)": avg_ub,
    "SAA Gap ($)": None,
    "SAA Gap (%)": None,
    "LB 95% CI Lower ($)": lb_ci_lower,
    "LB 95% CI Upper ($)": lb_ci_upper,
    "UB 95% CI Lower ($)": ub_ci_lower,
    "UB 95% CI Upper ($)": ub_ci_upper,
    "SAA Opt. Gap ($) [Eq.38]": saa_opt_gap_dollars,
    "SAA Opt. Gap (%) [Eq.38]": saa_opt_gap_pct,
    "Scenarios Total": None,
    "Scenarios Valid (OpenDSS OK)": None,
    "Scenarios Infeasible": None,
    "Avg Storm Cost - Test ($)": None,
    "Avg Load Shedding (kW)": None,
    "Avg OpenDSS Min Volt (pu)": None,
    "Avg OpenDSS Avg Volt (pu)": None,
    "Avg OpenDSS Max Volt (pu)": None,
    "Avg OpenDSS Served (kW)": None,
    "Avg Max Branch Loading (%)": None,
    "Avg NEMA VUB Max (%)": None,
    "Max NEMA VUB Max (%)": None,
    "Avg OpenDSS S Load (kVA)": None,
    "Global Min Volt (pu)": None,
    "Global Max Volt (pu)": None,
    "Global Max Branch Loading (%)": None,
    "Global Max NEMA VUB (%)": None,
    "Global Avg NEMA VUB (%)": None,
    "Global Avg S Load (kVA)": None,
    "Avg OpenDSS Losses (kW)": None,
    "Best Bound - MILP ($)": None,
    "MIP Gap Achieved (%)": None,
    "Solve Time (s)": total_solve_time_s,
}])

# Also add a row showing std deviations explicitly
std_ref = pd.DataFrame([{
    "Iteration": "STD DEVIATIONS",
    "DG 1 Bus": f"s_L=${std_lb:,.2f}",
    "DG 1 Size (kVA)": std_lb,
    "DG 2 Bus": f"s_U=${std_ub:,.2f}",
    "DG 2 Size (kVA)": std_ub,
    "DG 3 Bus": None,
    "DG 3 Size (kVA)": None,
    "Total DG Capacity (kVA)": None,
    "Investment Cost ($)": None,
    "O&M Cost ($)": None,
    "Expected Disaster Cost - SAA ($)": None,
    "Risk-Averse Objective (CVaR $)": None,
    "Lower Bound ($)": std_lb,
    "Upper Bound ($)": std_ub,
    "SAA Gap ($)": None,
    "SAA Gap (%)": None,
    "LB 95% CI Lower ($)": None,
    "LB 95% CI Upper ($)": None,
    "UB 95% CI Lower ($)": None,
    "UB 95% CI Upper ($)": None,
    "SAA Opt. Gap ($) [Eq.38]": None,
    "SAA Opt. Gap (%) [Eq.38]": None,
    "Scenarios Total": None,
    "Scenarios Valid (OpenDSS OK)": None,
    "Scenarios Infeasible": None,
    "Avg Storm Cost - Test ($)": None,
    "Avg Load Shedding (kW)": None,
    "Avg OpenDSS Min Volt (pu)": None,
    "Avg OpenDSS Avg Volt (pu)": None,
    "Avg OpenDSS Max Volt (pu)": None,
    "Avg OpenDSS Served (kW)": None,
    "Avg Max Branch Loading (%)": None,
    "Avg NEMA VUB Max (%)": None,
    "Max NEMA VUB Max (%)": None,
    "Avg OpenDSS S Load (kVA)": None,
    "Global Min Volt (pu)": None,
    "Global Max Volt (pu)": None,
    "Global Max Branch Loading (%)": None,
    "Global Max NEMA VUB (%)": None,
    "Global Avg NEMA VUB (%)": None,
    "Global Avg S Load (kVA)": None,
    "Avg OpenDSS Losses (kW)": None,
    "Best Bound - MILP ($)": None,
    "MIP Gap Achieved (%)": None,
    "Solve Time (s)": _solve_times.std(),
}])

# Cast all-None CI columns to object so pandas doesn't warn about dtype inference
_ci_cols = [
    "LB 95% CI Lower ($)", "LB 95% CI Upper ($)",
    "UB 95% CI Lower ($)", "UB 95% CI Upper ($)",
    "SAA Opt. Gap ($) [Eq.38]", "SAA Opt. Gap (%) [Eq.38]",
    "Best Bound - MILP ($)", "MIP Gap Achieved (%)", "Solve Time (s)",
    # Summary-only columns (None in per-iteration rows)
    "Global Min Volt (pu)", "Global Max Volt (pu)", "Global Max Branch Loading (%)",
    "Global Max NEMA VUB (%)", "Global Avg NEMA VUB (%)", "Global Avg S Load (kVA)",
]
for _col in _ci_cols:
    if _col not in df.columns:
        df[_col] = None
    df[_col] = df[_col].astype(object)

final_df = pd.concat([df, summary, stats_ref, std_ref], ignore_index=True)

# -----------------------------------------------------------------------
# Parameters sheet — read from the constants block of the first valid
# master JSON so the spreadsheet is self-contained and unambiguous.
# Supplemented with derived values (PWF, CRF, annual prob) for clarity.
# -----------------------------------------------------------------------
_params_source_file = None
_params_constants   = {}
for i in range(1, NUM_ITERATIONS + 1):
    if RUN_LABEL:
        _f = f"master_dg_placements_{RUN_LABEL}_{i}.json"
    else:
        _f = f"master_dg_placements_iter_{i}.json"
    if os.path.exists(_f):
        with open(_f) as _fh:
            _pd = json.load(_fh)
        _params_constants   = _pd.get("constants", {})
        _params_source_file = _f
        break

# Readable labels for each key
_PARAM_LABELS = {
    "COST_DG_FIXED":               "DG Fixed Cost ($/unit)",
    "COST_DG_PER_KW":              "DG Capital Cost ($/kW)",
    "COST_DG_OM_PER_KW_YR":        "DG O&M Cost ($/kW/yr)",
    "COST_GEN_PER_KWH":            "DG Fuel Cost during Disaster ($/kWh)",
    "HOURS_NORMAL":                "Normal Operation Hours (hr/yr)",
    "MAX_INVESTMENT_BUDGET":       "Maximum Investment Budget ($)",
    "PLANNING_YEARS":              "Planning Horizon (years)",
    "INTEREST_RATE":               "Interest / Discount Rate",
    "PWF":                         "Present Worth Factor (PWF)",
    "MIP_GAP":                     "Gurobi MIP Gap Tolerance",
    "LAMBDA_RISK":                 "Risk Aversion Weight λ (0=risk-neutral, 1=pure CVaR)",
    "ALPHA":                       "CVaR Confidence Level α",
    "VOLL_TIER_1":                 "Value of Lost Load — Residential ($/kWh)",
    "VOLL_TIER_2":                 "Value of Lost Load — Commercial/Industrial ($/kWh)",
    "VOLL_CRITICAL":               "Value of Lost Load — Critical Infrastructure ($/kWh)",
    "CRITICAL_BUSES":              "Critical Bus Overrides (bus IDs)",
    "CONDITIONAL_EVENT_OCCURRENCE": "Event Occurrence Multiplier (1=conditional, <1=annual freq.)",
}

_param_rows = []
for key, label in _PARAM_LABELS.items():
    val = _params_constants.get(key, "N/A")
    _param_rows.append({"Parameter": label, "Symbol / Key": key, "Value": val})

# Derived parameters not stored directly in JSON
if _params_constants:
    _r = _params_constants.get("INTEREST_RATE", 0)
    _y = _params_constants.get("PLANNING_YEARS", 0)
    _pwf = _params_constants.get("PWF", None)
    _crf = _r * (1 + _r)**_y / ((1 + _r)**_y - 1) if _r and _y else None
    _ann_prob = 1 - (1 - 1/475)**_y if _y else None
    _param_rows += [
        {"Parameter": "Capital Recovery Factor (CRF)", "Symbol / Key": "CRF", "Value": round(_crf, 6) if _crf else "N/A"},
        {"Parameter": "Annual Exceedance Probability (475-yr return)", "Symbol / Key": "p_annual", "Value": round(1/475, 6)},
        {"Parameter": "Cumulative Event Probability over Planning Horizon", "Symbol / Key": "p_lifetime", "Value": round(_ann_prob, 6) if _ann_prob else "N/A"},
        {"Parameter": "Run Label", "Symbol / Key": "RUN_LABEL", "Value": RUN_LABEL or "(none)"},
        {"Parameter": "Number of SAA Iterations (M)", "Symbol / Key": "NUM_ITERATIONS", "Value": NUM_ITERATIONS},
        {"Parameter": "Source master JSON", "Symbol / Key": "—", "Value": _params_source_file or "unknown"},
    ]

params_df = pd.DataFrame(_param_rows, columns=["Parameter", "Symbol / Key", "Value"])

# -----------------------------------------------------------------------
# Load Restoration Pivot Table
# -----------------------------------------------------------------------
# Rows = loads, columns = per-iteration rates + summary stats
# Sorted by mean restoration rate descending so the most reliably
# restored loads appear at the top.
restoration_rows = []
iter_cols = [f"Iter {k} (%)" for k in range(1, NUM_ITERATIONS + 1)]

for lname, info in restoration_by_load.items():
    rates = [info.get(c) for c in iter_cols]
    numeric_rates = [r for r in rates if r is not None]
    row = {
        "Load": lname,
        "Bus": info.get("Bus", ""),
        "Total kW": info.get("Total kW", 0),
        "Tier": info.get("Tier", ""),
        "VoLL ($/kWh)": info.get("VoLL ($/kWh)", 0),
    }
    for col in iter_cols:
        row[col] = info.get(col)
    row["Mean Restoration (%)"] = round(float(np.mean(numeric_rates)), 2) if numeric_rates else None
    row["Std Dev (%)"]           = round(float(np.std(numeric_rates, ddof=1)), 2) if len(numeric_rates) > 1 else 0.0
    row["Min (%)"]               = round(float(np.min(numeric_rates)), 2) if numeric_rates else None
    row["Max (%)"]               = round(float(np.max(numeric_rates)), 2) if numeric_rates else None
    restoration_rows.append(row)

restoration_df = pd.DataFrame(restoration_rows)
if not restoration_df.empty:
    restoration_df = restoration_df.sort_values("Mean Restoration (%)", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------
# DG Dispatch (OpenDSS P/Q) — per-DG aggregate across all iterations/scenarios
# plus the raw per-(iter, scenario, DG) records for plotting. Q is the actual
# reactive the slack Vsource supplied; |Q| is compared to the MILP cap so the
# (small) reactive exceedances are documented rather than hidden.
# -----------------------------------------------------------------------
dg_dispatch_df = pd.DataFrame(dg_dispatch_records)
dg_dispatch_summary = pd.DataFrame()
if not dg_dispatch_df.empty:
    dd = dg_dispatch_df.copy()
    dd["absQ"] = dd["Q (kvar)"].abs()
    # Apparent power S = sqrt(P^2 + Q^2): the kVA the slack actually delivered.
    # Max S per DG is the physical alternator rating needed (generators are
    # procured on kVA, not kW), which absorbs the small reactive surplus.
    dd["absS"] = (dd["P (kW)"] ** 2 + dd["Q (kvar)"] ** 2) ** 0.5
    grp = dd.groupby("DG Bus")
    dg_dispatch_summary = pd.DataFrame({
        "Scenarios (#)":     grp.size(),
        "Mean P (kW)":       grp["P (kW)"].mean().round(2),
        "Max P (kW)":        grp["P (kW)"].max().round(2),
        "Mean |Q| (kvar)":   grp["absQ"].mean().round(2),
        "Max |Q| (kvar)":    grp["absQ"].max().round(2),
        "Q Limit single-DG (kvar)": grp["Q Limit (kvar)"].max().round(2),
        "Island Q Limit (kvar)":    grp["Island Q Limit (kvar)"].max().round(2),
        "Mean S (kVA)":      grp["absS"].mean().round(2),
        "Max S (kVA)":       grp["absS"].max().round(2),
        # Single-DG count shows the aggregation artifact; island count is the
        # physically meaningful one (expected to be ~0).
        "Q Exceed single-DG (#)": grp["Within Limit"].apply(lambda x: int((x == False).sum())),
        "Q Exceed island (#)":    grp["Within Island Limit"].apply(lambda x: int((x == False).sum())),
        "Q Exceed island (%)":    grp["Within Island Limit"].apply(lambda x: round((x == False).mean() * 100, 2)),
    }).reset_index()

# -----------------------------------------------------------------------
# Write Excel
# -----------------------------------------------------------------------
label_suffix   = f"_{RUN_LABEL}" if RUN_LABEL else "_default"
excel_filename = f"SAA_Compiled_Results_Advanced{label_suffix}.xlsx"

with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    final_df.to_excel(writer, sheet_name="SAA Results", index=False)
    if not restoration_df.empty:
        restoration_df.to_excel(writer, sheet_name="Load Restoration", index=False)
    params_df.to_excel(writer, sheet_name="Parameters", index=False)
    if not dg_dispatch_summary.empty:
        dg_dispatch_summary.to_excel(writer, sheet_name="DG Dispatch", index=False)
        dg_dispatch_df.to_excel(writer, sheet_name="DG Dispatch (raw)", index=False)

print(f"\nSuccess! Check '{excel_filename}'.")
print(f"  Sheet 1 — SAA Results      : {len(final_df)} rows")
if not restoration_df.empty:
    print(f"  Sheet 2 — Load Restoration : {len(restoration_df)} loads × {NUM_ITERATIONS} iterations")
print(f"  Sheet 3 — Parameters       : {len(params_df)} entries (source: {_params_source_file})")
if not dg_dispatch_summary.empty:
    _tot_exc = int((dg_dispatch_df["Within Limit"] == False).sum())
    _tot_exc_island = int((dg_dispatch_df["Within Island Limit"] == False).sum())
    print(f"  Sheet 4 — DG Dispatch      : {len(dg_dispatch_summary)} DGs, "
          f"{len(dg_dispatch_df)} dispatch records | Q exceedances: "
          f"{_tot_exc} single-DG, {_tot_exc_island} island-aggregate")
