#!/usr/bin/env python3
"""
check_dg_violations.py
-----------------------
Scans the 6 labels x 30 iterations (180 master / 180 evaluation files = 360
total) produced by test_initial.py and flags DG dispatch problems.

Violation checks:
  P_LIMIT_MILP / P_LIMIT_ODS  -- P dispatch > 0.8 x S_cap  (tol: 0.5% of limit)
  Q_LIMIT_MILP / Q_LIMIT_ODS  -- Q dispatch > 0.6 x S_cap  (tol: 1.0% of limit)
  S_RATING_ODS                 -- OpenDSS S > S_cap          (zero tolerance)
  P_DIVERGENCE / Q_DIVERGENCE  -- MILP vs OpenDSS gap > --diff-abs AND > --diff-pct
  V_UNDER / V_OVER             -- dss_min_v < 0.95 pu / dss_max_v > 1.05 pu (zero
                                   tolerance -- mirrors the MILP's own exact
                                   U_min=0.95^2 / U_max=1.05^2 constraint)
  BRANCH_OVERLOAD               -- max_branch_loading_pct > 100%   (zero tolerance)
  TRAFO_OVERLOAD                 -- trafo_load_pct > 100%           (zero tolerance)
  NEMA_VUB_OVER                  -- nema_vub_max_pct > 2.0%         (zero tolerance)

Global + per-label extremes tracked (blackout scenarios excluded):
  - min pu voltage
  - max NEMA voltage unbalance % (nema_vub_max_pct / nema_vub_bus)
  - max branch loading %
  - max OpenDSS losses kW
  - worst single violation by exceedance %

Usage:
    python check_dg_violations.py [--dir DATA_DIR] [--iters N]
                                   [--p-frac 0.8] [--q-frac 0.6]
                                   [--diff-abs 5.0] [--diff-pct 10.0]
                                   [--json-out dg_violation_report.json]
"""

import argparse
import json
import os

# ==========================================================================
# CONFIG
# ==========================================================================
LABELS = [
    "pga023N30L000Y15R010_rho00_sig50",
    "pga023N30L075Y15R010_rho00_sig50",
    "pga030N30L000Y15R010_rho00_sig50",
    "pga030N30L075Y15R010_rho00_sig50",
    "pga040N30L000Y15R010_rho00_sig50",
    "pga040N30L075Y15R010_rho00_sig50",
]

P_TOL_FRAC = 0.005   # 0.5% of max_p
Q_TOL_FRAC = 0.010   # 1.0% of max_q
# S_RATING_ODS: zero tolerance

V_MIN_PU = 0.95   # hard cutoff -- mirrors the MILP's own exact U_min=0.95^2
V_MAX_PU = 1.05   # hard cutoff -- mirrors the MILP's own exact U_max=1.05^2
# Voltage: zero tolerance (see main_initial.py / test_initial.py U_min/U_max)

BRANCH_MAX_PCT   = 100.0   # hard cutoff -- line thermal rating (LINE_CAPACITY)
TRAFO_MAX_PCT    = 100.0   # hard cutoff -- transformer thermal rating (500 kVA)
NEMA_VUB_MAX_PCT = 2.0     # hard cutoff -- NEMA MG1 voltage-unbalance derating threshold
# Branch/trafo loading and NEMA VUB: zero tolerance

LOSSES_KEYS = [
    "total_losses_kw", "dss_losses_kw", "losses_kw",
    "total_loss_kw",   "dss_loss_kw",   "loss_kw",
]


# ==========================================================================
# HELPERS
# ==========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(description="Scan DG placement/evaluation JSONs for dispatch violations.")
    p.add_argument("--dir",      default=".",    help="Directory containing the JSON files (default: current dir)")
    p.add_argument("--iters",    type=int, default=30, help="Number of iterations per label (default: 30)")
    p.add_argument("--p-frac",   type=float, default=0.8,  help="max P as fraction of S_cap (default 0.8)")
    p.add_argument("--q-frac",   type=float, default=0.6,  help="max Q as fraction of S_cap (default 0.6)")
    p.add_argument("--diff-abs", type=float, default=5.0,  help="absolute kW/kvar gap for divergence check (default 5.0)")
    p.add_argument("--diff-pct", type=float, default=10.0, help="relative %% gap for divergence check (default 10.0)")
    p.add_argument("--json-out", default="dg_violation_report.json",
                   help="path to write the structured JSON report")
    return p


def candidate_master_filenames(data_dir, label, i):
    return [os.path.join(data_dir, b) for b in [
        f"master_dg_placements_{label}_{i}.json",
        f"master_dg_placements_{label}{i}.json",
    ]]


def candidate_eval_filenames(data_dir, label, i):
    return [os.path.join(data_dir, b) for b in [
        f"evaluation_results_500_{label}_{i}.json",
        f"evaluation_results_{label}_{i}.json",
        f"evaluation_results{label}_{i}.json",
        f"evaluation_results_500_{label}{i}.json",
    ]]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def safe_float(v, default=0.0):
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def load_dg_caps(master_path):
    caps = {}
    if not master_path:
        return caps
    try:
        with open(master_path) as f:
            master = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [warn] could not read {master_path}: {e}")
        return caps
    dgp = master.get("dg_placement", {})
    sizes = dgp.get("sizes_kw") or dgp.get("sizes_kva") or {}
    for k, v in sizes.items():
        caps[str(k)] = safe_float(v)
    return caps


def exceedance_pct(actual, limit):
    if limit <= 0:
        return 0.0
    return max(0.0, (abs(actual) - limit) / limit * 100.0)


def empty_extremes():
    """Return a fresh per-label extremes dict."""
    return {
        "total_scenarios":       0,
        "nonconverged_scenarios": 0,
        "violation_scenarios":  0,
        "violation_kind_tally": {},
        "worst_violation":      None,
        "min_pu_v":             None,
        "min_pu_v_loc":         None,
        "max_vunbal":           None,
        "max_vunbal_bus":       None,
        "max_vunbal_loc":       None,
        "max_branch_pct":       None,
        "max_branch_loc":       None,
        "max_losses_kw":        None,
        "max_losses_loc":       None,
        "max_trafo_load_pct":   None,
        "max_trafo_load_loc":   None,
        "max_trafo_loss_kw":    None,
        "max_trafo_loss_loc":   None,
        "max_line_loss_kw":     None,
        "max_line_loss_loc":    None,
    }


def update_extremes(ex, scen, scen_name, eval_path):
    """Update an extremes dict in-place from a single scenario."""
    min_v  = scen.get("dss_min_v")
    max_v  = scen.get("dss_max_v")   # tracked only globally (not per-label)

    if min_v is not None:
        if ex["min_pu_v"] is None or min_v < ex["min_pu_v"]:
            ex["min_pu_v"]     = min_v
            ex["min_pu_v_loc"] = (eval_path, scen_name)

    vunbal     = scen.get("nema_vub_max_pct")
    vunbal_bus = scen.get("nema_vub_bus")
    if vunbal is not None:
        vunbal = safe_float(vunbal)
        if ex["max_vunbal"] is None or vunbal > ex["max_vunbal"]:
            ex["max_vunbal"]     = vunbal
            ex["max_vunbal_bus"] = vunbal_bus
            ex["max_vunbal_loc"] = (eval_path, scen_name)

    branch_pct  = scen.get("max_branch_loading_pct")
    branch_name = scen.get("max_loaded_branch")
    if branch_pct is not None:
        if ex["max_branch_pct"] is None or branch_pct > ex["max_branch_pct"]:
            ex["max_branch_pct"] = branch_pct
            ex["max_branch_loc"] = (eval_path, scen_name, branch_name)

    for lk in LOSSES_KEYS:
        val = scen.get(lk)
        if val is not None:
            losses = safe_float(val)
            if ex["max_losses_kw"] is None or losses > ex["max_losses_kw"]:
                ex["max_losses_kw"]  = losses
                ex["max_losses_loc"] = (eval_path, scen_name)
            break

    trafo_load_pct = scen.get("trafo_load_pct")
    if trafo_load_pct is not None:
        trafo_load_pct = safe_float(trafo_load_pct)
        if ex["max_trafo_load_pct"] is None or trafo_load_pct > ex["max_trafo_load_pct"]:
            ex["max_trafo_load_pct"] = trafo_load_pct
            ex["max_trafo_load_loc"] = (eval_path, scen_name)

    trafo_loss_kw = scen.get("dss_trafo_loss_kw")
    if trafo_loss_kw is not None:
        trafo_loss_kw = safe_float(trafo_loss_kw)
        if ex["max_trafo_loss_kw"] is None or trafo_loss_kw > ex["max_trafo_loss_kw"]:
            ex["max_trafo_loss_kw"] = trafo_loss_kw
            ex["max_trafo_loss_loc"] = (eval_path, scen_name)

    line_loss_kw = scen.get("dss_line_loss_kw")
    if line_loss_kw is not None:
        line_loss_kw = safe_float(line_loss_kw)
        if ex["max_line_loss_kw"] is None or line_loss_kw > ex["max_line_loss_kw"]:
            ex["max_line_loss_kw"] = line_loss_kw
            ex["max_line_loss_loc"] = (eval_path, scen_name)


def print_extremes(ex, label=None):
    prefix = f"  [{label}] " if label else "  "
    if ex["min_pu_v"] is not None:
        print(f"{prefix}Min pu voltage : {ex['min_pu_v']:.4f}")
        print(f"           -> {ex['min_pu_v_loc'][0]}  |  {ex['min_pu_v_loc'][1]}")
    if ex["max_vunbal"] is not None:
        bus_str = f"  bus: {ex['max_vunbal_bus']}" if ex["max_vunbal_bus"] else ""
        print(f"{prefix}Max NEMA VUB   : {ex['max_vunbal']:.4f}%{bus_str}")
        print(f"           -> {ex['max_vunbal_loc'][0]}  |  {ex['max_vunbal_loc'][1]}")
    if ex["max_branch_pct"] is not None:
        b = ex["max_branch_loc"]
        print(f"{prefix}Max branch load: {ex['max_branch_pct']:.2f}%  (branch: {b[2]})")
        print(f"           -> {b[0]}  |  {b[1]}")
    if ex["max_losses_kw"] is not None:
        print(f"{prefix}Max losses     : {ex['max_losses_kw']:.4f} kW  (total)")
        print(f"           -> {ex['max_losses_loc'][0]}  |  {ex['max_losses_loc'][1]}")
    if ex["max_trafo_loss_kw"] is not None:
        print(f"{prefix}Max trafo loss : {ex['max_trafo_loss_kw']:.4f} kW")
        print(f"           -> {ex['max_trafo_loss_loc'][0]}  |  {ex['max_trafo_loss_loc'][1]}")
    if ex["max_line_loss_kw"] is not None:
        print(f"{prefix}Max line loss  : {ex['max_line_loss_kw']:.4f} kW")
        print(f"           -> {ex['max_line_loss_loc'][0]}  |  {ex['max_line_loss_loc'][1]}")
    if ex["max_trafo_load_pct"] is not None:
        print(f"{prefix}Max trafo load : {ex['max_trafo_load_pct']:.2f}%")
        print(f"           -> {ex['max_trafo_load_loc'][0]}  |  {ex['max_trafo_load_loc'][1]}")


def extremes_to_dict(ex):
    return {
        "total_scenarios":          ex["total_scenarios"],
        "nonconverged_scenarios":   ex["nonconverged_scenarios"],
        "violation_scenarios":      ex["violation_scenarios"],
        "violation_kind_tally":     ex["violation_kind_tally"],
        "worst_violation":          ex["worst_violation"],
        "min_pu_voltage":           {"value": ex["min_pu_v"],
                                     "file":  ex["min_pu_v_loc"][0] if ex["min_pu_v_loc"] else None,
                                     "scenario": ex["min_pu_v_loc"][1] if ex["min_pu_v_loc"] else None},
        "max_nema_vub_pct":         {"value": ex["max_vunbal"],
                                     "bus":   ex["max_vunbal_bus"],
                                     "file":  ex["max_vunbal_loc"][0] if ex["max_vunbal_loc"] else None,
                                     "scenario": ex["max_vunbal_loc"][1] if ex["max_vunbal_loc"] else None},
        "max_branch_loading_pct":   {"value":  ex["max_branch_pct"],
                                     "branch": ex["max_branch_loc"][2] if ex["max_branch_loc"] else None,
                                     "file":   ex["max_branch_loc"][0] if ex["max_branch_loc"] else None,
                                     "scenario": ex["max_branch_loc"][1] if ex["max_branch_loc"] else None},
        "max_losses_kw":            {"value": ex["max_losses_kw"],
                                     "file":  ex["max_losses_loc"][0] if ex["max_losses_loc"] else None,
                                     "scenario": ex["max_losses_loc"][1] if ex["max_losses_loc"] else None},
        "max_trafo_load_pct":       {"value": ex["max_trafo_load_pct"],
                                     "file":  ex["max_trafo_load_loc"][0] if ex["max_trafo_load_loc"] else None,
                                     "scenario": ex["max_trafo_load_loc"][1] if ex["max_trafo_load_loc"] else None},
        "max_trafo_loss_kw":        {"value": ex["max_trafo_loss_kw"],
                                     "file":  ex["max_trafo_loss_loc"][0] if ex["max_trafo_loss_loc"] else None,
                                     "scenario": ex["max_trafo_loss_loc"][1] if ex["max_trafo_loss_loc"] else None},
        "max_line_loss_kw":         {"value": ex["max_line_loss_kw"],
                                     "file":  ex["max_line_loss_loc"][0] if ex["max_line_loss_loc"] else None,
                                     "scenario": ex["max_line_loss_loc"][1] if ex["max_line_loss_loc"] else None},
    }


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    args = build_arg_parser().parse_args()
    data_dir   = args.dir
    P_FRAC, Q_FRAC     = args.p_frac, args.q_frac
    DIFF_ABS, DIFF_PCT = args.diff_abs, args.diff_pct

    # Global extremes (all labels combined)
    g = empty_extremes()
    g["max_pu_v"]     = None
    g["max_pu_v_loc"] = None

    # Per-label extremes
    label_ex = {lbl: empty_extremes() for lbl in LABELS}

    files_checked       = 0
    files_missing       = []
    report_violations   = []

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------
    for label in LABELS:
        lex = label_ex[label]

        for i in range(1, args.iters + 1):
            master_path = first_existing(candidate_master_filenames(data_dir, label, i))
            eval_path   = first_existing(candidate_eval_filenames(data_dir, label, i))

            if not eval_path:
                files_missing.append(f"evaluation_results [{label}, iter {i}]")
                continue
            if not master_path:
                files_missing.append(f"master_dg_placements [{label}, iter {i}]")

            dg_caps = load_dg_caps(master_path)

            try:
                with open(eval_path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  [warn] could not read {eval_path}: {e}")
                continue

            files_checked += 1
            scenario_details = data.get("scenario_details", {})

            for scen_name, scen in scenario_details.items():

                lex["total_scenarios"] += 1
                g["total_scenarios"]   += 1

                # ---- AC convergence check ------------------------------------
                # dss_loss_kw is only populated when dss.Solution.Converged()
                # returned True (see main_initial.py / test_initial.py). If it's
                # missing, OpenDSS failed to solve this scenario's AC power flow
                # and the scenario is excluded from every downstream check below,
                # consistent with resultspreadsheet.py's "valid_scenarios" filter.
                if scen.get("dss_loss_kw") is None:
                    lex["nonconverged_scenarios"] += 1
                    g["nonconverged_scenarios"]   += 1
                    continue

                # ---- skip blackout scenarios --------------------------------
                min_v = scen.get("dss_min_v")
                if min_v is not None and min_v == 0.0:
                    continue

                # ---- update extremes (per-label and global) -----------------
                update_extremes(lex, scen, scen_name, eval_path)
                update_extremes(g,   scen, scen_name, eval_path)

                # global max pu voltage (not in update_extremes to keep it global-only)
                max_v = scen.get("dss_max_v")
                if max_v is not None:
                    if g["max_pu_v"] is None or max_v > g["max_pu_v"]:
                        g["max_pu_v"]     = max_v
                        g["max_pu_v_loc"] = (eval_path, scen_name)

                # ---- Voltage limit violation check (hard cutoff, zero tolerance;
                # mirrors the MILP's own exact U_min/U_max constraint) ----------
                scen_reasons = []
                if min_v is not None and min_v < V_MIN_PU:
                    exc = (V_MIN_PU - min_v) / V_MIN_PU * 100.0
                    scen_reasons.append({"kind": "V_UNDER",
                                          "detail": f"Min voltage={min_v:.4f}pu < {V_MIN_PU}pu  (-{exc:.2f}%)",
                                          "exceedance_pct": round(exc, 4)})
                if max_v is not None and max_v > V_MAX_PU:
                    exc = (max_v - V_MAX_PU) / V_MAX_PU * 100.0
                    scen_reasons.append({"kind": "V_OVER",
                                          "detail": f"Max voltage={max_v:.4f}pu > {V_MAX_PU}pu  (+{exc:.2f}%)",
                                          "exceedance_pct": round(exc, 4)})

                # ---- Thermal / unbalance violation checks (hard cutoff, zero
                # tolerance) ---------------------------------------------------
                branch_pct = scen.get("max_branch_loading_pct")
                if branch_pct is not None and branch_pct > BRANCH_MAX_PCT:
                    exc = exceedance_pct(branch_pct, BRANCH_MAX_PCT)
                    scen_reasons.append({"kind": "BRANCH_OVERLOAD",
                                          "detail": f"Max branch loading={branch_pct:.2f}% > {BRANCH_MAX_PCT:.0f}%  (+{exc:.2f}%)",
                                          "exceedance_pct": round(exc, 4)})

                trafo_pct = scen.get("trafo_load_pct")
                if trafo_pct is not None and trafo_pct > TRAFO_MAX_PCT:
                    exc = exceedance_pct(trafo_pct, TRAFO_MAX_PCT)
                    scen_reasons.append({"kind": "TRAFO_OVERLOAD",
                                          "detail": f"Transformer loading={trafo_pct:.2f}% > {TRAFO_MAX_PCT:.0f}%  (+{exc:.2f}%)",
                                          "exceedance_pct": round(exc, 4)})

                vunbal_pct = scen.get("nema_vub_max_pct")
                if vunbal_pct is not None:
                    vunbal_pct = safe_float(vunbal_pct)
                    if vunbal_pct > NEMA_VUB_MAX_PCT:
                        exc = exceedance_pct(vunbal_pct, NEMA_VUB_MAX_PCT)
                        scen_reasons.append({"kind": "NEMA_VUB_OVER",
                                              "detail": f"NEMA VUB={vunbal_pct:.2f}% > {NEMA_VUB_MAX_PCT:.1f}%  (+{exc:.2f}%)",
                                              "exceedance_pct": round(exc, 4)})

                for r in scen_reasons:
                    for ex in (lex, g):
                        ex["violation_kind_tally"][r["kind"]] = \
                            ex["violation_kind_tally"].get(r["kind"], 0) + 1
                    exc = r.get("exceedance_pct", 0.0)
                    wv = {"exceedance_pct": exc, "kind": r["kind"],
                          "detail": r["detail"], "dg_id": "GRID",
                          "scenario": scen_name, "file": eval_path}
                    for ex in (lex, g):
                        if ex["worst_violation"] is None or \
                                exc > ex["worst_violation"]["exceedance_pct"]:
                            ex["worst_violation"] = wv

                # ---- DG dispatch violation checks ---------------------------
                dg_dispatch = scen.get("dg_dispatch", {}) or {}
                flagged = []

                for dg_id, dg in dg_dispatch.items():
                    s_cap = dg_caps.get(str(dg_id), safe_float(dg.get("s_cap_kva")))
                    if not s_cap:
                        continue

                    max_p  = P_FRAC * s_cap
                    max_q  = Q_FRAC * s_cap
                    tol_p  = P_TOL_FRAC * max_p
                    tol_q  = Q_TOL_FRAC * max_q

                    p_milp = safe_float(dg.get("p_milp_kw"))
                    q_milp = safe_float(dg.get("q_milp_kvar"))
                    p_ods  = safe_float(dg.get("p_ods_kw"))
                    q_ods  = safe_float(dg.get("q_ods_kvar"))
                    s_ods  = safe_float(dg.get("s_ods_kva"))

                    reasons = []

                    def _check(actual, limit, tol, kind, label_str):
                        if abs(actual) > limit + tol:
                            exc = exceedance_pct(actual, limit)
                            detail = f"{label_str} > limit={limit:.2f}  (+{exc:.2f}%)"
                            reasons.append({"kind": kind, "detail": detail,
                                            "exceedance_pct": round(exc, 4)})

                    _check(p_milp, max_p, tol_p, "P_LIMIT_MILP", f"MILP P={p_milp:.2f}kW")
                    _check(p_ods,  max_p, tol_p, "P_LIMIT_ODS",  f"OpenDSS P={p_ods:.2f}kW")
                    _check(q_milp, max_q, tol_q, "Q_LIMIT_MILP", f"MILP Q={q_milp:.2f}kvar")
                    _check(q_ods,  max_q, tol_q, "Q_LIMIT_ODS",  f"OpenDSS Q={q_ods:.2f}kvar")

                    if s_ods > s_cap:
                        exc = exceedance_pct(s_ods, s_cap)
                        reasons.append({"kind": "S_RATING_ODS",
                                        "detail": f"OpenDSS S={s_ods:.2f}kVA > S_cap={s_cap:.2f}kVA  (+{exc:.2f}%)",
                                        "exceedance_pct": round(exc, 4)})

                    diff_p  = abs(p_milp - p_ods)
                    diff_q  = abs(q_milp - q_ods)
                    denom_p = max(abs(p_milp), abs(p_ods), 1e-6)
                    denom_q = max(abs(q_milp), abs(q_ods), 1e-6)
                    if diff_p > DIFF_ABS and (diff_p / denom_p * 100.0) > DIFF_PCT:
                        pct_p = diff_p / denom_p * 100.0
                        reasons.append({"kind": "P_DIVERGENCE",
                                        "detail": (f"P diverges: MILP={p_milp:.2f}kW vs ODS={p_ods:.2f}kW "
                                                   f"(\u0394={diff_p:.2f}kW, {pct_p:.1f}%)"),
                                        "exceedance_pct": round(pct_p, 4)})
                    if diff_q > DIFF_ABS and (diff_q / denom_q * 100.0) > DIFF_PCT:
                        pct_q = diff_q / denom_q * 100.0
                        reasons.append({"kind": "Q_DIVERGENCE",
                                        "detail": (f"Q diverges: MILP={q_milp:.2f}kvar vs ODS={q_ods:.2f}kvar "
                                                   f"(\u0394={diff_q:.2f}kvar, {pct_q:.1f}%)"),
                                        "exceedance_pct": round(pct_q, 4)})

                    if reasons:
                        flagged.append((dg_id, reasons))
                        for r in reasons:
                            # tally per-label and global
                            for ex in (lex, g):
                                ex["violation_kind_tally"][r["kind"]] = \
                                    ex["violation_kind_tally"].get(r["kind"], 0) + 1
                            # worst violation per-label and global
                            exc = r.get("exceedance_pct", 0.0)
                            wv = {"exceedance_pct": exc, "kind": r["kind"],
                                  "detail": r["detail"], "dg_id": dg_id,
                                  "scenario": scen_name, "file": eval_path}
                            for ex in (lex, g):
                                if ex["worst_violation"] is None or \
                                        exc > ex["worst_violation"]["exceedance_pct"]:
                                    ex["worst_violation"] = wv

                if flagged or scen_reasons:
                    lex["violation_scenarios"] += 1
                    g["violation_scenarios"]   += 1

                    print("=" * 95)
                    print(f"VIOLATION  |  file: {eval_path}  |  scenario: {scen_name}")
                    if master_path:
                        print(f"            (DG ratings from: {master_path})")
                    print("-" * 95)
                    if scen_reasons:
                        print(f"  >> GRID:")
                        for r in scen_reasons:
                            print(f"       - {r['detail']}")
                    for dg_id, reasons in flagged:
                        print(f"  >> DG {dg_id}:")
                        for r in reasons:
                            print(f"       - {r['detail']}")
                    print()
                    print("  All DGs dispatched in this scenario:")
                    all_dgs_snapshot = []
                    for dg_id, dg in dg_dispatch.items():
                        s_cap = dg_caps.get(str(dg_id), safe_float(dg.get("s_cap_kva")))
                        snap = {
                            "dg_id":       dg_id,
                            "type":        dg.get("type", "?"),
                            "p_milp_kw":   safe_float(dg.get("p_milp_kw")),
                            "q_milp_kvar": safe_float(dg.get("q_milp_kvar")),
                            "p_ods_kw":    safe_float(dg.get("p_ods_kw")),
                            "q_ods_kvar":  safe_float(dg.get("q_ods_kvar")),
                            "s_ods_kva":   safe_float(dg.get("s_ods_kva")),
                            "s_cap_kva":   s_cap,
                        }
                        all_dgs_snapshot.append(snap)
                        print(
                            f"    DG {dg_id} [{snap['type']}]: "
                            f"MILP P={snap['p_milp_kw']:.2f}kW Q={snap['q_milp_kvar']:.2f}kvar | "
                            f"ODS P={snap['p_ods_kw']:.2f}kW Q={snap['q_ods_kvar']:.2f}kvar "
                            f"S={snap['s_ods_kva']:.2f}kVA | S_cap={snap['s_cap_kva']:.2f}kVA"
                        )
                    print()
                    flagged_dgs_out = [{"dg_id": dg_id, "reasons": reasons}
                                       for dg_id, reasons in flagged]
                    if scen_reasons:
                        flagged_dgs_out.insert(0, {"dg_id": "GRID", "reasons": scen_reasons})
                    report_violations.append({
                        "label":      label,
                        "iteration":  i,
                        "eval_file":  eval_path,
                        "master_file": master_path,
                        "scenario":   scen_name,
                        "flagged_dgs": flagged_dgs_out,
                        "all_dgs_in_scenario": all_dgs_snapshot,
                    })

    # ==========================================================================
    # GLOBAL SUMMARY
    # ==========================================================================
    W = 95
    print("#" * W)
    print("GLOBAL SUMMARY")
    print("#" * W)
    print(f"Files checked (evaluation_results): {files_checked}")
    print(f"Total scenarios scanned           : {g['total_scenarios']}")
    print(f"AC non-convergence (OpenDSS)       : {g['nonconverged_scenarios']} / {g['total_scenarios']}")
    print(f"Scenarios with violations (total) : {g['violation_scenarios']}")
    if files_missing:
        print(f"Missing files                     : {len(files_missing)} (not found, skipped)")

    if g["violation_kind_tally"]:
        print("\nViolation breakdown by check type (global):")
        for kind, count in sorted(g["violation_kind_tally"].items(), key=lambda kv: -kv[1]):
            print(f"  {kind:<16} {count}")

    if g["worst_violation"]:
        wv = g["worst_violation"]
        print(f"\nWORST single violation:  +{wv['exceedance_pct']:.2f}% above limit")
        print(f"  kind:     {wv['kind']}")
        print(f"  detail:   {wv['detail']}")
        print(f"  DG:       {wv['dg_id']}  |  scenario: {wv['scenario']}")
        print(f"  file:     {wv['file']}")

    print(f"\nGlobal extremes (non-blackout scenarios):")
    if g["min_pu_v"] is not None:
        print(f"  Min pu voltage : {g['min_pu_v']:.4f}")
        print(f"           -> {g['min_pu_v_loc'][0]}  |  {g['min_pu_v_loc'][1]}")
    if g["max_pu_v"] is not None:
        print(f"  Max pu voltage : {g['max_pu_v']:.4f}")
        print(f"           -> {g['max_pu_v_loc'][0]}  |  {g['max_pu_v_loc'][1]}")
    print_extremes(g)  # prints vunbal, branch, losses

    # ==========================================================================
    # PER-LABEL SUMMARY
    # ==========================================================================
    print()
    print("#" * W)
    print("PER-LABEL SUMMARY")
    print("#" * W)
    for label in LABELS:
        lex = label_ex[label]
        print()
        print(f"  LABEL: {label}")
        print(f"  {'-' * 85}")
        print(f"  Total scenarios scanned  : {lex['total_scenarios']}")
        print(f"  AC non-convergence       : {lex['nonconverged_scenarios']} / {lex['total_scenarios']}")
        print(f"  Scenarios with violations: {lex['violation_scenarios']}")
        if lex["violation_kind_tally"]:
            print("  Breakdown:")
            for kind, count in sorted(lex["violation_kind_tally"].items(), key=lambda kv: -kv[1]):
                print(f"    {kind:<16} {count}")
        if lex["worst_violation"]:
            wv = lex["worst_violation"]
            print(f"  Worst violation: +{wv['exceedance_pct']:.2f}%  [{wv['kind']}]  DG {wv['dg_id']}")
            print(f"    {wv['detail']}")
            print(f"    scenario: {wv['scenario']}  |  file: {wv['file']}")
        print("  Extremes:")
        print_extremes(lex)
    print()

    # ==========================================================================
    # WRITE JSON REPORT
    # ==========================================================================
    report = {
        "config": {
            "p_frac":              P_FRAC,
            "q_frac":              Q_FRAC,
            "p_tolerance_frac":    P_TOL_FRAC,
            "q_tolerance_frac":    Q_TOL_FRAC,
            "s_tolerance":         0,
            "diff_abs_threshold":  DIFF_ABS,
            "diff_pct_threshold":  DIFF_PCT,
            "v_min_pu":            V_MIN_PU,
            "v_max_pu":            V_MAX_PU,
            "v_tolerance":         0,
            "branch_max_pct":      BRANCH_MAX_PCT,
            "trafo_max_pct":       TRAFO_MAX_PCT,
            "nema_vub_max_pct":    NEMA_VUB_MAX_PCT,
            "thermal_vub_tolerance": 0,
            "labels_checked":      LABELS,
            "iterations_per_label": args.iters,
            "data_dir":            data_dir,
        },
        "global_summary": {
            "files_checked":             files_checked,
            "files_missing":             files_missing,
            **extremes_to_dict(g),
            "global_max_pu_voltage": {
                "value":    g["max_pu_v"],
                "file":     g["max_pu_v_loc"][0] if g["max_pu_v_loc"] else None,
                "scenario": g["max_pu_v_loc"][1] if g["max_pu_v_loc"] else None,
            },
        },
        "per_label_summary": {
            label: extremes_to_dict(label_ex[label]) for label in LABELS
        },
        "violations": report_violations,
    }

    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Structured JSON report written to: {args.json_out}")


if __name__ == "__main__":
    main()