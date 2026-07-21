"""
SAA Convergence Study Orchestrator
====================================
Sweeps N (scenarios-per-batch) and Lambda (risk aversion) at a given PGA,
with a consistent Gurobi MIP gap across all runs.

File naming convention: pga{PGA}N{N}L{Lambda}
  e.g. saa_scenario_pga025N30_1.json          ← shared fault file (no lambda)
       master_dg_placements_pga025N30L000_1.json
       evaluation_results_500_pga025N30L000_1.json
       SAA_Compiled_Results_Advanced_pga025N30L000.xlsx

Fault scenarios are generated once per (PGA, N) pair and shared across all
lambda values so the seismic draws are identical — making lambda comparisons
fair while saving regeneration time.

Usage:
    python saa_convergence_study.py
"""

import subprocess
import os
import sys
import time
import itertools

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# -----------------------------------------------------------------------
# CONVERGENCE STUDY CONFIGURATION  — edit here only
# -----------------------------------------------------------------------
PGA            = 0.40          # Peak Ground Acceleration (g)
MIP_GAP        = 0.01           # Gurobi MIP tolerance (0.01 = 1%)
NUM_ITERATIONS = 30             # R — independent SAA replications per combination
START_ITER     = 1             # Resume from this iteration (set to 1 for a full run).
                                # Iterations before this are assumed already complete;
                                # fault generation (STEP 1) is skipped when START_ITER > 1.

# --- Sweep lists. Each defaults to a SINGLE baseline value so the default run
#     reproduces today's behaviour exactly. Add values to a list to sweep it.
N_VALUES       = [30]           # Scenario count N (convergence: [15, 30, 60])
LAMBDA_VALUES  = [0.75]         # Risk-aversion λ: 0.0 = risk-neutral, 1.0 = pure CVaR (#4)
RHO_VALUES     = [0.0]          # Common-cause correlation ρ (e.g. [0.0, 0.3, 0.6])
SIGMA_VALUES   = [0.5]          # Repair-duration lognormal σ (#5 σ-part; e.g. [0.3, 0.5, 0.7])
ALPHA_VALUES   = [0.90]         # CVaR confidence α (#8; e.g. [0.80, 0.90, 0.95])
BUDGET_VALUES  = [1000000.0]    # Investment budget B_max (#7 Pareto; e.g. [4e5, 6e5, 8e5, 1e6])
MEDIAN_VALUES  = [1.0]          # Repair-MEDIAN multiplier (#5 median-part; e.g. [0.5, 1.0, 1.5])
CONDOCC_VALUES = [1.0]          # Conditional-event occurrence (#6; annualized ≈ 0.031)
# Scenario files depend on (PGA, N, ρ, σ, median); the solve depends on
# (λ, α, budget, cond-occ). Scenarios are regenerated per full combo (cheap,
# deterministic via fixed seed) so each combo's files never collide.
# -----------------------------------------------------------------------


def run(cmd, label=""):
    """Wrapper: runs a subprocess, prints timing, raises on failure."""
    t0 = time.time()
    print(f"  >>> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)
    elapsed = time.time() - t0
    print(f"  <<< Done in {elapsed/60:.1f} min" + (f" [{label}]" if label else ""))


def make_label(pga, n, lam, rho=0.0, sigma=0.5,
               alpha=0.90, budget=1000000.0, median=1.0, condocc=1.0):
    """Canonical run label: e.g. pga030N30L075Y15R010_rho00_sig05.

    Non-baseline sweep values append a compact tag (_a/_b/_m/_c) so swept runs
    get unique filenames, while a baseline run keeps the original label exactly
    (no orphaned files from earlier studies).
    """
    label = (f"pga{int(pga*100):03d}N{n}L{int(lam*100):03d}Y15R010"
             f"_rho{int(rho*100):02d}_sig{int(sigma*100):02d}")
    if abs(alpha - 0.90) > 1e-9:
        label += f"_a{int(round(alpha*100)):02d}"
    if abs(budget - 1000000.0) > 1e-6:
        label += f"_b{int(round(budget/1000))}k"
    if abs(median - 1.0) > 1e-9:
        label += f"_m{int(round(median*100)):03d}"
    if abs(condocc - 1.0) > 1e-9:
        label += f"_c{int(round(condocc*1000)):04d}"
    return label


# Flat list of all sweep combinations. Order MUST match the unpacking below.
COMBOS = list(itertools.product(
    N_VALUES, RHO_VALUES, SIGMA_VALUES, MEDIAN_VALUES,
    LAMBDA_VALUES, ALPHA_VALUES, BUDGET_VALUES, CONDOCC_VALUES))

print("=" * 70)
print(f"  SAA CONVERGENCE / SENSITIVITY STUDY")
print(f"  PGA = {PGA}g  |  MIPGap = {MIP_GAP*100:.1f}%  |  M = {NUM_ITERATIONS} iterations")
print(f"  N={N_VALUES}  λ={LAMBDA_VALUES}  ρ={RHO_VALUES}  σ={SIGMA_VALUES}")
print(f"  α={ALPHA_VALUES}  budget={BUDGET_VALUES}  median={MEDIAN_VALUES}  cond_occ={CONDOCC_VALUES}")
print(f"  Total combinations: {len(COMBOS)}  ({len(COMBOS) * NUM_ITERATIONS} MILP solves)")
print("=" * 70)

study_start = time.time()

for (N, RHO, SIGMA, MEDIAN, LAMBDA, ALPHA, BUDGET, CONDOCC) in COMBOS:
    LABEL = make_label(PGA, N, LAMBDA, RHO, SIGMA, ALPHA, BUDGET, MEDIAN, CONDOCC)

    print(f"\n{'='*70}")
    print(f"  STARTING  N={N} λ={LAMBDA} ρ={RHO} σ={SIGMA} med={MEDIAN} "
          f"α={ALPHA} B=${BUDGET:,.0f} cond={CONDOCC}")
    print(f"  Label={LABEL}")
    print(f"{'='*70}")
    combo_start = time.time()

    # --------------------------------------------------------------
    # STEP 1: Generate fault scenarios (depend on N, PGA, ρ, σ, median)
    #   generatefault.py <N> <PGA> <LABEL> <M> <RHO> <SEED> <SIGMA> <MEDIAN>
    # --------------------------------------------------------------
    if START_ITER > 1:
        print(f"\n[{LABEL}] STEP 1: SKIPPED (resuming from iter {START_ITER}; "
              f"fault files already exist).")
    else:
        print(f"\n[{LABEL}] STEP 1: Generating fault scenario files...")
        run(
            ["python", "generatefault.py",
             N, PGA, LABEL, NUM_ITERATIONS, RHO, 42, SIGMA, MEDIAN],
            label=f"generatefault {LABEL}"
        )

    # --------------------------------------------------------------
    # STEP 2: M × (MILP solve + upper-bound evaluation)
    #   main_initial.py  <iter> <LABEL> <MIPGap> <Lambda> <Alpha> <Budget> <CondOcc>
    #   test_initial.py  <iter> <LABEL>
    # --------------------------------------------------------------
    print(f"\n[{LABEL}] STEP 2: Running {NUM_ITERATIONS} SAA iterations...")
    for i in range(START_ITER, NUM_ITERATIONS + 1):
        print(f"\n  {'='*60}")
        print(f"  {LABEL} | ITER {i:>2d}/{NUM_ITERATIONS}")
        print(f"  {'='*60}")

        print(f"\n  [iter {i}] Solving master MILP "
              f"(λ={LAMBDA}, α={ALPHA}, B=${BUDGET:,.0f}, cond={CONDOCC}, MIPGap={MIP_GAP*100:.1f}%)...")
        run(
            ["python", "main_initial.py",
             i, LABEL, MIP_GAP, LAMBDA, ALPHA, BUDGET, CONDOCC],
            label=f"main_initial iter={i}"
        )

        print(f"\n  [iter {i}] Evaluating upper bound (500 scenarios)...")
        run(
            ["python", "test_initial.py",
             i, LABEL],
            label=f"test_initial iter={i}"
        )

    # --------------------------------------------------------------
    # STEP 3: Compile SAA statistics into Excel
    # --------------------------------------------------------------
    print(f"\n[{LABEL}] STEP 3: Compiling results into Excel...")
    run(
        ["python", "resultspreadsheet.py", LABEL],
        label=f"resultspreadsheet {LABEL}"
    )

    combo_elapsed = (time.time() - combo_start) / 3600
    print(f"\n  ✅  {LABEL} COMPLETE  ({combo_elapsed:.2f} hours)")
    print(f"      Output: SAA_Compiled_Results_Advanced_{LABEL}.xlsx")

total_elapsed = (time.time() - study_start) / 3600
print(f"\n{'='*70}")
print(f"  STUDY COMPLETE  ({total_elapsed:.2f} hours total)")
print(f"  Result files:")
for (N, RHO, SIGMA, MEDIAN, LAMBDA, ALPHA, BUDGET, CONDOCC) in COMBOS:
    LABEL = make_label(PGA, N, LAMBDA, RHO, SIGMA, ALPHA, BUDGET, MEDIAN, CONDOCC)
    print(f"    SAA_Compiled_Results_Advanced_{LABEL}.xlsx")
print("=" * 70)
