import random
import json
import os
import math
import sys
import opendssdirect as dss
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import csv

# Set working directory to script location
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# --- SIMULATION PARAMETERS ---
BIG_SCENARIO = 500    # big evaluation set — always fixed at 500
DSS_FILE = "ieee37.dss"

def load_bus_coords():
    """Extracts bus coordinates from the IEEE 37-bus CSV file."""
    coords = {}
    if os.path.exists("IEEE37_BusXY.csv"):
        with open("IEEE37_BusXY.csv", "r") as f:
            reader = csv.reader(f)
            for parts in reader:
                # Assuming format: NodeName, X, Y
                if len(parts) >= 3:
                    try:
                        bus_name = parts[0].strip().lower()
                        coords[bus_name] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
    return coords

# Pole Fragility Parameters from Baghmisheh & Mahsuli (2021)
POLE_PARAMS = {
    "9m": (0.569, 0.375),
    "12m": (0.462, 0.353),
    "15m": (0.386, 0.453)
}

def calculate_single_pole_failure(pga, theta, beta, z_common=0.0, rho=0.0):
    """
    S-Curve probability of a single pole failing due to PGA.

    Correlated-failure extension (one-factor common-cause model, standard in
    seismic probabilistic risk assessment): the fragility dispersion is split as
        beta^2 = rho*beta^2 (shared across all poles) + (1-rho)*beta^2 (pole-specific)
    Pole capacity: C = theta * exp(beta * eps), with
        eps = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    where Z ~ N(0,1) is drawn ONCE per earthquake scenario (event-wide effects:
    soil response, ground-motion variability) and eps_i is independent per pole.
    Conditional on Z, the per-pole failure probability becomes:
        P_f(a | Z) = Phi( (ln(a/theta)/beta - sqrt(rho)*Z) / sqrt(1-rho) )

    rho = 0 recovers the original independent model exactly. rho is the fraction
    of total fragility variance attributed to the common cause; positive rho
    fattens the tail of the failed-lines distribution while leaving the
    marginal (unconditional) per-pole failure probability unchanged.
    """
    if pga <= 0:
        return 0.0
    if rho <= 0.0:
        return norm.cdf(math.log(pga / theta) / beta)
    return norm.cdf((math.log(pga / theta) / beta - math.sqrt(rho) * z_common)
                    / math.sqrt(1.0 - rho))

def extract_grid_data():
    """Extracts line lengths from OpenDSS and calculates poles per line."""
    dss.Command('Clear')
    dss.Command(f'Compile "{DSS_FILE}"')
    
    line_data = {}
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        name = dss.Lines.Name().lower()
        
        # OpenDSS Length is scalar; assuming units=kft
        length_kft = dss.Lines.Length()
        length_m = length_kft * 304.8 
        
        # Pole every 50m, representative from jica;
        # applied uniformly to translate line length into pole count.
        POLE_SPAN_M = 50.0
        num_poles = math.ceil(length_m / POLE_SPAN_M) + 1
        
        # Assign a random pole type to the line
        pole_type = "12m"
        
        # FIXED: Removed the extra 'l' from the string formatting
        line_data[name] = {
            "length_m": length_m,
            "num_poles": num_poles,
            "pole_type": pole_type
        }
        dss.Lines.Next()
        
    return line_data

def calculate_lognormal_repair_time(num_broken_lines, total_lines=37, sigma=0.5, median_scale=1.0):
    """
    Scenario duration model for post-earthquake distribution circuit restoration.

    Median formula is order-of-magnitude consistent with HAZUS Table 8-27
    (Distribution Circuits, FEMA 2020) restoration ranges:
        Slight   (~4 % circuits):  ~7 h
        Moderate (~12% circuits):  ~24 h
        Extensive(~50% circuits):  ~72 h
    The slope (114 / total_lines) spans this range across the network's damage
    states; it is not a point-by-point fit to HAZUS data.  Restoration
    uncertainty is instead covered by the sigma sensitivity (default 0.5,
    per HAZUS lognormal dispersion convention).

    When num_broken_lines == 0 the formula returns a median of 6 h, representing
    base dispatch and assessment time for a seismic operational disturbance
    with no physical line damage.  The 120 h cap reflects the operational
    horizon of a diesel-backed microgrid (~3 fuel resupply cycles at 40 h each)
    rather than a HAZUS damage-state limit.

    Sources: HAZUS Earthquake Model Technical Manual (FEMA, 2020), Table 8-27;
             Baghmisheh & Mahsuli (2021) for pole fragility parameters.
    """
    # Slope scales linearly with network size:
    #   114 h = target_max_median (120 h) − base_dispatch (6 h)
    slope = 114.0 / max(total_lines, 1)
    # median_scale (default 1.0 = unchanged) multiplies the whole median so the
    # repair-MEDIAN sensitivity (#5) can shift restoration times up/down while
    # keeping the damage-vs-duration shape and the lognormal σ dispersion intact.
    median_hours = (6.0 + slope * num_broken_lines) * median_scale

    mu = math.log(median_hours)

    duration = np.random.lognormal(mean=mu, sigma=sigma)
    return round(min(max(duration, 4.0), 120.0), 1)

def generate_seismic_faults(line_data, pga, num_scenarios, rho=0.0, sigma=0.5, median_scale=1.0):
    generated_scenarios = {
        "metadata": {
            "pga_g": pga,
            "rho": rho,
            "sigma": sigma,
            "num_scenarios": num_scenarios
        }
    }
    generated_scenarios['Normal_Operation'] = {
        "faults": [],
        "duration_hours": 0.0
    }

    print(f"🌍 Earthquake Simulation: {pga}g PGA"
          + (f" | common-cause correlation ρ={rho}" if rho > 0.0 else ""))

    # 1. Calculate the marginal (unconditional) failure probability per pole type
    pole_failure_probs = {}
    for p_type, (theta, beta) in POLE_PARAMS.items():
        prob = calculate_single_pole_failure(pga, theta, beta)
        pole_failure_probs[p_type] = prob
        print(f"📈 S-Curve Failure Prob ({p_type} pole, marginal): {prob*100:.2f}%")

    # 2. Run Monte Carlo Scenarios
    for s in range(1, num_scenarios + 1):
        failed_lines = []

        # Correlated failures: one common latent factor per earthquake scenario.
        # Conditional per-pole probabilities are recomputed given this draw;
        # rho=0 falls back to the precomputed marginal probabilities.
        if rho > 0.0:
            z_event = np.random.normal()
            scenario_pole_probs = {
                p_type: calculate_single_pole_failure(pga, theta, beta, z_event, rho)
                for p_type, (theta, beta) in POLE_PARAMS.items()
            }
        else:
            scenario_pole_probs = pole_failure_probs

        for line_name, data in line_data.items():
            p_type = data["pole_type"]
            n_poles = data["num_poles"]

            single_pole_prob = scenario_pole_probs[p_type]
            # Cascading Failure: Line fails if ANY of its poles fail
            # (poles are conditionally independent given the common factor)
            cascading_line_prob = 1.0 - math.pow((1.0 - single_pole_prob), n_poles)

            if random.random() < cascading_line_prob:
                failed_lines.append(line_name)
        
        # Calculate statistically rigorous repair time
        scenario_duration = calculate_lognormal_repair_time(len(failed_lines), len(line_data), sigma, median_scale)
        
        generated_scenarios[f"Earthquake_Scenario_{s:02d}"] = {
            "faults": failed_lines,
            "duration_hours": scenario_duration
        }
        
    return generated_scenarios

def plot_all_scenarios(scenarios_dict, pga):
    print("🎨 Generating Visual Fault Maps for all Scenarios...")
    
    # Use the global helper function so we don't repeat the CSV logic
    pos = load_bus_coords()

    dss.Command('Clear')
    dss.Command(f'Compile "{DSS_FILE}"')
    
    G = nx.Graph()
    line_to_edge = {}
    
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        name = dss.Lines.Name().lower()
        b1 = dss.Lines.Bus1().split('.')[0].lower().strip()
        b2 = dss.Lines.Bus2().split('.')[0].lower().strip()
        
        if b1 in pos and b2 in pos:
            G.add_edge(b1, b2)
            line_to_edge[name] = (b1, b2)
            
        dss.Lines.Next()

    # Only real storm entries carry a 'faults' list — this skips the 'metadata'
    # header and 'Normal_Operation' (both lack 'faults' and would KeyError here).
    storms = [s for s, v in scenarios_dict.items()
              if isinstance(v, dict) and 'faults' in v and s != 'Normal_Operation']
    if not storms:
        print("   (no storm scenarios to plot)")
        return
    # Grid adapts to the scenario count so N=15 / N=60 sweeps don't overflow a
    # hardcoded 5×6 (IndexError for N>30) or leave it mis-shaped for N<30.
    ncols = 6
    nrows = max(1, math.ceil(len(storms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, max(4.0, 3.2 * nrows)))
    axes = np.array(axes).reshape(-1)   # flatten so axes[idx] works for any nrows
    fig.suptitle(f"Monte Carlo Fault Maps (PGA = {pga}g)", fontsize=20, fontweight='bold')
    
    for idx, storm_name in enumerate(storms):
        ax = axes[idx]
        
        # Extract the faults list from the new dictionary structure
        faulted_lines = scenarios_dict[storm_name]["faults"]
        duration = scenarios_dict[storm_name]["duration_hours"]
        
        fault_edges = [line_to_edge[name] for name in faulted_lines if name in line_to_edge]
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=1.0)
        
        if fault_edges:
            nx.draw_networkx_edges(G, pos, edgelist=fault_edges, ax=ax, edge_color='red', width=2.5)
            
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color='black')
        
        if '799' in pos:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=['799'], node_size=60, node_shape='s', node_color='blue')   
        
        ax.set_title(f"{storm_name}\n({len(faulted_lines)} lines | {duration} hrs)", fontsize=10)
        ax.axis('off')

    # Hide any leftover axes when len(storms) doesn't fill the grid (e.g. N=15).
    for j in range(len(storms), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save instead of show(): show() blocks on an interactive backend, which would
    # stall the orchestrator at every combo during an unattended sweep.
    out_png = f"fault_maps_pga{int(pga*100):03d}.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"   🗺️  Saved fault map: {out_png}")

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # ARGUMENT PARSING — called by saa_convergence_study.py as:
    #   python generatefault.py <N> <PGA> <LABEL> <M> [RHO] [SEED] [SIGMA] [MEDIAN_SCALE]
    # Standalone defaults replicate the original hardcoded behaviour.
    # RHO (optional, default 0.0): common-cause correlation fraction for the
    # correlated pole-failure sensitivity study. 0.0 = independent (baseline).
    # Suggested sensitivity value: 0.3 (e.g. LABEL="pga030rho03").
    # -----------------------------------------------------------------------
    NUM_SCENARIOS  = int(sys.argv[1])   if len(sys.argv) > 1 else 30    # N scenarios per batch
    PGA_G          = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40  # PGA (g)
    LABEL          = sys.argv[3]        if len(sys.argv) > 3 else ""     # file-naming label
    NUM_ITERATIONS = int(sys.argv[4])   if len(sys.argv) > 4 else 30    # M independent batches
    RHO_COMMON     = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # common-cause correlation ρ
    BASE_SEED      = int(sys.argv[6])   if len(sys.argv) > 6 else 42    # RNG base seed (reproducibility)
    SIGMA_REPAIR   = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5   # lognormal σ for repair duration
    MEDIAN_SCALE   = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0   # repair-MEDIAN multiplier (#5); 1.0 = baseline

    print("⚡ Extracting line lengths from OpenDSS...")
    grid_line_data = extract_grid_data()

    # --- 1. Generate M independent SAA batches (each with N scenarios) ---
    # FIX: loop uses NUM_ITERATIONS (M), NOT NUM_SCENARIOS (N).
    # Original bug: for i in range(1, NUM_SCENARIOS + 1) → only N files, not M.
    print(f"\n📦 Generating {NUM_ITERATIONS} SAA batches × {NUM_SCENARIOS} scenarios "
          f"| PGA={PGA_G}g | Label={LABEL!r}...")

    for i in range(1, NUM_ITERATIONS + 1):
        # Seed per batch index: batch i is reproducible on its own, and batches
        # stay statistically independent of each other. The same (seed, i) pair
        # always regenerates the identical scenario file — required for
        # common-random-number sensitivity comparisons.
        random.seed(BASE_SEED + i)
        np.random.seed(BASE_SEED + i)
        saa_set = generate_seismic_faults(grid_line_data, PGA_G, NUM_SCENARIOS, RHO_COMMON, SIGMA_REPAIR, MEDIAN_SCALE)

        if LABEL:
            filename = f"saa_scenario_{LABEL}_{i}.json"
        else:
            filename = f"saa_scenario_pga025N30_{i}.json"   # original hardcoded name (standalone)
        with open(filename, "w") as outfile:
            json.dump(saa_set, outfile, indent=4)

        print(f"✅ Saved: {filename}")

    # --- 2. Generate ONE big evaluation set (500 scenarios) ---
    print(f"\n📦 Generating BIG dataset with {BIG_SCENARIO} scenarios...")

    # Offset keeps the evaluation set disjoint from every training batch seed,
    # preserving the out-of-sample property of the N'=500 evaluation.
    random.seed(BASE_SEED + 100000)
    np.random.seed(BASE_SEED + 100000)
    big_scenarios = generate_seismic_faults(grid_line_data, PGA_G, BIG_SCENARIO, RHO_COMMON, SIGMA_REPAIR, MEDIAN_SCALE)

    if LABEL:
        big_filename = f"500_scenarios_{LABEL}.json"
    else:
        big_filename = "500_scenarios_pga025N30.json"        # original hardcoded name (standalone)
    with open(big_filename, "w") as outfile:
        json.dump(big_scenarios, outfile, indent=4)

    print(f"✅ Saved: {big_filename}")

    # --- Optional: plot ONLY one sample set (otherwise chaos) ---
    print("\n🎨 Plotting one sample SAA set...")
    plot_all_scenarios(saa_set, PGA_G)