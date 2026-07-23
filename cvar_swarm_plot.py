"""
DG Investment Distribution — PGA x Lambda grouped box + strip plot.

Reads the compiled SAA result spreadsheets (per-iteration investment from the
"SAA Results" sheet) and plots the risk-neutral vs risk-averse investment
distribution across hazard intensities. Visualises the inverse-hazard finding:
the CVaR investment premium (gap between the two boxes) is large at low PGA and
collapses as PGA rises, while the risk-averse boxes are taller (higher s_L).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# (PGA label, lambda, compiled-results filename) — current naming convention.
RUNS = [
    ("0.23 g", 0.00, "SAA_Compiled_Results_Advanced_pga023N30L000Y15R010_rho00_sig50.xlsx"),
    ("0.23 g", 0.75, "SAA_Compiled_Results_Advanced_pga023N30L075Y15R010_rho00_sig50.xlsx"),
    ("0.30 g", 0.00, "SAA_Compiled_Results_Advanced_pga030N30L000Y15R010_rho00_sig50.xlsx"),
    ("0.30 g", 0.75, "SAA_Compiled_Results_Advanced_pga030N30L075Y15R010_rho00_sig50.xlsx"),
    ("0.40 g", 0.00, "SAA_Compiled_Results_Advanced_pga040N30L000Y15R010_rho00_sig50.xlsx"),
    ("0.40 g", 0.75, "SAA_Compiled_Results_Advanced_pga040N30L075Y15R010_rho00_sig50.xlsx"),
]

NEUTRAL = r"Risk-neutral ($\lambda$=0)"
AVERSE  = r"Risk-averse ($\lambda$=0.75)"


def load_iter_rows(fname):
    """Read the 30 per-iteration rows from the 'SAA Results' sheet."""
    try:
        df = pd.read_excel(fname, sheet_name="SAA Results")
    except PermissionError:
        raise SystemExit(
            f"[locked] '{fname}' is open in Excel — close it and re-run."
        )
    # keep only numeric-Iteration rows (drops AVERAGE / STATS / STD summary rows)
    return df[pd.to_numeric(df["Iteration"], errors="coerce").notna()].copy()


frames = []
for pga, lam, fname in RUNS:
    if not os.path.exists(fname):
        print(f"[skip] missing: {fname}")
        continue
    rows = load_iter_rows(fname)
    rows["PGA"] = pga
    rows["Risk"] = NEUTRAL if lam == 0.0 else AVERSE
    rows["Investment"] = pd.to_numeric(rows["Investment Cost ($)"], errors="coerce")
    frames.append(rows[["PGA", "Risk", "Investment"]].dropna())

if not frames:
    raise SystemExit("No result files found — check filenames in RUNS.")

data = pd.concat(frames, ignore_index=True)

# ── plot ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(7.5, 5))

order = ["0.23 g", "0.30 g", "0.40 g"]
hue_order = [NEUTRAL, AVERSE]
palette = {NEUTRAL: "#957555", AVERSE: "#A30101"}

sns.boxplot(
    data=data, x="PGA", y="Investment", hue="Risk",
    order=order, hue_order=hue_order, palette=palette,
    width=0.6, linewidth=1.1, fliersize=0, ax=ax,
)
sns.stripplot(
    data=data, x="PGA", y="Investment", hue="Risk",
    order=order, hue_order=hue_order, dodge=True,
    size=4, color="#1A252F", alpha=0.55, jitter=0.15, ax=ax,
)

# de-duplicate legend (boxplot and stripplot each add the two hue entries)
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles[:2], 
    labels[:2], 
    title="", 
    loc="upper center",          # Top of the legend box anchors to the position
    bbox_to_anchor=(0.5, -0.15), # Pushed below the x-axis
    ncol=2,                      # Aligns them side-by-side
    frameon=False                # Often looks cleaner without the box border at the bottom
)

ax.set_xlabel("Peak Ground Acceleration (PGA)", fontsize=12, labelpad=8)
ax.set_ylabel("DG Investment Cost (USD)", fontsize=12, labelpad=8)
ax.set_title(
    "Risk-Averse vs. Risk-Neutral DG Investment Across Hazard Levels\n"
    "(N = 30 scenarios, R = 30 replications)",
    fontsize=11, pad=12,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))

plt.tight_layout()
out_path = "investment_pga_lambda_swarm.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {out_path}")
