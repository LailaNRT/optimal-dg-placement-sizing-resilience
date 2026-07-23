"""
DG Placement Heatmap across 30 SAA iterations.

Produces two separate figures:
  1. Network topology heatmap (no title, clean gray network skeleton).
  2. Side-by-side bar charts for selection frequency and average capacity.
"""

import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import opendssdirect as dss
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

NUM_ITERATIONS = 30
LABEL = "pga040N30L000Y15R010_rho00_sig50"
DSS_FILE = "ieee37.dss"

# ---------------------------------------------------------------------------
# 1. Aggregate DG data across all iterations
# ---------------------------------------------------------------------------
bus_freq      = defaultdict(int)    # how many iterations selected each bus
bus_cap_total = defaultdict(float)  # sum of capacity across iterations

for i in range(1, NUM_ITERATIONS + 1):
    fname = f"master_dg_placements_{LABEL}_{i}.json"
    if not os.path.exists(fname):
        print(f"  (skipped {fname})")
        continue
    with open(fname) as f:
        data = json.load(f)

    sizes = data.get("dg_placement", {}).get("sizes_kva", {})
    for bus_str, cap in sizes.items():
        bus_freq[bus_str] += 1
        bus_cap_total[bus_str] += cap

if not bus_freq:
    raise RuntimeError(f"No master_dg_placements_{LABEL}_*.json files found.")

all_buses      = sorted(bus_freq.keys(), key=lambda b: int(b))
frequencies    = np.array([bus_freq[b]      for b in all_buses])
avg_capacities = np.array([bus_cap_total[b] / bus_freq[b] for b in all_buses])

print(f"Buses selected at least once across {NUM_ITERATIONS} iterations:")
for b, f, c in zip(all_buses, frequencies, avg_capacities):
    print(f"  Bus {b:>4s} — {f:>2d}/{NUM_ITERATIONS} iterations, avg capacity = {c:,.1f} kVA")

# ---------------------------------------------------------------------------
# 2. Load network topology
# ---------------------------------------------------------------------------
def load_bus_coords():
    coords = {}
    if os.path.exists("IEEE37_BusXY.csv"):
        with open("IEEE37_BusXY.csv", "r") as f:
            for parts in csv.reader(f):
                if len(parts) >= 3:
                    try:
                        coords[parts[0].strip().lower()] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
    return coords

def map_bus(name):
    """Merges internal substation nodes into a single clean label to prevent text overlaps."""
    n = name.split('.')[0].lower().strip()
    if n in ['sourcebus', '799', '799r']:
        return '799/source bus'
    return n

print("\nCompiling IEEE 37-Bus System topology...")
dss.Command('Clear')
dss.Command(f'Compile "{DSS_FILE}"')

pos = load_bus_coords()
if not pos:
    raise RuntimeError("IEEE37_BusXY.csv not found or empty.")

if '799r' in pos:
    pos['799/source bus'] = pos['799r']
elif '799' in pos:
    pos['799/source bus'] = pos['799']
elif 'sourcebus' in pos:
    pos['799/source bus'] = pos['sourcebus']

G = nx.Graph()
hardwired_edges, closed_switches, open_switches = [], [], []

dss.Lines.First()
for _ in range(dss.Lines.Count()):
    name = dss.Lines.Name().lower()
    b1 = map_bus(dss.Lines.Bus1())
    b2 = map_bus(dss.Lines.Bus2())

    if b1 == b2:
        dss.Lines.Next()
        continue

    if b1 in pos and b2 in pos:
        G.add_edge(b1, b2)
        is_switch = name.startswith('sw')
        dss.Circuit.SetActiveElement(f"Line.{name}")
        is_open = dss.CktElement.IsOpen(1, 0) or dss.CktElement.IsOpen(2, 0)

        if is_switch:
            (open_switches if is_open else closed_switches).append((b1, b2))
        else:
            hardwired_edges.append((b1, b2))

    dss.Lines.Next()

dss.Transformers.First()
for _ in range(dss.Transformers.Count()):
    buses = dss.CktElement.BusNames()
    if len(buses) > 1:
        b1 = map_bus(buses[0])
        b2 = map_bus(buses[1])
        if b1 == b2:
            dss.Transformers.Next()
            continue
        if b1 in pos and b2 in pos:
            G.add_edge(b1, b2)
            hardwired_edges.append((b1, b2))
    dss.Transformers.Next()

substation_nodes = ['799/source bus']

# ---------------------------------------------------------------------------
# 3. Figure 1 — Network Topology Heatmap (No Title, Clean Skeleton)
# ---------------------------------------------------------------------------
C = {
    'lines':        "#B4B4B4",    
    'no_selection': "#8B8B8B",  
    'substation':   '#0F172A',    
}

fig1, ax1_net = plt.subplots(figsize=(12, 10)) 
fig1.patch.set_facecolor('#FFFFFF')
ax1_net.set_facecolor('#FFFFFF')

# Unified gray skeleton lines, fully opaque
nx.draw_networkx_edges(G, pos, edge_color=C['lines'], width=1.0, ax=ax1_net, alpha=1.0)

selected_lower = [b.lower() for b in all_buses]
non_selected = [n for n in G.nodes() if n not in selected_lower and n not in substation_nodes]
sub_exists = [n for n in substation_nodes if n in G.nodes()]

# Inactive nodes
nx.draw_networkx_nodes(G, pos, nodelist=non_selected,
                       node_color=C['no_selection'], node_size=50, ax=ax1_net,
                       linewidths=0.6, edgecolors='#E5E7EB')

cmap    = mcolors.LinearSegmentedColormap.from_list(
    "slate_blue", ["#DBEAFE", "#3B82F6", "#0F172A"]
)
norm    = mcolors.Normalize(vmin=1, vmax=NUM_ITERATIONS)
max_cap = avg_capacities.max()

min_size, max_size = 60, 280  

for b, freq, avg_cap in zip(all_buses, frequencies, avg_capacities):
    node_key = b.lower()
    if node_key not in pos or node_key not in G.nodes():
        continue
    
    size  = min_size + (avg_cap / max_cap) * (max_size - min_size)
    color = cmap(norm(freq))
    
    nx.draw_networkx_nodes(G, pos, ax=ax1_net, nodelist=[node_key],
                           node_size=size, node_color=[color],
                           linewidths=1.0, edgecolors='#7F1D1D') 

# Substation node
nx.draw_networkx_nodes(G, pos, nodelist=sub_exists,
                       node_shape='s', node_color=C['substation'],
                       node_size=150, ax=ax1_net)

# Clean annotations
for node, (x, y) in pos.items():
    if node not in G.nodes() or node in substation_nodes:
        continue
    if node in selected_lower:
        idx = selected_lower.index(node)
        b, freq, avg_cap = all_buses[idx], frequencies[idx], avg_capacities[idx]
        label = f"{b}\n{freq}/{NUM_ITERATIONS}\n{avg_cap:.0f} kVA"
    else:
        continue  

    ax1_net.annotate(label, xy=(x, y), xytext=(0, 12), textcoords="offset points",
                     fontsize=7.5, ha='center', va='bottom', fontweight='semibold',
                     color='#0F172A',
                     bbox=dict(facecolor='#F8FAFC', alpha=0.85, edgecolor='#CBD5E1',
                               boxstyle='round,pad=0.25', lw=0.8))

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar1 = fig1.colorbar(sm, ax=ax1_net, fraction=0.025, pad=0.03)
cbar1.set_label(f"Selection frequency (out of {NUM_ITERATIONS})", fontsize=10, labelpad=10)
cbar1.outline.set_linewidth(0.5)

ax1_net.axis("off")
plt.tight_layout()
fig1.savefig("dg_heatmap_networkpga040.png", dpi=300, bbox_inches="tight", facecolor=fig1.get_facecolor())
print("Saved: dg_heatmap_networkpga040.png")


# ---------------------------------------------------------------------------
# 4. Figure 2 — Bar Charts (Side-by-Side with Custom Styling)
# ---------------------------------------------------------------------------
fig2, (ax_freq, ax_cap) = plt.subplots(1, 2, figsize=(15, 6))
fig2.patch.set_facecolor('#FAFAFA')  # Premium off-white backdrop
ax_freq.set_facecolor('#FFFFFF')
ax_cap.set_facecolor('#FFFFFF')

x = np.arange(len(all_buses))
bar_colors = [cmap(norm(f)) for f in frequencies]

# --- Left Subplot: Selection Frequency ---
bars1 = ax_freq.bar(x, frequencies, color=bar_colors, edgecolor="#475569", linewidth=0.8, width=0.65, alpha=0.9)
ax_freq.set_title("Bus Selection Frequency", fontsize=11, fontweight="semibold", pad=12, color='#334155')
ax_freq.set_ylabel("Selection Frequency (Iterations)", fontsize=10, color='#334155')
ax_freq.set_xlabel("Bus Number", fontsize=10, color='#334155')
ax_freq.set_ylim(0, NUM_ITERATIONS + 2)
ax_freq.axhline(NUM_ITERATIONS, color="#EF4444", linestyle="--", linewidth=1.2, label=f"Max ({NUM_ITERATIONS})")
ax_freq.legend(loc="upper right", frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)
ax_freq.grid(True, axis='y', linestyle=':', alpha=0.5, color='#94A3B8')
ax_freq.set_axisbelow(True)
ax_freq.set_xticks(x)
ax_freq.set_xticklabels([str(b) for b in all_buses], rotation=45, ha="right", fontsize=9)

for bar, freq in zip(bars1, frequencies):
    ax_freq.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(freq), ha="center", va="bottom", fontsize=8.5, fontweight='bold', color='#1E293B')

# --- Right Subplot: Average Capacity ---
bars2 = ax_cap.bar(x, avg_capacities, color=bar_colors, edgecolor="#475569", linewidth=0.8, width=0.65, alpha=0.9)
ax_cap.set_title("Average Allocated Capacity", fontsize=11, fontweight="semibold", pad=12, color='#334155')
ax_cap.set_ylabel("Average Capacity (kVA)", fontsize=10, color='#334155')
ax_cap.set_xlabel("Bus Number", fontsize=10, color='#334155')
ax_cap.grid(True, axis='y', linestyle=':', alpha=0.5, color='#94A3B8')
ax_cap.set_axisbelow(True)
ax_cap.set_xticks(x)
ax_cap.set_xticklabels([str(b) for b in all_buses], rotation=45, ha="right", fontsize=9)

for bar, cap in zip(bars2, avg_capacities):
    ax_cap.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max_cap * 0.02),
                f"{cap:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight='bold', color='#1E293B')

# Modern spine styling (removing top/right borders)
for ax in [ax_freq, ax_cap]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
fig2.savefig("dg_heatmap_barspga040.png", dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
print("Saved: dg_heatmap_barspga040.png")

plt.show()
print("\nDone.")