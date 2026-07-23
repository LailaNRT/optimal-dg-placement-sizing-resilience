import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_json_distributions(file_path):
    # 1. Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    durations = []
    fault_counts = []

    # 2. Loop through the scenarios to extract the raw data
    for key, scenario in data.items():
        # Ignore the metadata and only look at the earthquake scenarios
        if key.startswith('Earthquake_Scenario'):
            durations.append(scenario['duration_hours'])
            fault_counts.append(len(scenario['faults']))

    # 3. Set up the plots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Duration Distribution
    sns.histplot(durations, kde=True, ax=axes[0], color='skyblue', bins=20)
    axes[0].set_title(f'Distribution of Outage Durations')
    axes[0].set_xlabel('Duration (Hours)')
    axes[0].set_ylabel('Frequency')

    # Plot 2: Fault Count Distribution
    sns.histplot(fault_counts, kde=True, ax=axes[1], color='salmon', bins=15)
    axes[1].set_title(f'Distribution of Broken Lines/Switches')
    axes[1].set_xlabel('Number of Faults per Scenario')
    axes[1].set_ylabel('Frequency')

    # 4. Save and display the graphs
    plt.tight_layout()
    plt.savefig("distributionlinehr.png", dpi=300, bbox_inches="tight")
    print("Saved: distributionlinehr.png")
    plt.show()

# You can run it on any of your files like this:
plot_json_distributions('saa_scenario_pga023N30L000Y15R010_rho00_sig50_1.json')