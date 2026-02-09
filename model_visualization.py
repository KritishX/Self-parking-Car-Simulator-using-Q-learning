import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_comparison():
    """
    Reads the comparison results from CSV and generates visualization plots.
    Saves plots to the 'visualization' directory.
    """
    input_file = "model_comparison_results.csv"
    output_dir = "visualization"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run model_comparison.py first.")
        return

    # Load data
    df = pd.read_csv(input_file)
    print("Loaded Data:")
    print(df)

    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Success Rate Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Success Rate (%)", data=df, palette="viridis")
    plt.title("Success Rate by Model", fontsize=16)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0, 105)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
    plt.close()
    print(f"Saved {output_dir}/success_rate_comparison.png")

    # 2. Average Reward Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Avg Reward", data=df, palette="magma")
    plt.title("Average Reward by Model", fontsize=16)
    plt.ylabel("Average Reward", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    
    plt.savefig(os.path.join(output_dir, "avg_reward_comparison.png"))
    plt.close()
    print(f"Saved {output_dir}/avg_reward_comparison.png")

    # 3. Collision vs Timeout Rate Stacked Bar
    # Transform data for stacked plot
    stacked_data = df[["Model", "Collision Rate (%)", "Timeout Rate (%)"]].set_index("Model")
    ax = stacked_data.plot(kind='bar', stacked=True, color=['#d62728', '#ff7f0e'], figsize=(10, 6))
    plt.title("Failure Modes: Collision vs Timeout", fontsize=16)
    plt.ylabel("Rate (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Failure Type")
    
    plt.savefig(os.path.join(output_dir, "failure_modes_comparison.png"))
    plt.close()
    print(f"Saved {output_dir}/failure_modes_comparison.png")

    # 4. Average Steps (Efficiency)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Avg Steps", data=df, palette="Blues_d")
    plt.title("Efficiency: Average Steps per Episode", fontsize=16)
    plt.ylabel("Steps (Lower is Better)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"))
    plt.close()
    print(f"Saved {output_dir}/efficiency_comparison.png")

if __name__ == "__main__":
    visualize_comparison()
