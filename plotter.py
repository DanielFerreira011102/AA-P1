import os
from math import log

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

results_dir = 'out'
# Load data from CSV files
df_brute_force = pd.read_csv(os.path.join(results_dir, 'results_brute_force.csv'), sep=';')
df_clever = pd.read_csv(os.path.join(results_dir, 'results_clever.csv'), sep=';')
df_greedy_v1 = pd.read_csv(os.path.join(results_dir, 'results_greedy_v1.csv'), sep=';')
df_greedy_v2 = pd.read_csv(os.path.join(results_dir, 'results_greedy_v2.csv'), sep=';')

def visualize_elapsed_time_by_nodes():
    df_brute_force_grouped = df_brute_force.groupby('nodes')['elapsed_time'].mean()
    df_clever_grouped = df_clever.groupby('nodes')['elapsed_time'].mean()
    df_greedy_v1_grouped = df_greedy_v1.groupby('nodes')['elapsed_time'].mean()
    df_greedy_v2_grouped = df_greedy_v2.groupby('nodes')['elapsed_time'].mean()

    plt.figure()
    plt.plot(df_brute_force_grouped, label='Brute Force')
    plt.plot(df_clever_grouped, label='Clever')
    plt.plot(df_greedy_v1_grouped, label='Greedy v1')
    plt.plot(df_greedy_v2_grouped, label='Greedy v2')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Elapsed time (s)')
    plt.title('Elapsed time by number of nodes')

    plt.xlim(0, 400)
    plt.ylim(0, 3600)

    plt.tight_layout()
    plt.show()

def visualize_elapsed_default(df, algorithm, out=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    unique_k_values = df['k'].unique()
    colors = ['#F1CE63', '#E15759', '#4E79A7', '#8CD17D', 'm', 'y', 'k']

    for i, k in enumerate(unique_k_values):
        ax = axes[i]
        ax.set_title(f'k = {k}')

        for edge_percentage, color in zip(df['edge_percentage'].unique(), colors):
            sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
            ax.plot(sub_df['nodes'], np.log2(sub_df['elapsed_time']), label=f'Edge Percentage = {edge_percentage}',
                    marker='o', color=color)

        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Log2(Elapsed Time (s))')

        ax.legend(loc='upper left')

    plt.suptitle(f'{algorithm} - Elapsed Time vs. Number of Nodes for Different k values and Edge Percentages',
                 fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.86, bottom=0.14, wspace=0.14)
    plt.show()

    if out is None:
        return

    os.makedirs(out, exist_ok=True)
    file_path = os.path.join(out, f'{df["algorithm"].unique()[0]}_etvsnnddkvaep_log.png')
    fig.savefig(file_path, dpi=300, bbox_inches='tight')

def main():
    visualize_elapsed_default(df_brute_force, 'Brute Force', out='data')
    visualize_elapsed_default(df_clever, 'Clever', out='data')
    visualize_elapsed_default(df_greedy_v1, 'Greedy v1', out='data')
    visualize_elapsed_default(df_greedy_v2, 'Greedy v2', out='data')

if __name__ == "__main__":
    main()

