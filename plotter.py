import inspect
import math
import os
import seaborn as sns
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
df_greedy_v1_counter = pd.read_csv(os.path.join(results_dir, 'results_greedy_v1_counter.csv'), sep=';')
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


def visualize_elapsed_individual(df, algorithm, window_size=3, log=False, out=None):
    unique_k_values = df['k'].unique()
    colors = ['#F1CE63', '#E15759', '#4E79A7', '#8CD17D', 'm', 'y', 'k']

    for i, k in enumerate(unique_k_values):
        fig, ax = plt.subplots(figsize=(6, 5))

        sorted_edge_percentages = sorted(df['edge_percentage'].unique())

        for edge_percentage, color in zip(sorted_edge_percentages, colors):
            sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
            smoothed_elapsed_time = sub_df['elapsed_time'].rolling(window=window_size).mean()
            epsilon = 1e-3
            if log:
                smoothed_elapsed_time = np.log2(smoothed_elapsed_time + epsilon)
            ax.plot(sub_df['nodes'], smoothed_elapsed_time,
                    label=f'Edge Percentage = {edge_percentage}', marker='o', color=color)

        ax.set_xlabel('Number of Nodes')

        if log:
            ax.set_ylabel('Log2(Elapsed Time (s))')
        else:
            ax.set_ylabel('Elapsed Time (s)')
        ax.legend(loc='upper left')
        ax.grid()

        plt.tight_layout()

        if out:
            os.makedirs(out, exist_ok=True)
            filename = f'{algorithm}_k{k}_etvsnnddkvaep'
            if log:
                filename += '_log'
            file_path = os.path.join(out, f'{filename}.png')
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()


def test(n, out=None):
    k_values = [0.125, 0.25, 0.5, 0.75]

    # Set a seaborn color palette for academic-style plots
    colors = sns.color_palette("tab10", n_colors=len(k_values) + 1)

    x_values = np.array([i for i in range(n + 1)])

    for i, k in enumerate(k_values):
        y_values = np.array([math.comb(n, int(k * n)) * (k ** 2) for n in x_values])
        y_values = np.log2(y_values)
        sns.regplot(x=x_values, y=y_values, label=r'$\binom{n}{%0.3f n} \cdot (%0.3f n)^2$' % (k, k), color=colors[i],
                    scatter=False)

    y_138 = np.array([1.38 ** i for i in range(n + 1)])
    y_138 = np.log2(y_138)
    sns.regplot(x=x_values, y=y_138, label=r'$1.38^n$', color="black", scatter=False)

    plt.grid()
    plt.xlabel('n')
    plt.ylabel('log2(iterations)')
    plt.legend()

    if out:
        os.makedirs(out, exist_ok=True)
        file_path = os.path.join(out, f'iterations_vs_n.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def compare_with_formal_proof(df, algorithm, formal_lambda, k=0.5, out=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ['#F1CE63', '#E15759', '#4E79A7', '#8CD17D', 'm', 'y', 'k']

    n = df['nodes'].max()
    x_values = np.array([i for i in range(1, n + 1)])

    num_args = len(inspect.signature(formal_lambda).parameters)

    ax.set_title(f'k = {k}')
    for edge_percentage, color in zip(df['edge_percentage'].unique(), colors):
        sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
        if num_args == 2:
            y_values = np.array([formal_lambda(i, k) for i in x_values])
            ax.plot(x_values, np.log2(y_values), label=f'Formal Proof Edge Percentage = {edge_percentage}',
                    marker='x', color=color)
        ax.plot(sub_df['nodes'], np.log2(sub_df['operations_count']), label=f'Edge Percentage = {edge_percentage}',
                marker='o', color=color)

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Log2(Elapsed Time (s))')
    ax.legend(loc='upper left')
    ax.grid()

    y_values = np.array([np.log2(formal_lambda(i)) for i in x_values])
    ax.plot(x_values, y_values, label='Formal Proof', color='black')

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        filename = f'{algorithm}_cf'
        file_path = os.path.join(out, f'{filename}.png')
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # test(400, out='data')
    compare_with_formal_proof(df_brute_force_counter, 'Brute Force', lambda n, k: math.comb(n, int(k * n)) * (k ** 2),
                              k=0.5, out='data')
    compare_with_formal_proof(df_greedy_v1_counter, 'Greedy V1', lambda n: n ** 2, k=0.5, out='data')


if __name__ == "__main__":
    main()
