import inspect
import math
import os

import seaborn as sns
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb

matplotlib.use('TkAgg')

results_dir = 'out'
# Load data from CSV files
df_brute_force = pd.read_csv(os.path.join(results_dir, 'results_brute_force.csv'), sep=';')
df_brute_force_counter = pd.read_csv(os.path.join(results_dir, 'results_brute_force_counter.csv'), sep=';')
df_clever = pd.read_csv(os.path.join(results_dir, 'results_clever.csv'), sep=';')
df_greedy_v1 = pd.read_csv(os.path.join(results_dir, 'results_greedy_v1.csv'), sep=';')
df_greedy_v1_counter = pd.read_csv(os.path.join(results_dir, 'results_greedy_v1_counter.csv'), sep=';')
df_greedy_v2 = pd.read_csv(os.path.join(results_dir, 'results_greedy_v2.csv'), sep=';')


def get_stats(df):
    unique_k_values = sorted(df['k'].unique())
    for k in unique_k_values:
        sub_df = df[df['k'] == k]
        print(f'k = {k}')
        print(sub_df['elapsed_time'].describe())

        # what do you call a table with mean, median, std, etc.?

        print(sub_df['elapsed_time'].median())
        print(sub_df['elapsed_time'].mean())
        print(sub_df['elapsed_time'].std())


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
    plt.xlabel('n')
    plt.ylabel('elapsed time (s)')

    plt.xlim(0, 400)
    plt.ylim(0, 3600)

    plt.tight_layout()
    plt.show()


def visualize_elapsed_default(df, algorithm, out=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    unique_k_values = df['k'].unique()
    colors = sns.color_palette("tab10", n_colors=len(unique_k_values) + 1)

    for i, k in enumerate(unique_k_values):
        ax = axes[i]
        ax.set_title(f'k = {k}')

        for edge_percentage, color in zip(df['edge_percentage'].unique(), colors):
            sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
            ax.plot(sub_df['nodes'], np.log2(sub_df['elapsed_time']), label=f'$\\varepsilon_p = {edge_percentage}$',
                    marker='o', color=color)

        ax.set_xlabel('n')
        ax.set_ylabel('log2(elapsed time (s))')

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


def visualize_elapsed_individual(df, algorithm, window_size=1, log=False, out=None):
    unique_k_values = df['k'].unique()

    for i, k in enumerate(unique_k_values):
        fig, ax = plt.subplots(figsize=(6, 5))

        sorted_edge_percentages = sorted(df['edge_percentage'].unique())

        colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

        for edge_percentage, color in zip(sorted_edge_percentages, colors):
            sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
            smoothed_elapsed_time = sub_df['elapsed_time'].rolling(window=window_size).mean()
            epsilon = 0.0001
            if log:
                smoothed_elapsed_time = np.log2(smoothed_elapsed_time + epsilon)
            ax.plot(sub_df['nodes'], smoothed_elapsed_time,
                    label=f'$\\varepsilon_p = {edge_percentage}$', marker='o', color=color)

        ax.set_xlabel('n')

        if log:
            ax.set_ylabel('log2(elapsed time (s))')
        else:
            ax.set_ylabel('elapsed time (s)')
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


def compare_with_formal_proof(df, algorithm, formal_lambda, k=0.5, log=False, out=None, formalf=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    n = df['nodes'].max()
    x_values = np.array([i for i in range(1, n + 1)])

    num_args = len(inspect.signature(formal_lambda).parameters)

    sorted_edge_percentages = sorted(df['edge_percentage'].unique())

    colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

    for edge_percentage, color in zip(sorted_edge_percentages, colors):
        sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
        elapsed_time = sub_df['operations_count']
        if log:
            elapsed_time = np.log2(elapsed_time)
            # label \varepsilon_p latex
        ax.plot(sub_df['nodes'], elapsed_time, label=f'$\\varepsilon_p = {edge_percentage}$',
                marker='o', color=color)

    if num_args == 2:
        y_values = np.array([formal_lambda(i, k) for i in x_values])
        if log:
            y_values = np.log2(y_values)
        ax.plot(x_values, y_values, label=f"{formalf if formalf else 'Formal Proof'}",
                marker='x', color='black')
    elif num_args == 1:
        y_values = np.array([formal_lambda(i) for i in x_values])
        if log:
            y_values = np.log2(y_values)
        ax.plot(x_values, y_values, marker='x', label=f"{formalf if formalf else 'Formal Proof'}", color='black')

    ax.set_xlabel('n')
    if log:
        ax.set_ylabel('log2(iterations)')
    else:
        ax.set_ylabel('iterations')
    ax.legend(loc='upper left')
    ax.grid()

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        filename = f'{algorithm}_{k}_cf' + ('_log' if log else '')
        file_path = os.path.join(out, f'{filename}.png')
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


def forecast_elapsed_time(df, formal_lambda, k=0.5, p0=None, maxfev=10000, log=False, out=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    p0 = [1] * (len(inspect.signature(formal_lambda).parameters) - 1) if p0 is None else p0

    sorted_edge_percentages = sorted(df['edge_percentage'].unique())

    colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

    for edge_percentage, color in zip(sorted_edge_percentages, colors):
        sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
        x_values = sub_df['nodes']
        y_values = sub_df['elapsed_time']
        popt = curve_fit(formal_lambda, x_values, y_values, p0=p0, maxfev=maxfev)
        print(popt[0])
        forecast_x_values = np.array([i for i in range(4, 1000)])
        forecast_y_values = np.array([formal_lambda(i, *popt[0]) for i in forecast_x_values])
        r2 = 1 - np.sum((y_values - formal_lambda(x_values, *popt[0])) ** 2) / np.sum((y_values - np.mean(y_values)) ** 2)
        print(f"R2 of {df['algorithm'].unique()[0]} with k = {k} and edge percentage = {edge_percentage}: {r2}")
        if log:
            forecast_y_values = np.log2(forecast_y_values)
        ax.plot(forecast_x_values, forecast_y_values, label=f'$\\varepsilon_p = {edge_percentage}$',
                color=color)

    ax.set_xlabel('n')
    if log:
        ax.set_ylabel('log2(elapsed time (s))')
    else:
        ax.set_ylabel('elapsed time (s)')
    ax.legend(loc='upper left')
    ax.grid()

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        filename = f'{df["algorithm"].unique()[0]}_{k}_forecast' + ('_log' if log else '')
        file_path = os.path.join(out, f'{filename}.png')
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

def solutions_tested(df, k=0.5, log=False, out=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    sorted_edge_percentages = sorted(df['edge_percentage'].unique())

    colors = sns.color_palette("tab10", n_colors=len(sorted_edge_percentages) + 1)

    for edge_percentage, color in zip(sorted_edge_percentages, colors):
        sub_df = df[(df['k'] == k) & (df['edge_percentage'] == edge_percentage)]
        elapsed_time = sub_df['operations_count']
        if log:
            elapsed_time = np.log2(elapsed_time)
        ax.plot(sub_df['nodes'], elapsed_time, label=f'$\\varepsilon_p = {edge_percentage}$',
                marker='o', color=color)

    ax.set_xlabel('n')

    if log:
        ax.set_ylabel('log2(solutions tested)')
    else:
        ax.set_ylabel('solutions tested')

    ax.legend(loc='upper left')
    ax.grid()

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        filename = f'{df["algorithm"].unique()[0]}_{k}_solutions_tested' + ('_log' if log else '')
        file_path = os.path.join(out, f'{filename}.png')
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_confusion(df, out=None):
    actual = df_clever

    confusion_stats = []

    true_positives = 0
    false_positives = 0

    true_negatives = 0
    false_negatives = 0

    # loop through all the rows and check result column for each row
    print(df['algorithm'].unique()[0])
    # print result of each first row

    for index, row in actual.iterrows():
        actual_result = row['result']
        k = row['k']
        n = row['nodes']
        edge_percentage = row['edge_percentage']
        predicted_result = df.iloc[index]['result']
        if actual_result:
            if not predicted_result:
                confusion_stats.append({'Nodes': n, 'k': k, 'Edge Percentage': edge_percentage, 'Count': 1})
                false_negatives += 1
            else:
                true_positives += 1
        else:
            if predicted_result:
                false_positives += 1
            else:
                true_negatives += 1

    print(f'True positives: {true_positives}')
    print(f'False positives: {false_positives}')
    print(f'True negatives: {true_negatives}')
    print(f'False negatives: {false_negatives}')

    confusion_stats = pd.DataFrame(confusion_stats)
    # Describe the DataFrame for summary statistics
    print('\nSummary Statistics:')
    k_counts = confusion_stats.groupby('k')['Count']
    # k with the most false negatives
    print('\nk with the most false negatives:')


    print(k_counts.describe())

    n_counts = confusion_stats.groupby('Nodes')['Count']
    # node with the most false negatives
    print('\nNode with the most false negatives:')

    print(n_counts.describe())

    edge_percentage_counts = confusion_stats.groupby('Edge Percentage')['Count']
    # edge percentage with the most false negatives
    print('\nEdge percentage with the most false negatives:')

    print(edge_percentage_counts.describe())


    plt.figure()
    sns.heatmap([[true_positives, false_positives], [false_negatives, true_negatives]], annot=True, fmt='d',
                cmap='Blues', cbar=False)
    plt.xlabel('predicted')
    plt.ylabel('actual')

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        file_path = os.path.join(out, f'confusion_matrix_{df["algorithm"].unique()[0]}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    plt.show()

    return true_positives, false_positives, true_negatives, false_negatives


def compare_confusion(df1, df2, algorithm1=None, algorithm2=None, out=None):
    true_positives1, false_positives1, true_negatives1, false_negatives1 = calculate_confusion(df1)
    true_positives2, false_positives2, true_negatives2, false_negatives2 = calculate_confusion(df2)

    fig, ax = plt.subplots(figsize=(10, 5), sharex=True, sharey=True, ncols=2)

    sns.heatmap([[true_positives1, false_positives1], [false_negatives1, true_negatives1]], annot=True, fmt='d',
                cmap='Blues', cbar=False, ax=ax[0])
    plt.subplot(1, 2, 1)

    title = f'{algorithm1 if algorithm1 else df1["algorithm"].unique()[0]}'
    plt.title(title, pad=15)

    plt.subplot(1, 2, 2)
    sns.heatmap([[true_positives2, false_positives2], [false_negatives2, true_negatives2]], annot=True, fmt='d',
                cmap='Blues', cbar=False, ax=ax[1])

    title = f'{algorithm2 if algorithm2 else df2["algorithm"].unique()[0]}'
    plt.title(title, pad=15)

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        file_path = os.path.join(out, f'confusion_matrix_{df1["algorithm"].unique()[0]}_{df2["algorithm"].unique()[0]}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    fig.text(0.5, 0.04, 'predicted', ha='center')
    fig.text(0.04, 0.5, 'actual', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.15, left=0.095, right=0.9, bottom=0.15)
    plt.show()


def main():
    # test(400, out='data')
    # k = 0.75
    # compare_with_formal_proof(df_brute_force_counter, 'Brute Force', lambda n, k: math.comb(n, int(k * n)) * (k ** 2), k=k, log=True, out='data', formalf=r"$\binom{n}{%0.3f n} \cdot (%0.3f n)^2$" % (k, k))
    # compare_with_formal_proof(df_greedy_v1_counter, 'Greedy V1', lambda n: n ** 2, k=k, log=True, out='data', formalf=r"$n^2$")

    # visualize_elapsed_individual(df_brute_force_counter, 'Brute Force', window_size=1, log=True, out='data')
    # visualize_elapsed_individual(df_clever, 'Clever', window_size=1, log=True, out='data')
    # visualize_elapsed_individual(df_greedy_v1, 'Greedy V1', window_size=1, log=True, out='data')
    # visualize_elapsed_individual(df_greedy_v2, 'Greedy V2', window_size=1, log=True, out='data')

    # calculate_confusion(df_greedy_v1, out='data')
    # calculate_confusion(df_greedy_v2, out='data')
    # solutions_tested(df_brute_force_counter, log=True, out='data')
    # solutions_tested(df_greedy_v1_counter, log=True, out='data')
    # compare_confusion(df_greedy_v1, df_greedy_v2, algorithm1='Greedy Heuristics', algorithm2='Improved Greedy Heuristics', out='data')

    k = 0.5
    # brute force has (n choose k*n) * k^2 lambda
    forecast_elapsed_time(df_brute_force, lambda n, a, b: a * comb(n, k * n) * (k ** 2) + b, k=k, log=True, out='data')
    forecast_elapsed_time(df_greedy_v1, lambda n, a, b: a * n ** 2 + b, k=k, log=True, out='data')

    pass


if __name__ == "__main__":
    main()
