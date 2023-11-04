import os

from graph_generator import GraphGenerator
import pandas as pd


def validate_solutions(graph_dir, solutions_file):
    """
    Validates the solutions in the given directory.
    """

    solutions = pd.read_csv(solutions_file, sep=';')

    for solution in solutions.itertuples():
        print('Validating solution {}'.format(solution))
        #graph = GraphGenerator.load_graph(os.path.join(graph_dir, solution.name))


def main():
    """
    Main function.
    """

    validate_solutions('graphs', 'out/results_clever.csv')

if __name__ == '__main__':
    main()