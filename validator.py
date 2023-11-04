import os

from graph_generator import GraphGenerator
import pandas as pd

from independent_set_solver import Solution, Algorithm
from loggingdf import Logger, LogLevel

logger = Logger(level=LogLevel.DEBUG, icon=True)


def extract_nodes(solution_str):
    # Split the solution string by ':' to separate the individual node coordinates
    if solution_str == '-1':
        return None

    nodes = solution_str.split(':')
    node_set = set()

    for node in nodes:
        s = node[1:-1]
        x, y = map(int, s.replace(' ', '').split(','))
        node_set.add((int(x), int(y)))

    return node_set


def validate_solutions(graph_dir, solutions_file, visualize=False):
    solutions = pd.read_csv(solutions_file, sep=';')

    invalid_solutions = []

    for index, row in solutions.iterrows():
        logger.info('Validating solution for graph {}.'.format(row['name']))

        solution_str = row['solution']
        name = row['name']

        nodes = extract_nodes(solution_str)
        graph = GraphGenerator.load_graph(os.path.join(graph_dir, name))


        solution = Solution(nodes, bool(row['result']), float(row['elapsed_time']), graph, Algorithm(row['algorithm']),
                            int(row['independent_k']), float(row['k']))

        if visualize:
            solution.visualize()

        if not solution.validate():
            logger.error('Solution for graph {} is invalid.'.format(row['name']))
            invalid_solutions.append(solution)
        else:
            logger.success('Solution for graph {} is valid.'.format(row['name']))

    return invalid_solutions


def main():
    invalid_solutions = validate_solutions('graphs', 'out/results_clever.csv')
    print(f"Number of invalid solutions = {len(invalid_solutions)}")
    print(f"Invalid solutions = {invalid_solutions}")


if __name__ == '__main__':
    main()
