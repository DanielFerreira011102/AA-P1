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


def compare_solutions(solutions_file1, solutions_file2):
    # check the results for every line
    solutions1 = pd.read_csv(solutions_file1, sep=';')
    solutions2 = pd.read_csv(solutions_file2, sep=';')

    invalid_solutions = []

    merged_solutions = solutions1.merge(solutions2, on=['name', 'k'], how='inner', suffixes=('_1', '_2'))

    for index, row in merged_solutions.iterrows():
        name = row['name']
        k = row['k']
        result1, result2 = row['result_1'], row['result_2']

        logger.debug(f"Comparing solution for graph {name} and k = {k}")

        logger.debug(f"row = {row.to_dict()}")

        logger.info(f"result1 = {result1}, result2 = {result2}")

        if result1 != result2:
            logger.error(f"Solution for graph {name} is different")
            invalid_solutions.append((name, k, "different"))
        else:
            logger.success(f"Solution for graph {name} is the same and k = {k}")

    return invalid_solutions, len(solutions1.index), len(solutions1[solutions1['result'] == True].index)


def validate_solutions(graph_dir, solutions_file, visualize=False):
    solutions = pd.read_csv(solutions_file, sep=';')

    invalid_solutions = []

    for index, row in solutions.iterrows():
        logger.info('Validating solution for graph {}.'.format(row['name']))

        solution_str = row['solution']
        name = row['name']
        result = row['result']

        if not result:
            continue

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


def correct_file(solutions_file):
    with open(solutions_file, 'r') as infile, open(f'{solutions_file}_modified', 'w') as outfile:
        for line in infile:
            # Split the line into fields using a comma as the delimiter
            fields = line.strip().split(',', maxsplit=8)

            modified_fields = fields[:-1] + fields[-1].rsplit(',', maxsplit=1)

            new_line = ';'.join(modified_fields) + '\n'

            outfile.write(new_line)


def correct_file_2(solutions_file):
    with open(solutions_file, 'r') as infile, open(f'{solutions_file}_modified', 'w') as outfile:
        for line in infile:
            # Split the line into fields using a comma as the delimiter
            fields = line.strip().split(',', maxsplit=11)

            modified_fields = fields[:-3] + fields[-3].rsplit(',', maxsplit=1) + fields[-2:]

            new_line = ';'.join(modified_fields) + '\n'

            outfile.write(new_line)

def fix_zero_values(solutions_file):
    df = pd.read_csv(solutions_file, sep=';')

    column1 = 'operations_count'
    column2 = 'solutions_count'

    df[column1] = df[column1].replace(-1, 0)
    df[column2] = df[column2].replace(-1, 0)

    df.to_csv(solutions_file + ".tmp", index=False, sep=';')


def main():
    # invalid_solutions, total_lines, total_positive_lines = compare_solutions('out/results_clever.csv', 'out/results_greedy_v1.csv')
    # print(f"Number of positive lines = {total_positive_lines}")
    # print(f"Number of invalid solutions = {len(invalid_solutions)}")
    # print(f"False negative rate = {len(invalid_solutions) / total_positive_lines * 100}%")
    # print(f"Percentage of invalid solutions = {len(invalid_solutions) / total_lines * 100}%")
    # print(f"Invalid solutions = {invalid_solutions}")

    # invalid_solutions = validate_solutions('graphs', 'out/results_greedu-3.csv')
    # print(f"Number of invalid solutions = {len(invalid_solutions)}")
    # print(f"Invalid solutions = {invalid_solutions}")

    # fix_zero_values('out/results_greedy_v1_counter.csv')

    # correct_file('out/results_greedy_v1_counter.csv.tmp')
    pass


if __name__ == '__main__':
    main()
