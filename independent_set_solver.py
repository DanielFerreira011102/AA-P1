import os
import time
from itertools import combinations
from typing import Tuple, Callable, Union, List

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

from graph_generator import GraphGenerator, Graph
from loggingdf import Logger, LogLevel
from multiple_value_enum import MultipleValueEnum

logger = Logger(level=LogLevel.DEBUG)

matplotlib.use('TkAgg')


class Algorithm(MultipleValueEnum):
    BRUTE_FORCE = "Brute Force", "BruteForce", "brute_force", "bf", "BF"
    GREEDY = "Greedy", "greedy", "GR", "gr"

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper


class Solution:
    def __init__(self, solution: set, elapsed_time: float, graph: Graph, algorithm: Algorithm, independent_k: int,
                 k: float):
        self.solution = solution
        self.elapsed_time = elapsed_time
        self.graph = graph
        self.algorithm = algorithm
        self.k = k
        self.independent_k = independent_k
        self.result = solution is not None

    def visualize(self):
        # create figure
        fig = plt.figure(figsize=(10, 5))
        delta = 10
        limit = (0 - delta, 100 + delta)
        fig.suptitle(
            f"""
        For {self.graph}
        Independent set for k={self.k} using {self.algorithm.value} = {self.result}
        """, fontsize=16)

        ax1 = fig.add_subplot(1, 2, 1)
        self._draw_graph(self.graph, ax1)
        ax1.axis('on')
        ax1.set_title("Graph")
        ax1.grid(True)
        ax1.set_xlim(*limit)
        ax1.set_ylim(*limit)
        ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        ax2 = fig.add_subplot(1, 2, 2)
        subgraph = self.graph.subgraph(self.solution if self.solution else set())
        self._draw_graph(subgraph, ax2)
        ax2.axis('on')
        ax2.set_title("Independent set")
        ax2.grid(True)
        ax2.set_xlim(*limit)
        ax2.set_ylim(*limit)
        ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.tight_layout()
        plt.show()

    def _draw_graph(self, graph: Graph, ax: plt.Axes):
        pos = {node: node for node in graph.nodes()}

        nx.draw(graph, pos=pos, node_size=300, node_color='lightblue', ax=ax, width=2)
        nx.draw_networkx_labels(graph, pos=pos, font_size=10)

    def __bool__(self):
        return self.result

    def __str__(self):
        return f"Solution(solution={self.solution}, elapsed_time={self.elapsed_time}, graph={self.graph}, algorithm={self.algorithm}, independent_k={self.independent_k}, k={self.k}, result={self.result})"

    def __repr__(self):
        return str(self)


class SolutionSet:
    def __init__(self, solutions: List[Solution] = None):
        self.solutions = solutions

    def __bool__(self):
        return any(self.solutions)

    def __str__(self):
        return f"SolutionSet(solutions={self.solutions})"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.solutions)

    def __getitem__(self, item: int):
        return self.solutions[item]

    def __len__(self):
        return len(self.solutions)

    def __add__(self, other: 'SolutionSet'):
        return self.solutions + other.solutions

    def add(self, solution: Solution):
        self.solutions.append(solution)

    def save(self, directory: str, filename: str):
        os.makedirs(directory, exist_ok=True)
        results_file_path = os.path.join(directory, filename)
        with open(results_file_path, 'w') as f:
            f.write(f"name,nodes,edges,edge_percentage,algorithm,k,independent_k,result,solution,elapsed_time\n")
            for solution in self:
                solution.graph.save(os.path.join(directory, 'graphs'))
                f.write(f"{solution.graph.name}," +
                        f"{solution.graph.number_of_nodes()}," +
                        f"{solution.graph.number_of_edges()}," +
                        f"{solution.graph.edge_percentage}," +
                        f"{solution.algorithm.value}," +
                        f"{solution.k}," +
                        f"{solution.independent_k}," +
                        f"{solution.result}," +
                        f"{':'.join([str(item) for item in solution.solution]) if solution.result else -1}," +
                        f"{solution.elapsed_time}\n")

    def get_by_elasped_time(self, condition_func: Callable):
        filtered_records = filter(lambda solution: condition_func(solution.elapsed_time), self.solutions)
        return list(filtered_records)

    def get_solutions(self, graph: Graph = None, algorithm: Algorithm = None, independent_k: int = None,
                      k: float = None, result: bool = None):
        filtered_solutions = []
        for solution in self.solutions:
            if graph is not None and solution.graph != graph:
                continue
            if independent_k is not None and solution.independent_k != independent_k:
                continue
            if k is not None and solution.k != k:
                continue
            if algorithm is not None and solution.algorithm != algorithm:
                continue
            if result is not None and solution.result != result:
                continue
            filtered_solutions.append(solution)
        return filtered_solutions


class IndependentSetSolver:

    def __init__(self, k: Union[Tuple[float, ...], float] = (0.125, 0.25, 0.5, 0.75)):
        self.k = k

    def solve(self, graphs: Union[Graph, List[Graph]], algorithm: Algorithm = Algorithm.BRUTE_FORCE,
              k: Union[Tuple[float, ...], float] = None, visualize: bool = False, save: bool = False,
              directory: str = 'output', filename: str = 'results2.csv'):
        if k is None:
            k = self.k

        if isinstance(k, float):
            k = (k,)

        match algorithm:
            case Algorithm.GREEDY:
                func = self._solve_greedy_heuristics
            case Algorithm.BRUTE_FORCE | _:
                func = self._solve_brute_force

        logger.info(f"Using algorithm={algorithm}")

        if isinstance(graphs, Graph):
            graphs = [graphs]
        solutions = self._solve_for_all(func, graphs, algorithm, k)

        if save:
            logger.info(f"Saving solutions to {directory}/{filename}")
            solutions.save(directory, filename)

        if visualize:
            for solution in solutions:
                logger.info(f"Visualizing {solution}")
                solution.visualize()

        return solutions

    def _solve_for_all(self, func: Callable, graphs: List[Graph], algorithm: Algorithm, k: Tuple[float, ...]):
        logger.info(f"Solving all graphs for k={k}")
        solutions = SolutionSet([])
        for graph in graphs:
            for kn in k:
                logger.info(f"Solving {graph} for k={kn}")
                independent_k = self._get_k_percentage_nodes(graph, kn)
                logger.info(f"Number of independent vertices={independent_k} for k={kn}")
                result, elapsed_time = func(graph.copy(), independent_k)
                logger.info(f"No independent set found" if result is None else f"Solution {result} found in {elapsed_time} seconds")
                solution = Solution(result, elapsed_time, graph, algorithm, independent_k, kn)
                solutions.add(solution)
        return solutions

    @timeit
    def _solve_brute_force(self, graph: Graph, k: int):
        nodes = list(graph.nodes())
        for subset in combinations(nodes, k):
            is_independent_set = True
            for u, v in combinations(subset, 2):
                if graph.has_edge(u, v):
                    is_independent_set = False
                    break
            if is_independent_set:
                return set(subset)
        return None

    @timeit
    def _solve_greedy_heuristics(self, graph: Graph, k: int):
        mis = set()
        while graph.number_of_nodes() > 0:
            node = graph.pick_best_vertex()
            mis.add(node)
            graph.remove_nodes_from(list(graph.neighbors(node)))
            graph.remove_node(node)
            if len(mis) == k:
                return mis

        return None

    def _get_k_percentage_nodes(self, graph: Graph, k: float):
        return int(graph.number_of_nodes() * k)


def main():
    student_number = 102885
    num_vertices = tuple(range(4, 50))
    edge_percentages = (0.125, 0.25, 0.5, 0.75)

    generator = GraphGenerator(student_number, num_vertices, edge_percentages)
    generator.generate_graphs()

    k = (0.125, 0.25, 0.5, 0.75)
    solver = IndependentSetSolver(k)

    solutions = solver.solve(generator.output_graphs, Algorithm.BRUTE_FORCE, save=True, visualize=True)

    quick_solutions = solutions.get_by_elasped_time(lambda elapsed_time: elapsed_time < 0.00001)


if __name__ == "__main__":
    main()
