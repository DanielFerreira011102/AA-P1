import glob
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

logger = Logger(level=LogLevel.DEBUG, )

matplotlib.use('TkAgg')


class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        logger.warning("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_DISPLAY_REQUIRED)

    def uninhibit(self):
        import ctypes
        logger.warning("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


class Algorithm(MultipleValueEnum):
    BRUTE_FORCE = "brute_force", "Brute Force", "BruteForce", "bf", "BF", "exhaustive_search", "Exhaustive Search", "ExhaustiveSearch", "es", "ES"
    CLEVER = "clever", "Clever", "cl", "CL"
    GREEDY_V1 = "greedy_v1", "Greedy V1", "GreedyV1", "gv1", "GV1", "greedy_heuristics_v1", "Greedy Heuristics V1", "GreedyHeuristicsV1", "ghv1", "GHV1"
    GREEDY_V2 = "greedy_v2", "Greedy V2", "GreedyV2", "gv2", "GV2", "greedy_heuristics_v2", "Greedy Heuristics V2", "GreedyHeuristicsV2", "ghv2", "GHV2"


class Solution:
    def __init__(self, solution: set, result: bool, operations_count: int, solutions_count: int, elapsed_time: float, graph: Graph, algorithm: Algorithm,
                 independent_k: int,
                 k: float):
        self.solution = solution
        self.elapsed_time = elapsed_time
        self.operations_count = operations_count
        self.solutions_count = solutions_count
        self.graph = graph
        self.algorithm = algorithm
        self.k = k
        self.independent_k = independent_k
        self.result = result

    def validate(self):
        if self.solution is None or not self.result:
            return True

        if len(self.solution) != self.independent_k:
            print(
                f"Number of nodes in solution ({len(self.solution)}) does not match independent_k ({self.independent_k})")
            return False

        for node in self.solution:
            if node not in self.graph.nodes():
                print(f"Node {node} is not in the graph")
                return False
            for neighbor in self.graph.neighbors(node):
                if neighbor in self.solution:
                    print(f"Node {node} and {neighbor} are neighbors")
                    return False
        return True

    def visualize(self):
        plt.figure(figsize=(10, 5))
        delta = 10
        limit = (0 - delta, 100 + delta)

        plt.title(f"Independent set for k={self.k} using {self.algorithm.value} = {self.result}")

        plt.axis('on')
        plt.grid(True)

        pos = {node: node for node in self.graph.nodes()}
        nx.draw_networkx_nodes(self.graph, pos, node_size=100,
                               node_color=["tab:red" if self.solution and node in self.solution else "tab:blue" for node
                                           in self.graph.nodes()])
        nx.draw_networkx_edges(self.graph, pos, width=1)

        plt.xlim(*limit)
        plt.ylim(*limit)
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.tight_layout()
        plt.show()

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

    def filter(self, condition_func: Callable, as_list: bool = False):
        filtered_records = filter(condition_func, self.solutions)
        return list(filtered_records) if as_list else filtered_records


class IndependentSetSolver:

    def __init__(self, k: Union[Tuple[float, ...], float] = (0.125, 0.25, 0.5, 0.75)):
        self.k = k

    def solve(self, graphs: Union[Graph, List[Graph]], algorithm: Algorithm = Algorithm.BRUTE_FORCE,
              k: Union[Tuple[float, ...], float] = None, visualize: bool = False, save: bool = False,
              directory: str = 'out', filename: str = 'results.csv'):
        if k is None:
            k = self.k

        if isinstance(k, float):
            k = (k,)

        match algorithm:
            case Algorithm.GREEDY_V1:
                func = self._solve_greedy_heuristics_v1
            case Algorithm.GREEDY_V2:
                func = self._solve_greedy_heuristics_v2
            case Algorithm.CLEVER:
                func = self._solve_clever
            case Algorithm.BRUTE_FORCE | _:
                func = self._solve_brute_force

        logger.info(f"Using algorithm={algorithm}")

        if isinstance(graphs, Graph):
            graphs = [graphs]

        solutions = self._solve_all(func, graphs, algorithm, k, visualize, save, directory, filename)

        return solutions

    def _solve_all(self, func: Callable, graphs: List[Graph], algorithm: Algorithm, k: Tuple[float, ...],
                   visualize: bool = False, save: bool = False, directory: str = 'out', filename: str = 'results.csv'):
        logger.info(f"Solving all graphs for k={k}")

        if save:
            os.makedirs(directory, exist_ok=True)
            with open(f'{directory}/{filename}', 'w') as file:
                file.write("name;nodes;edges;edge_percentage;algorithm;k;independent_k;result;solution;elapsed_time;operations_count;solutions_count\n")

        # solutions = SolutionSet([
        #     self._solve_single(func, graph, algorithm, kn, visualize, save, directory, filename) for graph in graphs for
        #     kn in k
        # ])

        solutions = SolutionSet([])

        for graph in graphs:
            for kn in k:
                self._solve_single(func, graph, algorithm, kn, visualize, save, directory, filename)

        return solutions

    def _solve_single(self, func: Callable, graph: Graph, algorithm: Algorithm, k: float, visualize: bool = False,
                      save: bool = False, directory: str = 'out', filename: str = 'results.csv'):
        logger.info(f"Solving {graph} for k={k}")
        independent_k = self._get_k_percentage_nodes(graph, k)
        logger.info(f"Number of independent vertices={independent_k} for k={k}")
        start_time = time.time()
        result = func(graph.copy(), independent_k)
        elapsed_time = time.time() - start_time
        if isinstance(result, tuple):
            if len(result) == 2:
                op_count, sol_count = None, None
                result, solution = result[0], result[1]
            else:
                result, (op_count, sol_count), solution = result

        else:
            op_count, sol_count = None, None
            solution = None

        logger.info(
            f"No independent set found" if result is None else f"Solution {result} found in {elapsed_time} seconds")
        solution = Solution(solution, result, op_count, sol_count, elapsed_time, graph, algorithm, independent_k, k)

        if visualize:
            logger.info(f"Visualizing solution for {solution}")
            solution.visualize()

        if save:
            self._write_solution_to_file(solution, directory, filename)

        return solution

    def _write_solution_to_file(self, solution, directory, filename):
        with open(f'{directory}/{filename}', 'a') as file:
            file.write(f"{solution.graph.name};" +
                       f"{solution.graph.number_of_nodes()};" +
                       f"{solution.graph.number_of_edges()};" +
                       f"{solution.graph.edge_percentage()};" +
                       f"{solution.algorithm.value};" +
                       f"{solution.k};" +
                       f"{solution.independent_k};" +
                       f"{solution.result};" +
                       f"{':'.join([str(item) for item in solution.solution]) if solution.solution and solution.result else -1};" +
                       f"{solution.elapsed_time};" +
                       f"{solution.operations_count if solution.operations_count is not None  else -1};" +
                       f"{solution.solutions_count if solution.solutions_count is not None else -1}\n")

    def _solve_brute_force(self, graph: Graph, k: int):
        operations_count = 0
        solutions_count = 0

        if k == 0:
            return True, (operations_count, solutions_count), set()
        if graph.number_of_nodes() == 0:
            return False, (operations_count, solutions_count), None
        if k > graph.number_of_nodes():
            return False, (operations_count, solutions_count), None

        nodes = list(graph.nodes())

        for subset in combinations(nodes, k):
            operations_count += 1
            is_independent_set = True
            for u, v in combinations(subset, 2):
                operations_count += 1
                if graph.has_edge(u, v):
                    is_independent_set = False
                    solutions_count += 1
                    break

            if is_independent_set:
                return True, (operations_count, solutions_count), set(subset)

        return False, (operations_count, solutions_count), None

    def _solve_clever(self, graph: Graph, k: int):
        if k == 0:
            return True, set()
        if graph.number_of_nodes() == 0:
            return False, None
        if k > graph.number_of_nodes():
            return False, None
        if v := next((node for node, d in graph.degree() if d <= 1), None):
            result, solution = self._solve_clever(graph.removed_adjacency_subgraph(v), k - 1)
            return (result, solution) if not result else (result, solution | {v})
        if v := next((node for node, d in graph.degree() if d >= 3), None):
            result_inc, solution_inc = self._solve_clever(graph.removed_adjacency_subgraph(v), k - 1)
            return result_inc, solution_inc | {v} if result_inc else self._solve_clever(graph.removed_node_subgraph(v),
                                                                                        k)
        components = list(nx.connected_components(graph))
        if sum(len(c) // 2 for c in components) >= k:
            selected_nodes = set()
            for component in components:
                while k > 0 and len(component) > 0:
                    node = component.pop()
                    neighbors = graph.neighbors(node)
                    component.difference_update(neighbors)
                    selected_nodes.add(node)
                    k -= 1
            return True, selected_nodes
        return False, None

    def _solve_greedy_heuristics_v1(self, graph: Graph, k: int):
        # could have sorted the nodes by degree and then picked the first k nodes
        # but that does not speed up the algorithm
        # the algorithm is still O(n^2) because of the remove_adjacency function


        if k == 0:
            return True, set()
        if graph.number_of_nodes() == 0:
            return False, None
        if k > graph.number_of_nodes():
            return False, None

        mis = set()

        while graph.number_of_nodes() > 0 and len(mis) < k:
            print(graph.number_of_nodes())

            node = min(graph.nodes(), key=lambda x: graph.degree(x))
            mis.add(node)

            graph.remove_adjacency(node)

        if len(mis) == k:
            return True, mis
        return False, None

    def _solve_greedy_heuristics_v2(self, graph: Graph, k: int):
        if k == 0:
            return True
        if graph.number_of_nodes() == 0:
            return False
        if k > graph.number_of_nodes():
            return False
        mis = set()
        while graph.number_of_nodes() > 0 and len(mis) < k:
            node = graph.pick_best_vertex()
            mis.add(node)
            graph.remove_adjacency(node)
        if len(mis) == k:
            return True, mis
        return False, None

    def _get_k_percentage_nodes(self, graph: Graph, k: float):
        return int(graph.number_of_nodes() * k)


def main():
    student_number = 102885
    num_vertices = tuple(range(80, 401))
    edge_percentages = (0.125, 0.25, 0.5, 0.75)

    graphs = GraphGenerator.load_graphs('graphs')

    k = (0.125, 0.25, 0.5, 0.75)
    solver = IndependentSetSolver(k)

    osSleep = None

    if os.name == 'nt':
        osSleep = WindowsInhibitor()
        osSleep.inhibit()

    solutions = solver.solve(graphs, Algorithm.GREEDY_V1, save=True, filename='results_greedy_v1.csv')
    solutions = solver.solve(graphs, Algorithm.GREEDY_V2, save=True, filename='results_greedy_v2.csv')

    if osSleep:
        osSleep.uninhibit()

    for solution in solutions:
        solution.visualize()

    quick_solutions = solutions.filter(lambda x: x.elapsed_time < 0.00001)

    print("Quick solutions:")
    for solution in quick_solutions:
        print(solution)


if __name__ == "__main__":
    main()
