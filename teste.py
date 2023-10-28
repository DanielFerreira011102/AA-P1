from itertools import combinations

import networkx as nx
import random

from matplotlib import pyplot as plt

from graph_generator import Graph
from independent_set_solver import IndependentSetSolver, Algorithm




# Example usage:
N = 10  # Number of vertices
P = 15  # Number of edges
k = 4   # Minimum size of the independent set
graph, solution, ret = generate_graph_with_independent_set(N, P, k)


def _solve_brute_force(graph: Graph, k: int):
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

if ret:
    print("Found a valid graph with independent set:")
    print(f"Graph: {graph}")
    print(f"Independent set: {solution}")
    csolution = _solve_brute_force(graph, k)
    print(f"Solution: {csolution}")
    plt.figure()
    nx.draw(
        graph,
        pos=nx.spring_layout(graph),
        with_labels=True,
        node_color=["tab:red" if n in solution else "tab:blue" for n in graph],
    )
    plt.tight_layout()
    plt.show()

def memoize(func):
    cache = {}

    def wrapper(cls, *args):
        vertex, n = args
        prev_degree = (vertex, n - 1)

        if prev_degree not in cache:
            degree_sum, open_nodes, closed_nodes = func(cls, vertex, n)
            cache[args] = (open_nodes, closed_nodes)
            return degree_sum

        prev_open_nodes, prev_closed_nodes = cache[prev_degree]
        degree_sum, open_nodes, closed_nodes = func(cls, vertex, 2, open_nodes=prev_open_nodes,
                                                    closed_nodes=prev_closed_nodes)
        cache[args] = (open_nodes, closed_nodes)
        return degree_sum

    return wrapper
