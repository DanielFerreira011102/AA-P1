import math
from itertools import combinations

from graph_generator import GraphGenerator


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


g = GraphGenerator.load_graph('graphs/graph_79_0.75.gml')
print(g.number_of_edges())

n = 5
e = 5

for k in range(5):
    if e > ((n - k) * (n - k - 1)) / 2 + k:
        continue
    print(k)

print("aqui")
for k in range(5):
    if e < n - k + 1:
        print(k)
