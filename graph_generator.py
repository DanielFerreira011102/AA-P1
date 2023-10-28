import os.path
from typing import List, Tuple, Union

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import random
from loggingdf import Logger, LogLevel

logger = Logger(level=LogLevel.DEBUG)

matplotlib.use('TkAgg')


class Graph(nx.Graph):
    def __init__(self, num_vertices: int = None, edge_percentage: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_vertices = num_vertices
        self.edge_percentage = edge_percentage
        self.name = f"graph_{num_vertices}_{edge_percentage}.gml"

    def __str__(self):
        return f"Graph(name={self.name}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()}, edge_percentage={self.edge_percentage})"

    def __repr__(self):
        return str(self)

    def visualize(self):
        plt.figure()
        plt.title(self.name)
        nx.draw(self, with_labels=True, node_size=200, node_color='lightblue', font_size=8)
        plt.show()

    def pick_best_vertex(self):
        degree = 1

        nodes = self.nodes()

        while True:

            node_degrees = {node: self.nth_degree(node, degree) for node in nodes}

            if degree % 2 == 0:
                max_degree = max(node_degrees.values())
                max_nodes = [node for node in nodes if node_degrees[node] == max_degree]

                if len(max_nodes) == 1:
                    return max_nodes[0]

                if len(max_nodes) == len(nodes):
                    return max_nodes[0]

                nodes = max_nodes

            else:
                min_degree = min(node_degrees.values())
                min_nodes = [node for node in nodes if node_degrees[node] == min_degree]

                if len(min_nodes) == 1:
                    return min_nodes[0]

                if len(min_nodes) == len(nodes):
                    return min_nodes[0]

                nodes = min_nodes

            degree += 1

    def nth_degree(self, vertex: Tuple[int, int], n: int):
        open_nodes = {vertex}
        closed_nodes = set()

        for i in range(1, n):
            new_open_nodes = set()
            for node in open_nodes:
                new_open_nodes.update(self.neighbors(node))
                closed_nodes.add(node)
            open_nodes = new_open_nodes - closed_nodes

        return sum(self.degree(node) for node in open_nodes)

    def save(self, directory: str = os.getcwd()):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.name)
        nx.write_gml(self, path)


class GraphGenerator:
    def __init__(self, seed: int, num_vertices: Union[Tuple[int, ...], int] = (),
                 edge_percentages: Union[Tuple[float, ...], float] = (),
                 output_folder: str = 'out', coordinate_range: Tuple[int, int] = (1, 100)):
        self.seed = seed
        self.num_vertices = (num_vertices,) if isinstance(num_vertices, int) else num_vertices
        self.edge_percentages = (edge_percentages,) if isinstance(edge_percentages, float) else edge_percentages
        self.coordinate_range = coordinate_range
        self.output_folder = output_folder
        self.output_graphs = []

    def generate_graphs(self):
        logger.info("Generating graphs")

        for num_vertex in self.num_vertices:
            for edge_percentage in self.edge_percentages:
                self._generate_random_graph(num_vertex, edge_percentage)

        logger.success(f"Generated {len(self.output_graphs)} graphs")

    def _generate_random_graph(self, n: int, edge_percentage: float):
        logger.info(f"Generating {n}-vertex graph with {edge_percentage * 100}% edges")

        random.seed(self.seed)
        G = Graph(n, edge_percentage)

        vertices = self._generate_random_coordinates(n)
        number_of_edges = int(self._generate_maximum_number_of_edges(n) * edge_percentage)

        G.add_edges_from(self._generate_edges(vertices, number_of_edges))

        G.add_nodes_from(vertices)

        self.output_graphs.append(G)

        logger.success(f"Generated {n}-vertex graph with {edge_percentage * 100}% edges")

    def _generate_independent_set_solution(self, n: int, edge_percentage: float, k: float):
        pass

    def visualize_graphs(self, grid: bool = False):
        self._visualize_graph_grid() if grid else self._visualize_graph_all()

    def _visualize_graph_all(self):
        logger.info("Visualizing all stored graphs")

        for G in self.output_graphs:
            G.visualize()

    def _generate_edges(self, vertices: List[Tuple[int, int]], number_of_edges: int, loops: bool = False,
                        duplicate_edges: bool = False, cycles: bool = True):
        edges = []

        while len(edges) < number_of_edges:
            random_vertex1, random_vertex2 = random.sample(vertices, 2)

            if random_vertex1 == random_vertex2 and not loops:
                continue

            edge = (random_vertex1, random_vertex2)

            if not duplicate_edges and edge in edges:
                continue

            if not cycles and self._forms_cycle(edges, edge):
                continue

            edges.append(edge)

        return edges

    def _forms_cycle(self, edges: List[Tuple[int, int]], edge: Tuple[int, int]):
        G = Graph()
        G.add_edges_from(edges)
        G.add_edge(*edge)
        return nx.cycle_basis(G) != []

    def _generate_maximum_number_of_edges(self, n: int):
        return int(n * (n - 1) / 2)

    def _generate_random_coordinates(self, n: int):
        return [(random.randint(*self.coordinate_range), random.randint(*self.coordinate_range)) for _ in range(n)]

    def save_graphs(self):
        logger.info("Saving all stored graphs")

        self._create_output_dir()

        for G in self.output_graphs:
            self._save_graph_to_file(G)

        logger.success(f"Saved {len(self.output_graphs)} graphs")

    def _save_graph_to_file(self, G: Graph):
        logger.info(f"Saving {G.num_vertices}-vertex graph with {G.edge_percentage * 100}% edges")

        G.save(self.output_folder)

        logger.success(
            f"Saved {G.num_vertices}-vertex graph with {G.edge_percentage * 100}% edges to {self.output_folder} with name {G.name}")

    def _visualize_graph_grid(self):
        logger.info("Visualizing graphs in grid")

        num_rows = len(self.num_vertices)
        num_cols = len(self.edge_percentages)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

        for k, G in enumerate(self.output_graphs):
            i = k // num_cols
            j = k % num_cols
            nx.draw(G, with_labels=True, ax=axes[i, j], node_size=200, node_color='lightblue', font_size=8)
            axes[i, j].set_title(G.name)

        plt.tight_layout()
        plt.show()

    def _create_output_dir(self):
        os.makedirs(self.output_folder, exist_ok=True)

    def __str__(self):
        return f"GraphGenerator(seed={self.seed}, num_vertices={self.num_vertices}, edge_percentages={self.edge_percentages}, output_folder={self.output_folder}, coordinate_range={self.coordinate_range})"

    def __repr__(self):
        return str(self)


def main():
    student_number = 102885
    num_vertices = (20, 50, 100)
    edge_percentages = (0.125, 0.25, 0.5, 0.75)

    generator = GraphGenerator(student_number, num_vertices, edge_percentages)
    generator.generate_graphs()
    generator.visualize_graphs()


if __name__ == "__main__":
    main()
