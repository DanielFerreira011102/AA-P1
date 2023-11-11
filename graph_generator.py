import os.path
import re
from typing import List, Tuple, Union
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import random
import math

from loggingdf import Logger, LogLevel

logger = Logger(level=LogLevel.DEBUG)

matplotlib.use('TkAgg')


def generate_max_spread_points(N, x_range, y_range):
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    points = [(random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))]

    while len(points) < N:
        max_min_distance = -1
        best_point = None

        for _ in range(1000):
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])
            min_distance = min(euclidean_distance((x, y), p) for p in points)

            if min_distance > max_min_distance and points not in points:
                max_min_distance = min_distance
                best_point = (x, y)

        if best_point is not None:
            points.append(best_point)

    return points


class Graph(nx.Graph):
    def __init__(self, num_vertices: int = None, edge_percentage: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph['num_vertices'] = num_vertices
        self.graph['edge_percentage'] = edge_percentage
        self.name = f"graph_{num_vertices}_{edge_percentage}.gml"

    @staticmethod
    def from_nx_graph(G: nx.Graph):
        this = Graph()
        this.graph = G.graph
        this.add_nodes_from(G.nodes(data=True))
        this.add_edges_from(G.edges(data=True))
        this.name = G.name
        return this

    def __str__(self):
        return f"Graph(name={self.name}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()}, edge_percentage={self.edge_percentage()})"

    def __repr__(self):
        return str(self)

    def visualize(self):
        plt.figure()
        plt.title(self.name)
        nx.draw(self, with_labels=True, node_size=200, node_color='lightblue', font_size=8)
        plt.show()

    def removed_node_subgraph(self, node):
        return self.subgraph(set(self.nodes()) - {node})

    def removed_adjacency_subgraph(self, node):
        return self.subgraph(set(self.nodes()) - set(self.neighbors(node)) - {node})

    def remove_adjacency(self, node):
        self.remove_nodes_from(list(self.neighbors(node)))
        self.remove_node(node)

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

    def number_of_vertices(self):
        return self.graph.get('num_vertices', None)

    def edge_percentage(self):
        return self.graph.get('edge_percentage', None)

    def save(self, directory: str = os.getcwd()):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.name)
        nx.write_gml(self, path)


class GraphGenerator:

    @staticmethod
    def generate_graphs(seed: int = 0, num_vertices: Union[Tuple[int, ...], int] = (),
                        edge_percentages: Union[Tuple[float, ...], float] = (),
                        coordinate_range: Tuple[int, int] = (1, 100)):
        logger.info("Generating graphs")

        if isinstance(num_vertices, int):
            num_vertices = (num_vertices,)
        if isinstance(edge_percentages, float):
            edge_percentages = (edge_percentages,)

        return [
            GraphGenerator.generate_graph(seed, num_vertex, edge_percentage, coordinate_range)
            for num_vertex in num_vertices for edge_percentage in edge_percentages
        ]

    @staticmethod
    def generate_graph(seed: int = 0, num_vertices: int = (), edge_percentage: float = (),
                       coordinate_range: Union[Tuple[int, int], List[Tuple[int, int]]] = (1, 100)):
        logger.info(f"Generating {num_vertices}-vertex graph with {edge_percentage * 100}% edges")

        random.seed(seed)
        G = Graph(num_vertices, edge_percentage)

        vertices = GraphGenerator.generate_random_coordinates(num_vertices, coordinate_range)
        G.add_nodes_from(vertices)

        number_of_edges = int(GraphGenerator.generate_maximum_number_of_edges(num_vertices) * edge_percentage)
        G.add_edges_from(GraphGenerator.generate_edges(vertices, number_of_edges))

        return G

    @staticmethod
    def visualize_graphs(graphs: List[Graph], grid: bool = False):
        GraphGenerator.visualize_graph_grid(graphs) if grid else GraphGenerator.visualize_graph_all(graphs)

    @staticmethod
    def visualize_graph_all(graphs: List[Graph]):
        logger.info("Visualizing all stored graphs")

        for G in graphs:
            G.visualize()

    @staticmethod
    def generate_edges(vertices: List[Tuple[int, int]], number_of_edges: int, loops: bool = False,
                       duplicate_edges: bool = False):
        edges = []

        while len(edges) < number_of_edges:
            vertex1, vertex2 = random.sample(vertices, 2)

            if not loops and vertex1 == vertex2:
                continue

            edge = (vertex1, vertex2)

            if not duplicate_edges and (edge in edges or edge[::-1] in edges):
                continue

            edges.append(edge)

        return edges

    @staticmethod
    def generate_maximum_number_of_edges(num_vertices: int):
        return int(num_vertices * (num_vertices - 1) / 2)

    @staticmethod
    def load_graphs(path: str = os.getcwd(), files: List[str] = None):
        if files is not None:
            logger.info(f"Loading graphs from files {files}")
            return [GraphGenerator.load_graph(os.path.join(path, file)) for file in files]

        logger.info(f"Loading graphs from {path}")

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\\d+)', s)]

        return [GraphGenerator.load_graph(os.path.join(path, filename)) for filename in sorted(os.listdir(path), key=natural_sort_key) if
                filename.endswith('.gml')]

    @staticmethod
    def load_graph(file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found")
        if not file.endswith('.gml'):
            raise ValueError(f"File {file} is not a .gml file")

        def destringize_to_tuple(s):
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
                x, y = map(int, s.split(","))
                return x, y
            return s

        G = nx.read_gml(file, destringizer=destringize_to_tuple)
        return Graph.from_nx_graph(G)

    @staticmethod
    def generate_random_coordinates(num_vertices: int, coordinate_range: Tuple[int, int] = (1, 100),
                                    coincident: bool = False, spread: bool = True):
        if isinstance(coordinate_range, tuple):
            coordinate_range_x = coordinate_range_y = coordinate_range
        else:
            coordinate_range_x, coordinate_range_y = coordinate_range

        if spread:
            return generate_max_spread_points(num_vertices, coordinate_range_x, coordinate_range_y)

        coordinates_array = []
        while len(coordinates_array) < num_vertices:
            coordinates = (random.randint(*coordinate_range_x), random.randint(*coordinate_range_y))
            if not coincident and coordinates in coordinates_array:
                continue
            coordinates_array.append(coordinates)
        return coordinates_array

    @staticmethod
    def save_graphs(graphs: List[Graph], directory: str = os.getcwd()):
        logger.info("Saving all stored graphs")

        for G in graphs:
            GraphGenerator.save_graph(G, directory)

    @staticmethod
    def save_graph(G: Graph, directory: str = os.getcwd()):
        logger.info(f"Saving {G.number_of_vertices()}-vertex graph with {G.edge_percentage() * 100}% edges")

        os.makedirs(directory, exist_ok=True)

        G.save(directory)

    @staticmethod
    def visualize_graph_grid(graphs: List[Graph], cols: int = 3):
        logger.info("Visualizing graphs in grid")

        num_cols = cols
        num_graphs = len(graphs)
        num_rows = (num_graphs + num_cols - 1) // num_cols

        figsize = (2.5 * num_cols, 7 * num_rows)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        limit = (-10, 110)

        for k in range(num_rows * num_cols):
            if k < num_graphs:
                i = k // num_cols
                j = k % num_cols

                axes[i, j].axis('on')
                axes[i, j].set_title(graphs[k].name)
                axes[i, j].grid(True)

                pos = {node: node for node in graphs[k].nodes()}

                nx.draw_networkx_nodes(graphs[k], pos, node_color='tab:blue', node_size=100, ax=axes[i, j])
                nx.draw_networkx_edges(graphs[k], pos, width=1, ax=axes[i, j])

                axes[i, j].set_xlim(*limit)
                axes[i, j].set_ylim(*limit)
                axes[i, j].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            else:
                axes.flat[k].axis('off')  # Hide the empty subplots

        plt.tight_layout()
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.4, wspace=0.1)
        plt.show()


def main():
    student_number = 102885
    num_vertices = (4, 5, 6)
    edge_percentages = (0.125, 0.25, 0.5, 0.75)

    graphs = GraphGenerator.generate_graphs(student_number, num_vertices, edge_percentages)


if __name__ == "__main__":
    main()
