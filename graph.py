from typing import Dict, Set
from itertools import product


def get_components(adjacency_dict: Dict[str, Set]) -> Dict[str, int]:
    """
    Enumerate the disconnected components in a graph
    :param adjacency_dict: the graph in the form of Dict[Set]
    :return: a dictionary mapping a node in each components to the size of that component
    """
    graph = dict(adjacency_dict)

    def delete_component(user: str):
        if user in graph:
            neighbors = graph[user]  # store
            del graph[user]
            for n in neighbors:
                delete_component(n)

    connected_components, last_size = dict(), len(graph)
    for node in list(graph.keys()):
        if node in graph:
            delete_component(node)
            connected_components[node] = last_size - len(graph)
            last_size = len(graph)

    return connected_components


def extract_component(graph: Dict[str, Set], node_from_component: str, visited=None) -> Dict[str, Set]:
    """
    basically DFS to grab a network
    :param graph: the base graph
    :param node_from_component: the node to conduct the traversal from
    :param visited: nodes that have already been visited
    :return: the sub-graph
    """
    if visited is None:
        visited = set()

    if node_from_component in visited:
        return dict()

    new_graph = dict()
    neighbors = new_graph[node_from_component] = graph[node_from_component]
    visited |= {node_from_component}
    for n in neighbors:
        new_graph.update(extract_component(graph, n, visited))

    return new_graph


def floyd_warshall(matrix):
    """
    Finding shortest distances between all pairs
    :param matrix: an adjacency matrix
    :return: the distance matrix
    """
    V = len(matrix)
    dist = list(map(lambda i: list(map(lambda j: j, i)), matrix))
    for k in range(V):
        for i, j in product(range(V), range(V)):
            dist[i][j] = min(
                dist[i][j],
                dist[i][k] + dist[k][j]
            )
    return dist
