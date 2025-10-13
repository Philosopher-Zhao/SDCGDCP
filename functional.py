from __future__ import annotations
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import subgraph
from collections import deque


def bfs_source_to_all(G, source, max_depth=100):
    visited = {source: [source]}
    edge_paths = {source: []}
    queue = deque([(source, 0)])
    edges = {(u, v): idx for idx, (u, v) in enumerate(G.edges())}
    num_edges = len(edges)

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + [neighbor]
                u, v = sorted([node, neighbor])
                edge_idx = edges.get((u, v), -1)
                edge_paths[neighbor] = edge_paths[node] + [edge_idx]
                queue.append((neighbor, depth + 1))
    return visited, edge_paths


def batched_shortest_path_distance(data, edge_mask=None, max_depth=100):
    if isinstance(data, Batch):
        graphs = []
        shift = 0
        for sub_data in data.to_data_list():
            if edge_mask is not None:
                sub_edge_mask = edge_mask[shift:shift + sub_data.num_edges]
                sub_edge_index, sub_edge_attr = subgraph(
                    sub_edge_mask,
                    sub_data.edge_index,
                    edge_attr=sub_data.edge_attr,
                    relabel_nodes=False
                )
                sub_data = Data(
                    x=sub_data.x,
                    edge_index=sub_edge_index,
                    edge_attr=sub_edge_attr,
                    y=sub_data.y[sub_edge_mask] if sub_data.y is not None else None
                )
            G = to_networkx(sub_data, to_undirected=True)
            num_edges = sub_data.edge_index.size(1)
            paths = {}
            for node in G.nodes():
                node_paths, edge_paths = bfs_source_to_all(G, node, max_depth)
                paths[node] = (node_paths, edge_paths)
            graphs.append(paths)
            shift += G.number_of_nodes()

        node_paths = {}
        edge_paths = {}
        for graph in graphs:
            node_paths.update({k: v[0] for k, v in graph.items()})
            edge_paths.update({k: v[1] for k, v in graph.items()})
        return node_paths, edge_paths
    else:
        G = to_networkx(data, to_undirected=True)
        num_edges = data.edge_index.size(1)
        node_paths = {}
        edge_paths = {}
        for node in G.nodes():
            n_paths, e_paths = bfs_source_to_all(G, node, max_depth)
            node_paths[node] = n_paths
            edge_paths[node] = e_paths
        return node_paths, edge_paths




