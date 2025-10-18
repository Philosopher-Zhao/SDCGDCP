import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from config import Config
from functional import batched_shortest_path_distance

def load_drug_data():
    drug_data = pd.read_csv(Config.drug_data_path)
    target_data = pd.read_csv(Config.target_data_path)
    graph_feature_data = pd.read_csv(Config.drug_graph_feature_path)
    drug_features = {}
    for _, row in drug_data.iterrows():
        name = row['drugName']
        target = target_data[target_data['drugName'] == name].values[0][1:]
        target = target.astype(np.float32)
        ecfp = drug_data[drug_data['drugName'] == name].values[0][1:]
        ecfp = ecfp.astype(np.float32)
        graph_feat = graph_feature_data[graph_feature_data['drug_name'] == name].values[0][1:]
        graph_feat = graph_feat.astype(np.float32)

        drug_features[name] = {
            'ecfp': torch.tensor(ecfp, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'graph': torch.tensor(graph_feat, dtype=torch.float32)
        }
    return drug_features

def create_graph():
    drug_features = load_drug_data()
    interactions = pd.read_csv(Config.interaction_data_path)
    cell_data = pd.read_csv(Config.cell_data_path)

    node_features = []
    edge_index = []
    edge_attr = []
    edge_labels = []
    edge_info = []
    node_map = {}
    node_id = 0

    for _, row in interactions.iterrows():
        drug1 = row['Drug1'].strip('"')
        drug2 = row['Drug2'].strip('"')
        cell_line = row['Cell line']
        values = cell_data[cell_data.iloc[:, 0] == cell_line].values[0][1:]
        cell_feat = torch.tensor(values.astype(np.float32), dtype=torch.float32)

        drug1_cell_key = f"{drug1}-{cell_line}"
        drug2_cell_key = f"{drug2}-{cell_line}"

        if drug1_cell_key not in node_map:
            node_map[drug1_cell_key] = node_id
            node_feat1 = torch.cat([drug_features[drug1]['ecfp'], drug_features[drug1]['target'], drug_features[drug1]['graph'], cell_feat])
            node_features.append(node_feat1)
            node_id += 1

        if drug2_cell_key not in node_map:
            node_map[drug2_cell_key] = node_id
            node_feat2 = torch.cat([drug_features[drug2]['ecfp'], drug_features[drug2]['target'], drug_features[drug2]['graph'], cell_feat])
            node_features.append(node_feat2)
            node_id += 1

        edge_index.append([node_map[drug1_cell_key], node_map[drug2_cell_key]])
        edge_attr.append(cell_feat)
        edge_labels.append(1 if row['classification'] == 'synergy' else 0)
        edge_info.append((drug1, drug2, cell_line))

    data = Data(
        x=torch.stack(node_features),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.stack(edge_attr),
        y=torch.tensor(edge_labels, dtype=torch.float32),
        edge_info=edge_info  
    )

    indices = range(len(data.y))
    train_idx, test_idx = train_test_split(indices, test_size=0.4, random_state=42)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    data.train_mask = torch.zeros(len(data.y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(data.y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(data.y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    def _precompute_paths(edge_mask):
        filtered_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, edge_mask],
            edge_attr=data.edge_attr[edge_mask],
            y=data.y[edge_mask]
        )
        node_paths, edge_paths = batched_shortest_path_distance(filtered_data)
        return edge_paths

    data.train_paths = _precompute_paths(data.train_mask)
    data.val_paths = _precompute_paths(data.val_mask)
    data.test_paths = _precompute_paths(data.test_mask)

    def _split_edges(indices, batch_size, global_paths):
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            edge_index_sub = data.edge_index[:, batch_indices]
            edge_attr_sub = data.edge_attr[batch_indices]
            y_sub = data.y[batch_indices]
            batch_edge_info = [data.edge_info[idx] for idx in batch_indices]

            src_nodes = edge_index_sub[0].tolist()
            dst_nodes = edge_index_sub[1].tolist()

            batch_paths_dict = {}
            for src, dst in zip(src_nodes, dst_nodes):
                if src not in batch_paths_dict:
                    batch_paths_dict[src] = {}
                if src in global_paths and dst in global_paths[src]:
                    batch_paths_dict[src][dst] = global_paths[src][dst]
                else:
                    batch_paths_dict[src][dst] = []

            sub_data = Data(
                x=data.x,
                edge_index=edge_index_sub,
                edge_attr=edge_attr_sub,
                y=y_sub,
                paths=batch_paths_dict,
                edge_info=batch_edge_info
            )
            batches.append(sub_data)
        return batches

    train_batches = _split_edges(train_idx, Config.batch_size, data.train_paths)
    val_batches = _split_edges(val_idx, Config.batch_size, data.val_paths)
    test_batches = _split_edges(test_idx, Config.batch_size, data.test_paths)

    train_loader = DataLoader(train_batches, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_batches, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_batches, shuffle=False, num_workers=0, pin_memory=True)


    return train_loader, val_loader, test_loader, data, data, data
