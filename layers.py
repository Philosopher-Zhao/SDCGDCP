from typing import Tuple
import torch
from torch import nn
from torch_geometric.utils import degree
from utils import decrease_to_max_value
import torch.nn.functional as F

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        in_degree = decrease_to_max_value(
            degree(index=edge_index[1], num_nodes=num_nodes).long() + degree(index=edge_index[0],
                                                                             num_nodes=num_nodes).long(),
            self.max_in_degree - 1)
        out_degree = decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long() + degree(index=edge_index[1],
                                                                             num_nodes=num_nodes).long(),
            self.max_out_degree - 1)
        x += self.z_in[in_degree] + self.z_out[out_degree]
        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        for src in paths:
            for dst in paths[src]:
                path = paths[src][dst]
                if path:  # Check if path exists
                    spatial_matrix[src][dst] = self.b[min(len(path), self.max_path_distance) - 1]
        return spatial_matrix


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(max_path_distance, edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        device = edge_attr.device
        max_edge_idx = edge_attr.size(0) - 1
        cij = torch.zeros((x.size(0), x.size(0)), device=device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path = edge_paths[src][dst][:self.max_path_distance]
                path_tensor = torch.tensor(path, device=device).long()
                valid_mask = (path_tensor >= 0) & (path_tensor <= max_edge_idx)
                path_tensor = path_tensor[valid_mask]

                if len(path_tensor) == 0:
                    cij[src, dst] = 0.0
                else:
                    weights = self.edge_vector[:len(path_tensor)]
                    edge_feats = edge_attr[path_tensor]
                    cij[src, dst] = (weights * edge_feats).sum(dim=1).mean()
        return cij

class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)


        self.gamma_b = nn.Parameter(torch.tensor(1.0))
        self.gamma_c = nn.Parameter(torch.tensor(1.0))
        self.gamma_d = nn.Parameter(torch.tensor(1.0))

        self.sim_proj = nn.Linear(dim_in, dim_in)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, b: torch.Tensor, edge_paths, ptr=None) -> torch.Tensor:
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)

        projected_x = self.sim_proj(x)
        x_norm = F.normalize(projected_x, p=2, dim=-1)
        d = torch.abs(x_norm @ x_norm.t())
        a = (a + self.gamma_b*b + self.gamma_c * c + self.gamma_d*d ) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5
        return a


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, b: torch.Tensor, edge_paths, ptr) -> torch.Tensor:
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.node_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.ff_dim, self.node_dim)
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, b: torch.Tensor, edge_paths, ptr) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if b is None:
            b = torch.zeros((x.size(0), x.size(0)), device=x.device)
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new


class GatedGCNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.U = nn.Linear(node_dim, node_dim)
        self.V = nn.Linear(node_dim, node_dim)
        self.E = nn.Linear(edge_dim, node_dim)
        self.gate = nn.Linear(2 * node_dim, 1)
        self.edge_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 2*edge_dim),
            nn.ReLU(),
            nn.Linear(2*edge_dim, edge_dim)
        )
        self.bn = nn.BatchNorm1d(node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.edge_update[0].weight)
        nn.init.xavier_uniform_(self.edge_update[2].weight)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        e = torch.sigmoid(self.E(edge_attr))
        messages = self.V(x[dst]) * e
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, src, messages)

        combined = torch.cat([self.U(x), aggregated], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        x_new = x + torch.relu(self.bn(gate * self.U(x) + (1 - gate) * aggregated))

        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        updated_edge_attr = self.edge_update(edge_input)
        return x_new, updated_edge_attr


class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, pos_dim: int, max_freqs: int = 10):
        super().__init__()
        self.pos_dim = pos_dim
        self.max_freqs = max_freqs
        self.register_buffer("freq_linspace", torch.linspace(0.0, 1.0, self.max_freqs))
        self.linear = nn.Linear(2 * max_freqs, pos_dim)

    def _compute_laplacian_eigenvectors(self, edge_index, num_nodes):
        device = edge_index.device
        adj = torch.sparse_coo_tensor(
            edge_index.to(device),
            torch.ones(edge_index.size(1), device=device),
            (num_nodes, num_nodes)
        ).to_dense()

        degree = adj.sum(dim=1)
        D = torch.diag(degree)
        L = D - adj

        _, eigenvectors = torch.linalg.eigh(L)
        return eigenvectors[:, 1:self.max_freqs + 1]

    def forward(self, edge_index, num_nodes):
        eig_vecs = self._compute_laplacian_eigenvectors(edge_index, num_nodes)
        pe = torch.cat([
            torch.sin(self.freq_linspace.to(eig_vecs.device) * eig_vecs),
            torch.cos(self.freq_linspace.to(eig_vecs.device) * eig_vecs)
        ], dim=-1)

        return self.linear(pe)
