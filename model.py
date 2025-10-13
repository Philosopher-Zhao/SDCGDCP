import torch.nn as nn
from layers import CentralityEncoding, SpatialEncoding, EdgeEncoding, GraphormerEncoderLayer, GatedGCNLayer, \
    LaplacianPositionalEncoding
from config import Config
import torch


class EnhancedFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        return self.proj(x)
class SampledGraphormer(nn.Module):
    def __init__(self, input_dim, cell_dim):
        super().__init__()
        self.dropout = nn.Dropout(0.3)

        self.node_proj_in = EnhancedFeatureProjector(input_dim, Config.node_dim)
        self.edge_proj_in = EnhancedFeatureProjector(cell_dim, Config.cell_dim)

        self.gated_gcn_layers = nn.ModuleList([
            GatedGCNLayer(Config.node_dim, Config.cell_dim) for _ in range(1)
        ])

        self.centrality_enc = CentralityEncoding(
            max_in_degree=Config.max_degree,
            max_out_degree=Config.max_degree,
            node_dim=Config.node_dim
        )
        self.sign_net = LaplacianPositionalEncoding(Config.node_dim, max_freqs=20)


        self.spatial_enc = SpatialEncoding(Config.max_path_length)
        self.edge_enc = EdgeEncoding(Config.cell_dim, Config.max_path_length)
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                Config.node_dim, Config.cell_dim, Config.n_heads, Config.ff_dim, Config.max_path_length
            ) for _ in range(Config.num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(2 * Config.node_dim + Config.cell_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, data, mode='train'):
        x = self.dropout(self.node_proj_in(data.x))
        edge_attr = self.dropout(self.edge_proj_in(data.edge_attr))

        for layer in self.gated_gcn_layers:
            x, edge_attr = layer(x, data.edge_index, edge_attr)
            x = self.dropout(x)
            edge_attr = self.dropout(edge_attr)

        x = self.centrality_enc(x, data.edge_index)
        sign_pe = self.sign_net(data.edge_index, x.size(0))
        x = x + sign_pe

        b = self.spatial_enc(x, data.paths)
        c = self.edge_enc(x, edge_attr, data.paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, data.paths, None)

        # 最终预测
        src, dst = data.edge_index
        h = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.head(h).squeeze(), data.y
