"""
Signed Graph Neural Network models (SGCN / SiGAT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class SignedGCN(nn.Module):
    """
    Simplified Signed Graph Convolutional Network.
    Uses separate weight matrices for positive and negative edges.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Positive edge convolutions
        self.pos_convs = nn.ModuleList()
        # Negative edge convolutions
        self.neg_convs = nn.ModuleList()

        self.pos_convs.append(GCNConv(in_channels, hidden_channels))
        self.neg_convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.pos_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.neg_convs.append(GCNConv(hidden_channels, hidden_channels))

        self.pos_convs.append(GCNConv(hidden_channels, out_channels))
        self.neg_convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, pos_edge_index, neg_edge_index):
        for i in range(self.num_layers - 1):
            x_pos = self.pos_convs[i](x, pos_edge_index)
            x_neg = self.neg_convs[i](x, neg_edge_index)
            x = x_pos - x_neg  # balance theory inspired combination
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_pos = self.pos_convs[-1](x, pos_edge_index)
        x_neg = self.neg_convs[-1](x, neg_edge_index)
        x = x_pos - x_neg
        return x


class SiGAT(nn.Module):
    """
    Signed Graph Attention Network.
    Uses GATConv for both positive and negative subgraphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.pos_gats = nn.ModuleList()
        self.neg_gats = nn.ModuleList()

        # First layer
        self.pos_gats.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.neg_gats.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.pos_gats.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.neg_gats.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # Output layer
        self.pos_gats.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.neg_gats.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, pos_edge_index, neg_edge_index):
        for i in range(self.num_layers - 1):
            x_pos = self.pos_gats[i](x, pos_edge_index)
            x_neg = self.neg_gats[i](x, neg_edge_index)
            x = x_pos - x_neg
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_pos = self.pos_gats[-1](x, pos_edge_index)
        x_neg = self.neg_gats[-1](x, neg_edge_index)
        x = x_pos - x_neg
        return x


class ETFRegressor(nn.Module):
    """
    Wrapper that takes node embeddings and produces a scalar score for each ETF
    (used for ranking to select the best ETF).
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        return self.fc(embeddings).squeeze(-1)
