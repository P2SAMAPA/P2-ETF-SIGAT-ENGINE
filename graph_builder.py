"""
Construct signed adjacency matrices from rolling correlations.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import config


def correlation_to_adjacency(corr_matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert a correlation matrix into a signed adjacency matrix.
    Positive correlations -> positive edge weight (+)
    Negative correlations -> negative edge weight (-)
    Edges with |corr| < threshold are dropped.
    """
    n = corr_matrix.shape[0]
    adj = np.zeros_like(corr_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                adj[i, j] = corr
                adj[j, i] = corr
    return adj


def build_rolling_graphs(returns: pd.DataFrame) -> list:
    """
    Build a sequence of signed adjacency matrices using a rolling window.
    Returns a list of (date, adjacency_matrix) tuples.
    """
    tickers = [col.replace("_ret", "") for col in returns.columns]
    n = len(tickers)
    dates = returns.index
    lookback = config.LOOKBACK_WINDOW
    freq = config.REBALANCE_FREQ

    graphs = []
    for i in range(lookback, len(returns), freq):
        window_ret = returns.iloc[i - lookback : i]
        if len(window_ret) < lookback // 2:
            continue
        corr = window_ret.corr().values
        adj = correlation_to_adjacency(corr, threshold=0.1)
        date = dates[i]
        graphs.append((date, adj, tickers))
    return graphs


def get_latest_graph(graphs: list) -> tuple:
    """Return the most recent graph (date, adj, tickers)."""
    if not graphs:
        return None, None, None
    return graphs[-1]


def edge_index_from_adjacency(adj: np.ndarray):
    """
    Convert adjacency matrix to PyG edge_index and edge_weight tensors.
    Includes self-loops.
    """
    import torch
    n = adj.shape[0]
    rows, cols = np.where(adj != 0)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float)
    # Add self-loops
    self_loops = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)
    self_weights = torch.ones(n, dtype=torch.float)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    edge_weight = torch.cat([edge_weight, self_weights])
    return edge_index, edge_weight
