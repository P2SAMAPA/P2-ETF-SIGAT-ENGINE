"""
Global and Shrinking Window training with SiGAT.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from graph_builder import build_rolling_graphs, edge_index_from_adjacency
from sgat_model import SiGAT, ETFRegressor
from selector import select_top_etf_from_scores
from us_calendar import next_trading_day
from push_results import push_daily_result


def prepare_graph_data(graphs, train_end_date):
    """Extract graph data up to train_end_date."""
    train_graphs = [g for g in graphs if g[0] <= train_end_date]
    if not train_graphs:
        return None
    last_date, adj, tickers = train_graphs[-1]
    return last_date, adj, tickers


def train_model(model, regressor, node_features, pos_edge_index, neg_edge_index,
                y_train, y_val, epochs, lr, patience, device):
    """Train the GNN + regressor to predict next-day return ranking."""
    optimizer = optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=lr)
    criterion = nn.MSELoss()

    x = node_features.to(device)
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        regressor.train()
        optimizer.zero_grad()
        embeddings = model(x, pos_edge_index, neg_edge_index)
        pred = regressor(embeddings)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        regressor.eval()
        with torch.no_grad():
            val_pred = regressor(model(x, pos_edge_index, neg_edge_index))
            val_loss = criterion(val_pred, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                'model': model.state_dict(),
                'regressor': regressor.state_dict()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state['model'])
    regressor.load_state_dict(best_state['regressor'])
    return model, regressor


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    """Compute performance metrics for a given ETF ticker."""
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}

    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    hit_rate = (ret_series > 0).mean()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
    }


def train_global(universe: str, returns: pd.DataFrame, graphs: list) -> dict:
    """Global training (80/10/10 split)."""
    total_days = len(returns)
    train_end_idx = int(total_days * config.TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end_idx]
    val_ret = returns.iloc[train_end_idx:val_end_idx]
    test_ret = returns.iloc[val_end_idx:]

    train_end_date = train_ret.index[-1]
    # Use the most recent graph up to train_end_date
    last_date, adj, tickers = prepare_graph_data(graphs, train_end_date)
    if adj is None:
        return {"ticker": None, "metrics": {}, "test_start": "", "test_end": ""}

    # Prepare features: use lagged returns (1 day) as node features
    # For simplicity, we use the last day's returns as node features.
    # In a full implementation, you could use more sophisticated features.
    scaler = StandardScaler()
    node_feats = scaler.fit_transform(train_ret.iloc[-config.LOOKBACK_WINDOW:].T.values)
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    # Targets: average next-day return during training period (or ranking)
    # We'll use average forward return as target for regression.
    y_all = train_ret.shift(-1).mean().values  # crude but works
    y_train = torch.tensor(y_all, dtype=torch.float)
    y_val = torch.tensor(val_ret.mean().values, dtype=torch.float)  # placeholder

    # Build edge indices
    pos_mask = adj > 0
    neg_mask = adj < 0
    pos_adj = adj * pos_mask
    neg_adj = -adj * neg_mask
    pos_edge_index, _ = edge_index_from_adjacency(pos_adj)
    neg_edge_index, _ = edge_index_from_adjacency(neg_adj)

    in_channels = node_feats.shape[1]
    model = SiGAT(in_channels, config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS,
                  num_layers=config.NUM_LAYERS, dropout=config.DROPOUT)
    regressor = ETFRegressor(config.HIDDEN_CHANNELS)

    model, regressor = train_model(
        model, regressor, node_feats, pos_edge_index, neg_edge_index,
        y_train, y_val, config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE
    )

    # Predict on test graph (use last graph before test)
    test_graphs = [g for g in graphs if g[0] <= test_ret.index[0]]
    if not test_graphs:
        top_etf = tickers[0]
    else:
        _, test_adj, _ = test_graphs[-1]
        pos_mask_t = test_adj > 0
        neg_mask_t = test_adj < 0
        pos_adj_t = test_adj * pos_mask_t
        neg_adj_t = -test_adj * neg_mask_t
        pos_edge_idx_t, _ = edge_index_from_adjacency(pos_adj_t)
        neg_edge_idx_t, _ = edge_index_from_adjacency(neg_adj_t)

        model.eval()
        regressor.eval()
        with torch.no_grad():
            embeddings = model(node_feats, pos_edge_idx_t, neg_edge_idx_t)
            scores = regressor(embeddings).numpy()
        top_idx = np.argmax(scores)
        top_etf = tickers[top_idx]

    metrics = evaluate_etf(top_etf, test_ret)
    return {
        "ticker": top_etf,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_shrinking_window(universe: str, returns: pd.DataFrame, graphs: list) -> dict:
    """Shrinking window training."""
    results = []
    tickers = [col.replace("_ret", "") for col in returns.columns]

    for start_year in config.SHRINKING_START_YEARS:
        start_date = f"{start_year}-01-01"
        mask = returns.index >= start_date
        if mask.sum() < config.MIN_TRAIN_DAYS:
            continue
        window_ret = returns.loc[mask]
        total_days = len(window_ret)
        train_end_idx = int(total_days * config.TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_days * config.VAL_RATIO)

        train_ret = window_ret.iloc[:train_end_idx]
        val_ret = window_ret.iloc[train_end_idx:val_end_idx]
        test_ret = window_ret.iloc[val_end_idx:]

        if len(val_ret) < 20 or len(test_ret) < 20:
            continue

        train_end_date = train_ret.index[-1]
        last_date, adj, _ = prepare_graph_data(graphs, train_end_date)
        if adj is None:
            continue

        scaler = StandardScaler()
        node_feats = scaler.fit_transform(train_ret.iloc[-config.LOOKBACK_WINDOW:].T.values)
        node_feats = torch.tensor(node_feats, dtype=torch.float)

        y_train = torch.tensor(train_ret.shift(-1).mean().values, dtype=torch.float)
        y_val = torch.tensor(val_ret.mean().values, dtype=torch.float)

        pos_mask = adj > 0
        neg_mask = adj < 0
        pos_adj = adj * pos_mask
        neg_adj = -adj * neg_mask
        pos_edge_index, _ = edge_index_from_adjacency(pos_adj)
        neg_edge_index, _ = edge_index_from_adjacency(neg_adj)

        in_channels = node_feats.shape[1]
        model = SiGAT(in_channels, config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS,
                      num_layers=config.NUM_LAYERS, dropout=config.DROPOUT)
        regressor = ETFRegressor(config.HIDDEN_CHANNELS)

        model, regressor = train_model(
            model, regressor, node_feats, pos_edge_index, neg_edge_index,
            y_train, y_val, config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE
        )

        # Predict on test graph
        test_graphs = [g for g in graphs if g[0] <= test_ret.index[0]]
        if not test_graphs:
            top_etf = tickers[0]
        else:
            _, test_adj, _ = test_graphs[-1]
            pos_mask_t = test_adj > 0
            neg_mask_t = test_adj < 0
            pos_adj_t = test_adj * pos_mask_t
            neg_adj_t = -test_adj * neg_mask_t
            pos_edge_idx_t, _ = edge_index_from_adjacency(pos_adj_t)
            neg_edge_idx_t, _ = edge_index_from_adjacency(neg_adj_t)

            model.eval()
            regressor.eval()
            with torch.no_grad():
                embeddings = model(node_feats, pos_edge_idx_t, neg_edge_idx_t)
                scores = regressor(embeddings).numpy()
            top_idx = np.argmax(scores)
            top_etf = tickers[top_idx]

        metrics = evaluate_etf(top_etf, test_ret)
        results.append({
            "window_start": start_date,
            "train_end": train_ret.index[-1].strftime("%Y-%m-%d"),
            "val_end": val_ret.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
            "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
            "ticker": top_etf,
            "metrics": metrics,
        })

    if not results:
        return {"ticker": None, "windows": []}

    weighted_pick = aggregate_windows(results)
    return {"ticker": weighted_pick, "windows": results}


def aggregate_windows(windows: list) -> str:
    scores = {}
    for w in windows:
        ticker = w["ticker"]
        ret = w["metrics"].get("ann_return", 0.0)
        sharpe = w["metrics"].get("sharpe", 0.0)
        max_dd = w["metrics"].get("max_dd", -1.0)
        hit_rate = w["metrics"].get("hit_rate", 0.0)

        if ret <= 0:
            weight = 0.0
        else:
            dd_score = 1.0 / (1.0 + abs(max_dd))
            weight = (config.WEIGHT_RETURN * ret +
                      config.WEIGHT_SHARPE * sharpe +
                      config.WEIGHT_HITRATE * hit_rate +
                      config.WEIGHT_MAXDD * dd_score)
        scores[ticker] = scores.get(ticker, 0.0) + weight
    if not scores:
        return windows[-1]["ticker"] if windows else None
    return max(scores, key=scores.get)


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"Processing {universe} universe...")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        # Build graphs from full returns (for global and shrinking)
        graphs = build_rolling_graphs(returns)
        global_res = train_global(universe, returns, graphs)
        shrinking_res = train_shrinking_window(universe, returns, graphs)
        all_results[universe] = {
            "global": global_res,
            "shrinking": shrinking_res,
        }
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
