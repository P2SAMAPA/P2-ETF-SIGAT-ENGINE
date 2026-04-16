"""
Global and Shrinking Window training with SiGAT (Corrected with Logging).
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
from push_results import push_daily_result


def prepare_graph_data(graphs, train_end_date):
    """Extract graph data up to train_end_date."""
    if not graphs:
        print("  No graphs available.")
        return None, None, None
    train_graphs = [g for g in graphs if g[0] <= train_end_date]
    if not train_graphs:
        print(f"  No graph found on or before {train_end_date}. Earliest graph date: {graphs[0][0]}")
        return None, None, None
    last_date, adj, tickers = train_graphs[-1]
    print(f"  Using graph from {last_date} with shape {adj.shape}")
    return last_date, adj, tickers


def train_model(model, regressor, node_features, pos_edge_index, neg_edge_index,
                y_train, y_val, epochs, lr, patience, device):
    """Train the GNN + regressor with logging."""
    print(f"  Training on device: {device}")
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

        model.eval()
        regressor.eval()
        with torch.no_grad():
            val_pred = regressor(model(x, pos_edge_index, neg_edge_index))
            val_loss = criterion(val_pred, y_val)

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

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
                print(f"    Early stopping at epoch {epoch}")
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
    print(f"\n--- Global Training: {universe} ---")
    total_days = len(returns)
    train_end_idx = int(total_days * config.TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end_idx]
    val_ret = returns.iloc[train_end_idx:val_end_idx]
    test_ret = returns.iloc[val_end_idx:]

    print(f"  Data split: train={train_ret.index[0].date()} to {train_ret.index[-1].date()}, "
          f"val={val_ret.index[0].date()} to {val_ret.index[-1].date()}, "
          f"test={test_ret.index[0].date()} to {test_ret.index[-1].date()}")

    train_end_date = train_ret.index[-1]
    last_date, adj, tickers = prepare_graph_data(graphs, train_end_date)
    if adj is None:
        print("  Skipping global training due to missing graph.")
        return {"ticker": None, "metrics": {}, "test_start": "", "test_end": "", "pred_return": None}

    # Node features: use recent returns (last LOOKBACK_WINDOW days)
    recent_returns = train_ret.iloc[-config.LOOKBACK_WINDOW:]
    node_feats = recent_returns.T.values  # shape: (n_assets, lookback)
    scaler = StandardScaler()
    node_feats = scaler.fit_transform(node_feats)
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    # Targets: average forward return during training (simple proxy)
    y_all = train_ret.shift(-1).mean().values
    y_train = torch.tensor(y_all, dtype=torch.float)
    y_val = torch.tensor(val_ret.mean().values, dtype=torch.float)

    # Build edge indices
    pos_mask = adj > 0
    neg_mask = adj < 0
    pos_adj = adj * pos_mask
    neg_adj = -adj * neg_mask
    pos_edge_index, _ = edge_index_from_adjacency(pos_adj)
    neg_edge_index, _ = edge_index_from_adjacency(neg_adj)

    print(f"  Positive edges: {pos_edge_index.shape[1]}, Negative edges: {neg_edge_index.shape[1]}")

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
        print("  No test graph available; using last training graph.")
        test_adj = adj
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
    pred_return = float(scores[top_idx])
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%")

    metrics = evaluate_etf(top_etf, test_ret)
    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_shrinking_window(universe: str, returns: pd.DataFrame, graphs: list) -> dict:
    """Shrinking window training."""
    print(f"\n--- Shrinking Window: {universe} ---")
    results = []
    tickers = [col.replace("_ret", "") for col in returns.columns]

    for start_year in config.SHRINKING_START_YEARS:
        start_date = f"{start_year}-01-01"
        mask = returns.index >= start_date
        if mask.sum() < config.MIN_TRAIN_DAYS:
            print(f"  Window {start_year}: insufficient data ({mask.sum()} days)")
            continue
        window_ret = returns.loc[mask]
        total_days = len(window_ret)
        train_end_idx = int(total_days * config.TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_days * config.VAL_RATIO)

        train_ret = window_ret.iloc[:train_end_idx]
        val_ret = window_ret.iloc[train_end_idx:val_end_idx]
        test_ret = window_ret.iloc[val_end_idx:]

        if len(val_ret) < 20 or len(test_ret) < 20:
            print(f"  Window {start_year}: val/test too short")
            continue

        print(f"  Window {start_year}: train {train_ret.index[0].date()} to {train_ret.index[-1].date()}")

        train_end_date = train_ret.index[-1]
        last_date, adj, _ = prepare_graph_data(graphs, train_end_date)
        if adj is None:
            continue

        recent_returns = train_ret.iloc[-config.LOOKBACK_WINDOW:]
        node_feats = recent_returns.T.values
        scaler = StandardScaler()
        node_feats = scaler.fit_transform(node_feats)
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

        test_graphs = [g for g in graphs if g[0] <= test_ret.index[0]]
        test_adj = adj if not test_graphs else test_graphs[-1][1]
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
        pred_return = float(scores[top_idx])

        metrics = evaluate_etf(top_etf, test_ret)
        print(f"    -> Selected {top_etf}, Ann Return: {metrics.get('ann_return', 0)*100:.1f}%, Predicted Return: {pred_return*100:.2f}%")
        results.append({
            "window_start": start_date,
            "train_end": train_ret.index[-1].strftime("%Y-%m-%d"),
            "val_end": val_ret.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
            "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
            "ticker": top_etf,
            "pred_return": pred_return,
            "metrics": metrics,
        })

    if not results:
        print("  No shrinking window results produced.")
        return {"ticker": None, "windows": [], "pred_return": None}

    weighted_pick = aggregate_windows(results)
    # Get predicted return for the weighted pick (use the first window's pred_return as fallback)
    pred_for_pick = next((w["pred_return"] for w in results if w["ticker"] == weighted_pick), None)
    print(f"  Weighted ensemble pick: {weighted_pick}, Predicted Return: {pred_for_pick*100:.2f}%" if pred_for_pick else f"  Weighted ensemble pick: {weighted_pick}")
    return {"ticker": weighted_pick, "pred_return": pred_for_pick, "windows": results}


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
        print(f"\n{'='*50}\nProcessing {universe.upper()} universe\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            print(f"  No returns data for {universe}, skipping.")
            continue

        print(f"  Building rolling graphs (lookback={config.LOOKBACK_WINDOW}, freq={config.REBALANCE_FREQ})...")
        graphs = build_rolling_graphs(returns)
        print(f"  Built {len(graphs)} graphs.")

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
