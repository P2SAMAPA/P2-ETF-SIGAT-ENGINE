"""
Configuration for P2-ETF-SIGAT-ENGINE.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-sigat-engine-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]
# Combined universe: FI + Equity
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

# Macro columns (available in dataset)
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2  # 2 years

# Shrinking window start years
SHRINKING_START_YEARS = list(range(2008, 2025))  # 2008 through 2024

# Selection weights for shrinking window aggregation
WEIGHT_RETURN = 0.6
WEIGHT_SHARPE = 0.1
WEIGHT_HITRATE = 0.1
WEIGHT_MAXDD = 0.2

TRADING_DAYS_PER_YEAR = 252

# Graph construction parameters
LOOKBACK_WINDOW = 60  # days for rolling correlation
REBALANCE_FREQ = 20   # recompute graph every N trading days

# GNN Model hyperparameters
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 20  # early stopping

# Device (CPU for GitHub Actions)
DEVICE = "cpu"
