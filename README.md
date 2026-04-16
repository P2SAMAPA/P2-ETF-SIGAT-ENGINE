# P2 ETF SiGAT Engine

**Signed Graph Attention Network (SiGAT) for ETF Lead‑Lag & Adversarial Relationship Modeling**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-SIGAT-ENGINE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-SIGAT-ENGINE/actions/workflows/daily_run.yml)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit)](https://p2-etf-sigat-engine.streamlit.app)

---

## 📖 Overview

Traditional unsigned graph neural networks fail to capture critical adversarial relationships between ETFs (e.g., TLT ↔ SPY).  
**P2 ETF SiGAT Engine** leverages **Signed Graph Convolutional Networks (SGCN)** and **Signed Graph Attention Networks (SiGAT)** to model both positive (correlated) and negative (anti‑correlated) edges, extracting robust cross‑universe signals.

The engine operates on three distinct ETF universes:
- **Fixed Income / Commodities** (TLT, VCIT, LQD, HYG, VNQ, GLD, SLV)
- **Equity Sectors** (QQQ, IWM, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, GDX, XME)
- **Combined Universe** (all of the above)

For each universe, two training paradigms are maintained:
- **Global Training** – fixed 80/10/10 split on the full historical dataset (2008–2026 YTD)
- **Shrinking Window** – rolling annual start years with a weighted ensemble (60% Return · 10% Sharpe · 10% Hit Rate · 20% Max DD inverted)

The final output is a daily **top ETF pick** for the next US trading day, presented via a clean Streamlit dashboard.

---

## 🧠 Methodology

### 1. Signed Graph Construction
- Daily log returns are computed for all ETFs.
- A rolling correlation matrix (60‑day lookback, recomputed every 20 days) is converted into a **signed adjacency matrix**:
  - Positive correlation → positive edge weight
  - Negative correlation → negative edge weight
- Self‑loops are added to preserve node identity.

### 2. Graph Neural Network Architecture
The default model is **SiGAT** – a signed variant of Graph Attention Networks:
- Separate GAT layers for positive and negative subgraphs.
- Node features are derived from recent return patterns.
- A final regressor maps node embeddings to a scalar **expected return score**.

### 3. Training & Selection
- The model is trained to predict next‑day relative return ranking.
- During validation, the ETF with the highest score is selected.
- For Shrinking Window, the final pick is determined by aggregating all window selections using the weighted scheme described above (negative‑return years receive zero weight).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Hugging Face Hub](https://huggingface.co/) account and token (for data access)
- GitHub repository with `HF_TOKEN` secret configured

### Installation
```bash
git clone https://github.com/P2SAMAPA/P2-ETF-SIGAT-ENGINE.git
cd P2-ETF-SIGAT-ENGINE
pip install -r requirements.txt
Configuration
All parameters are centralized in config.py:

Universes, tickers, macro columns

Training ratios, shrinking window years

Weighting scheme for ensemble

GNN hyperparameters (hidden channels, layers, dropout, learning rate, etc.)

Running Locally
bash
# Execute full training pipeline (pushes results to Hugging Face if HF_TOKEN is set)
python training.py

# Launch the Streamlit dashboard
streamlit run streamlit_app.py
GitHub Actions Automation
The workflow .github/workflows/daily_run.yml triggers every day at 06:00 UTC.
It installs dependencies, runs training.py, and uploads the output to the Hugging Face dataset.

📁 Repository Structure
text
P2-ETF-SIGAT-ENGINE/
├── .github/workflows/
│   └── daily_run.yml           # Scheduled training job
├── config.py                   # Central configuration
├── data_manager.py             # Data fetching & preprocessing
├── graph_builder.py            # Rolling signed adjacency matrix construction
├── sgat_model.py               # SiGAT / SGCN model definitions
├── training.py                 # Global & Shrinking Window orchestration
├── selector.py                 # ETF selection helpers
├── us_calendar.py              # NYSE trading calendar utilities
├── push_results.py             # Hugging Face dataset upload / download
├── streamlit_app.py            # Professional UI with three tabs
├── utils.py                    # Logging & auxiliary functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
📊 Output Data
Results are stored in the Hugging Face dataset:
P2SAMAPA/p2-etf-sigat-engine-results

Each daily run produces a JSON file named sigat_YYYY-MM-DD.json containing:

Global and Shrinking Window selections for each universe

Out‑of‑sample performance metrics

Complete window‑by‑window history for shrinking window aggregation

🎨 Streamlit Dashboard
The dashboard provides a clean, card‑based interface with three tabs (FI/Commodities, Equity, Combined).
For each universe and training mode you can:

View the current day’s top ETF pick with conviction

Examine historical out‑of‑sample metrics (ann. return, volatility, Sharpe, max drawdown, hit rate)

Drill down into individual shrinking windows via a dropdown

Inspect the weighted scoring table that determines the final shrinking window selection

⚙️ Customization
Modify ETF Universes: Edit FI_COMMODITY_TICKERS, EQUITY_TICKERS, and COMBINED_TICKERS in config.py.

Adjust Weighting: Change WEIGHT_RETURN, WEIGHT_SHARPE, WEIGHT_HITRATE, WEIGHT_MAXDD in config.py.

Tune GNN: Update HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT, LEARNING_RATE, EPOCHS, PATIENCE.

Graph Frequency: Change LOOKBACK_WINDOW and REBALANCE_FREQ for rolling adjacency matrix.

📄 License
This project is proprietary and intended for internal use by the repository owner.
