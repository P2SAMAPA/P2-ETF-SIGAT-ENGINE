def select_top_etf_from_scores(scores, tickers):
    idx = scores.argmax()
    return tickers[idx]
