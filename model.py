import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.stats import t as student_t

BINANCE_URL   = "https://data-api.binance.vision/api/v3/klines"
SPAN_FAST     = 10
SPAN_SLOW     = 30
NU_MIN        = 3.0
NU_MAX        = 30.0
NU_WINDOW     = 200
N_SIM_DEFAULT = 10_000


def fetch_btc_data(limit=500, symbol="BTCUSDT", interval="1h"):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    resp   = requests.get(BINANCE_URL, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)
    return df.set_index("open_time")[["open","high","low","close","volume"]]


def conservative_vol(log_returns):
    v5  = float(log_returns.ewm(span=5).std().iloc[-1])
    v10 = float(log_returns.ewm(span=SPAN_FAST).std().iloc[-1])
    v30 = float(log_returns.ewm(span=SPAN_SLOW).std().iloc[-1])
    return max(v5, v10, v30) * 1.10


def fit_nu(log_returns, window=NU_WINDOW):
    recent = log_returns.iloc[-window:].dropna().values
    if len(recent) < 30:
        return 6.0
    try:
        fitted_df, _loc, _scale = student_t.fit(recent, floc=0)
        return float(np.clip(fitted_df, NU_MIN, NU_MAX))
    except Exception:
        return 6.0


def vol_regime(log_returns):
    vol_fast = float(log_returns.ewm(span=SPAN_FAST).std().iloc[-1])
    vol_slow = float(log_returns.ewm(span=SPAN_SLOW).std().iloc[-1])
    ratio    = vol_fast / vol_slow if vol_slow > 0 else 1.0
    if ratio > 1.4:
        return "HIGH",   "🔴"
    if ratio < 0.7:
        return "LOW",    "🟢"
    return "NORMAL", "🟡"


def predict_range(close_series, alpha=0.05, n_sim=N_SIM_DEFAULT):
    min_bars = SPAN_SLOW + 5
    if len(close_series) < min_bars:
        raise ValueError(f"predict_range needs >= {min_bars} bars; got {len(close_series)}")

    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    S0          = float(close_series.iloc[-1])
    mu          = float(log_returns.iloc[-SPAN_SLOW:].mean())
    sigma       = conservative_vol(log_returns)
    nu          = fit_nu(log_returns)

    t_scale     = sigma / np.sqrt(nu / (nu - 2.0))
    innovations = stats.t.rvs(df=nu, scale=t_scale, size=n_sim)
    next_prices = S0 * np.exp(mu + innovations)

    lower = float(np.percentile(next_prices, 100.0 * alpha / 2.0))
    upper = float(np.percentile(next_prices, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper, S0, nu


def winkler_score(lower, upper, actual, alpha=0.05):
    width = upper - lower
    if actual < lower:
        return width + (2.0 / alpha) * (lower - actual)
    if actual > upper:
        return width + (2.0 / alpha) * (actual - upper)
    return width


from dataclasses import dataclass

_REGIME_THRESHOLDS = [
    ("CALM",     0.45),
    ("NORMAL",   0.75),
    ("ELEVATED", 1.20),
    ("EXTREME",  9999.),
]
REGIME_COLORS = {
    "CALM": "#00C48C", "NORMAL": "#F7931A",
    "ELEVATED": "#FFB800", "EXTREME": "#FF4B4B",
}


@dataclass
class Prediction:
    lower_50: float
    upper_50: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    current_price:  float
    volatility_1h:  float
    volatility_ann: float
    regime:         str
    nu:             float
    drift_1h:       float

    @property
    def midpoint(self):
        return (self.lower_95 + self.upper_95) / 2.0

    def width_at(self, level):
        return getattr(self, f"upper_{level}") - getattr(self, f"lower_{level}")

    def hit_at(self, actual, level):
        lo = getattr(self, f"lower_{level}")
        hi = getattr(self, f"upper_{level}")
        return bool(lo <= actual <= hi)


def classify_regime(sigma_1h):
    vol_ann = sigma_1h * np.sqrt(8_760)
    for label, threshold in _REGIME_THRESHOLDS:
        if vol_ann < threshold:
            return label
    return "EXTREME"


def predict_multi(close_series, n_sim=10_000):
    min_bars = SPAN_SLOW + 5
    if len(close_series) < min_bars:
        raise ValueError(f"Need >= {min_bars} bars; got {len(close_series)}")

    log_rets = np.log(close_series / close_series.shift(1)).dropna()
    S0       = float(close_series.iloc[-1])
    mu       = float(log_rets.iloc[-SPAN_SLOW:].mean())
    sigma    = conservative_vol(log_rets)
    nu       = fit_nu(log_rets)

    t_scale     = sigma / np.sqrt(nu / (nu - 2.0))
    innovations = stats.t.rvs(df=nu, scale=t_scale, size=n_sim)
    next_prices = S0 * np.exp(mu + innovations)

    def pct(p):
        return float(np.percentile(next_prices, p))

    return Prediction(
        lower_50=pct(25.0), upper_50=pct(75.0),
        lower_80=pct(10.0), upper_80=pct(90.0),
        lower_95=pct( 2.5), upper_95=pct(97.5),
        current_price  = S0,
        volatility_1h  = sigma,
        volatility_ann = sigma * np.sqrt(8_760),
        regime         = classify_regime(sigma),
        nu             = nu,
        drift_1h       = mu,
    )


def score_prediction(pred, actual):
    return {
        "hit_95":     pred.hit_at(actual, 95),
        "width_95":   pred.width_at(95),
        "winkler_95": winkler_score(pred.lower_95, pred.upper_95, actual, 0.05),
        "hit_80":     pred.hit_at(actual, 80),
        "width_80":   pred.width_at(80),
        "winkler_80": winkler_score(pred.lower_80, pred.upper_80, actual, 0.20),
        "hit_50":     pred.hit_at(actual, 50),
        "width_50":   pred.width_at(50),
        "winkler_50": winkler_score(pred.lower_50, pred.upper_50, actual, 0.50),
    }
