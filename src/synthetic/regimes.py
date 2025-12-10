"""
Regime module for synthetic universe generation.

Clusters Greg sensors from real market data to infer volatility regimes,
then uses those regimes to drive synthetic RV/IV paths with AR(1) dynamics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


GREG_REGIME_FILE = Path("data/greg_regimes.json")

CANONICAL_DELTAS = [0.10, 0.25, 0.50, 0.75, 0.90]

DEFAULT_SKEW_TEMPLATE = {
    "0.10": 0.30,
    "0.25": 0.10,
    "0.50": 0.00,
    "0.75": -0.05,
    "0.90": -0.10,
}


@dataclass
class RegimeParams:
    """Parameters defining a volatility regime for synthetic simulation."""
    name: str
    mu_rv_30d: float
    mu_vrp_30d: float
    iv_level_sigma: float
    skew_template: Dict[str, float] = field(default_factory=lambda: DEFAULT_SKEW_TEMPLATE.copy())
    phi_iv: float = 0.9
    phi_skew: float = 0.85
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeParams":
        """Create from dict."""
        return cls(**d)


@dataclass
class RegimeModel:
    """Complete regime model with parameters and transition matrix."""
    regimes: Dict[int, RegimeParams]
    transition_matrix: np.ndarray
    underlying: str
    n_clusters: int
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "underlying": self.underlying,
            "n_clusters": self.n_clusters,
            "created_at": self.created_at,
            "regimes": {str(k): v.to_dict() for k, v in self.regimes.items()},
            "transition_matrix": self.transition_matrix.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeModel":
        """Create from dict."""
        regimes = {int(k): RegimeParams.from_dict(v) for k, v in d["regimes"].items()}
        return cls(
            underlying=d["underlying"],
            n_clusters=d["n_clusters"],
            created_at=d.get("created_at", ""),
            regimes=regimes,
            transition_matrix=np.array(d["transition_matrix"]),
        )


GREG_SENSOR_COLUMNS = [
    "vrp_30d",
    "vrp_7d",
    "adx_14d",
    "chop_factor_7d",
    "iv_rank_6m",
    "term_structure_spread",
    "skew_25d",
    "rsi_14d",
    "price_vs_ma200",
    "rv_30d",
    "iv_atm_30d",
]


def cluster_greg_sensors_from_real_data(
    df_sensors: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster Greg sensor data to infer volatility regimes.
    
    Args:
        df_sensors: DataFrame with sensor columns (vrp_30d, adx_14d, etc.)
        n_clusters: Number of regime clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with 'regime_id' column, fitted KMeans model)
    """
    available_cols = [c for c in GREG_SENSOR_COLUMNS if c in df_sensors.columns]
    if not available_cols:
        raise ValueError(f"No valid sensor columns found. Expected any of: {GREG_SENSOR_COLUMNS}")
    
    df_work = df_sensors[available_cols].copy()
    
    df_work = df_work.ffill().bfill()
    
    for col in df_work.columns:
        if df_work[col].isna().any():
            df_work[col] = df_work[col].fillna(df_work[col].median())
    
    df_work = df_work.dropna()
    
    if len(df_work) < n_clusters:
        raise ValueError(f"Not enough data points ({len(df_work)}) for {n_clusters} clusters")
    
    means = df_work.mean()
    stds = df_work.std()
    stds = stds.replace(0, 1)
    df_normalized = (df_work - means) / stds
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(df_normalized)
    
    result = df_sensors.loc[df_work.index].copy()
    result["regime_id"] = labels
    
    return result, kmeans


def fit_regime_params_from_clusters(
    df_sensors_with_regime: pd.DataFrame,
) -> Dict[int, RegimeParams]:
    """
    Aggregate per-cluster statistics to build RegimeParams.
    
    Args:
        df_sensors_with_regime: DataFrame with 'regime_id' column and sensor data
        
    Returns:
        Dict mapping regime_id to RegimeParams
    """
    if "regime_id" not in df_sensors_with_regime.columns:
        raise ValueError("DataFrame must have 'regime_id' column")
    
    regimes: Dict[int, RegimeParams] = {}
    
    for regime_id in sorted(df_sensors_with_regime["regime_id"].unique()):
        cluster_data = df_sensors_with_regime[df_sensors_with_regime["regime_id"] == regime_id]
        
        if "rv_30d" in cluster_data.columns:
            mu_rv_30d = float(cluster_data["rv_30d"].mean())
        else:
            mu_rv_30d = 50.0
        
        if "vrp_30d" in cluster_data.columns:
            vrp = cluster_data["vrp_30d"].mean()
            if mu_rv_30d > 0:
                mu_vrp_30d = float(vrp / mu_rv_30d) if mu_rv_30d > 0 else 0.1
            else:
                mu_vrp_30d = 0.1
        else:
            mu_vrp_30d = 0.1
        
        if "iv_atm_30d" in cluster_data.columns and "rv_30d" in cluster_data.columns:
            iv_std = cluster_data["iv_atm_30d"].std()
            iv_level_sigma = float(iv_std) if not np.isnan(iv_std) else 5.0
        else:
            iv_level_sigma = 5.0
        
        if "skew_25d" in cluster_data.columns:
            skew_25d = cluster_data["skew_25d"].mean()
            skew_template = _estimate_skew_template_from_25d(float(skew_25d))
        else:
            skew_template = DEFAULT_SKEW_TEMPLATE.copy()
        
        if "adx_14d" in cluster_data.columns:
            adx = cluster_data["adx_14d"].mean()
            phi_iv = 0.95 if adx > 30 else (0.85 if adx < 15 else 0.90)
        else:
            phi_iv = 0.90
        
        phi_skew = 0.85
        
        name = _name_regime(regime_id, mu_rv_30d, mu_vrp_30d, phi_iv)
        
        regimes[int(regime_id)] = RegimeParams(
            name=name,
            mu_rv_30d=round(mu_rv_30d, 2),
            mu_vrp_30d=round(mu_vrp_30d, 4),
            iv_level_sigma=round(iv_level_sigma, 2),
            skew_template=skew_template,
            phi_iv=round(phi_iv, 2),
            phi_skew=round(phi_skew, 2),
        )
    
    return regimes


def _name_regime(regime_id: int, mu_rv: float, mu_vrp: float, phi_iv: float) -> str:
    """Generate a descriptive name for a regime."""
    vol_level = "high_vol" if mu_rv > 60 else ("low_vol" if mu_rv < 40 else "mid_vol")
    vrp_level = "rich_vrp" if mu_vrp > 0.15 else ("cheap_vrp" if mu_vrp < 0.05 else "fair_vrp")
    trend = "trending" if phi_iv > 0.92 else "ranging"
    return f"regime_{regime_id}_{vol_level}_{vrp_level}_{trend}"


def _estimate_skew_template_from_25d(skew_25d: float) -> Dict[str, float]:
    """
    Estimate a full skew template from the 25-delta skew observation.
    
    Args:
        skew_25d: Observed 25-delta skew (IV_25d_put - IV_25d_call) / ATM_IV
        
    Returns:
        Skew template dict mapping delta strings to skew multipliers
    """
    base_otm_put_skew = abs(skew_25d) if skew_25d > 0 else 0.10
    
    return {
        "0.10": round(base_otm_put_skew * 2.5, 3),
        "0.25": round(base_otm_put_skew, 3),
        "0.50": 0.0,
        "0.75": round(-base_otm_put_skew * 0.3, 3),
        "0.90": round(-base_otm_put_skew * 0.6, 3),
    }


def estimate_transition_matrix(
    regime_sequence: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """
    Estimate a Markov transition matrix from observed regime sequence.
    
    Args:
        regime_sequence: Array of regime IDs over time
        n_regimes: Number of unique regimes
        
    Returns:
        Row-stochastic transition matrix of shape (n_regimes, n_regimes)
    """
    counts = np.zeros((n_regimes, n_regimes))
    
    for i in range(len(regime_sequence) - 1):
        from_regime = int(regime_sequence[i])
        to_regime = int(regime_sequence[i + 1])
        if 0 <= from_regime < n_regimes and 0 <= to_regime < n_regimes:
            counts[from_regime, to_regime] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = counts / row_sums
    
    for i in range(n_regimes):
        if transition_matrix[i].sum() < 1e-9:
            transition_matrix[i] = np.ones(n_regimes) / n_regimes
    
    return transition_matrix


def sample_regime_path(
    T: int,
    transition_matrix: np.ndarray,
    start_regime: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a sequence of regime_ids of length T from a Markov chain.
    
    Args:
        T: Length of sequence to generate
        transition_matrix: Row-stochastic transition matrix
        start_regime: Initial regime ID
        rng: Random number generator (uses default if None)
        
    Returns:
        Array of regime IDs of length T
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_regimes = transition_matrix.shape[0]
    path = np.zeros(T, dtype=int)
    path[0] = start_regime
    
    for t in range(1, T):
        current = path[t - 1]
        probs = transition_matrix[current]
        probs = probs / probs.sum()
        path[t] = rng.choice(n_regimes, p=probs)
    
    return path


def evolve_iv_and_skew(
    iv_atm_prev: float,
    rv_30d_t: float,
    regime: RegimeParams,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Evolve ATM IV and skew state using AR(1) dynamics with VRP target.
    
    Args:
        iv_atm_prev: Previous ATM IV (as percentage, e.g., 50.0 for 50%)
        rv_30d_t: Current 30-day realized volatility (percentage)
        regime: Regime parameters
        rng: Random number generator
        
    Returns:
        Tuple of (iv_atm_t, skew_state_t)
        - iv_atm_t: New ATM IV (percentage)
        - skew_state_t: Skew factor that modulates regime.skew_template
    """
    if rng is None:
        rng = np.random.default_rng()
    
    iv_target_t = rv_30d_t * (1 + regime.mu_vrp_30d)
    
    deviation = iv_atm_prev - iv_target_t
    mean_revert = iv_target_t + regime.phi_iv * deviation
    
    eps_iv = rng.normal(0, regime.iv_level_sigma)
    iv_atm_t = mean_revert + eps_iv
    
    iv_atm_t = max(5.0, min(200.0, iv_atm_t))
    
    skew_state_t = regime.phi_skew * rng.normal(0, 0.1)
    
    return iv_atm_t, skew_state_t


def iv_for_delta(
    iv_atm_t: float,
    regime: RegimeParams,
    skew_state_t: float,
    delta: float,
) -> float:
    """
    Compute IV for a given delta using the regime skew template.
    
    Args:
        iv_atm_t: Current ATM IV (percentage)
        regime: Regime parameters with skew_template
        skew_state_t: Current skew state factor
        delta: Option delta (0.0 to 1.0 for calls)
        
    Returns:
        IV for the given delta (percentage)
    """
    skew_template = regime.skew_template
    
    sorted_items = sorted([(float(k), v) for k, v in skew_template.items()], key=lambda x: x[0])
    delta_keys = [item[0] for item in sorted_items]
    delta_vals = [item[1] for item in sorted_items]
    
    delta = max(0.05, min(0.95, delta))
    
    if delta <= delta_keys[0]:
        base_skew = delta_vals[0]
    elif delta >= delta_keys[-1]:
        base_skew = delta_vals[-1]
    else:
        for i in range(len(delta_keys) - 1):
            if delta_keys[i] <= delta <= delta_keys[i + 1]:
                d1, d2 = delta_keys[i], delta_keys[i + 1]
                s1, s2 = delta_vals[i], delta_vals[i + 1]
                t = (delta - d1) / (d2 - d1 + 1e-9)
                base_skew = s1 + t * (s2 - s1)
                break
        else:
            base_skew = 0.0
    
    effective_skew = base_skew * (1 + skew_state_t)
    
    iv = iv_atm_t * (1 + effective_skew)
    
    iv = max(5.0, min(300.0, iv))
    
    return iv


def load_regime_model(underlying: str, path: Optional[Path] = None) -> Optional[RegimeModel]:
    """
    Load a saved regime model from disk.
    
    Args:
        underlying: Asset symbol (BTC, ETH)
        path: Path to JSON file (defaults to data/greg_regimes.json)
        
    Returns:
        RegimeModel if found, None otherwise
    """
    path = path or GREG_REGIME_FILE
    
    if not path.exists():
        return None
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        key = underlying.upper()
        if key not in data:
            return None
        
        return RegimeModel.from_dict(data[key])
    except Exception:
        return None


def save_regime_model(model: RegimeModel, path: Optional[Path] = None) -> None:
    """
    Save a regime model to disk.
    
    Args:
        model: RegimeModel to save
        path: Path to JSON file (defaults to data/greg_regimes.json)
    """
    path = path or GREG_REGIME_FILE
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    data[model.underlying.upper()] = model.to_dict()
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_default_regimes() -> Dict[int, RegimeParams]:
    """
    Return default hard-coded regimes as fallback.
    
    These are used when no calibrated regime model is available.
    """
    return {
        0: RegimeParams(
            name="default_low_vol_fair_vrp",
            mu_rv_30d=35.0,
            mu_vrp_30d=0.10,
            iv_level_sigma=4.0,
            phi_iv=0.90,
            phi_skew=0.85,
        ),
        1: RegimeParams(
            name="default_mid_vol_rich_vrp",
            mu_rv_30d=50.0,
            mu_vrp_30d=0.18,
            iv_level_sigma=6.0,
            phi_iv=0.88,
            phi_skew=0.82,
        ),
        2: RegimeParams(
            name="default_high_vol_cheap_vrp",
            mu_rv_30d=75.0,
            mu_vrp_30d=0.05,
            iv_level_sigma=10.0,
            phi_iv=0.92,
            phi_skew=0.80,
        ),
        3: RegimeParams(
            name="default_trending_high_vol",
            mu_rv_30d=65.0,
            mu_vrp_30d=0.12,
            iv_level_sigma=8.0,
            phi_iv=0.95,
            phi_skew=0.88,
        ),
        4: RegimeParams(
            name="default_ranging_mid_vol",
            mu_rv_30d=45.0,
            mu_vrp_30d=0.08,
            iv_level_sigma=5.0,
            phi_iv=0.85,
            phi_skew=0.80,
        ),
        5: RegimeParams(
            name="default_crisis_spike",
            mu_rv_30d=100.0,
            mu_vrp_30d=0.25,
            iv_level_sigma=15.0,
            phi_iv=0.80,
            phi_skew=0.75,
        ),
    }


def get_default_transition_matrix(n_regimes: int = 6) -> np.ndarray:
    """
    Return a default transition matrix with high persistence on diagonal.
    """
    matrix = np.full((n_regimes, n_regimes), 0.02)
    for i in range(n_regimes):
        matrix[i, i] = 0.90 - 0.02 * (n_regimes - 1)
    return matrix / matrix.sum(axis=1, keepdims=True)
