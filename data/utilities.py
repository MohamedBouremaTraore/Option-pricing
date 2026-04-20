import os
import logging
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from typing import Literal
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# 1. DATA LOADING + FUSION
# =========================================================

def find_files(
    root_path: str,
    sector: str,
    keyword: Optional[str] = None
) -> List[str]:
    """
    Recursively find CSV/XLSX/XLS files in directory.
    """
    base_path = os.path.join(root_path, sector)
    keyword = keyword.lower() if keyword else None

    matched_files = []

    for folder, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(folder, file)
            lower_path = full_path.lower()

            if lower_path.endswith((".csv", ".xlsx", ".xls")):
                if keyword is None or keyword in lower_path:
                    matched_files.append(full_path)

    logger.info(f"{len(matched_files)} files found for keyword: {keyword}")
    print(matched_files)
    return matched_files


def load_file(path: str) -> pd.DataFrame:
    """
    Load a single file (CSV or Excel).
    """
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        df["source_file"] = os.path.basename(path)
        return df

    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return pd.DataFrame()


def merge_files(files: List[str]) -> pd.DataFrame:
    """
    Merge multiple files into a single DataFrame.
    """
    dfs = [load_file(f) for f in files]
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        logger.warning("No valid files loaded.")
        return pd.DataFrame()

    df_final = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged dataset shape: {df_final.shape}")

    return df_final


def fusionner_fichiers(
    racine: str,
    sector: str,
    mot_cle: Optional[str] = None
) -> pd.DataFrame:
    """
    Full pipeline: find + load + merge files.
    """
    files = find_files(racine, sector, mot_cle)
    return merge_files(files)


# =========================================================
# 2. DATA CLEANING
# =========================================================

def remove_duplicates(
    df: pd.DataFrame,
    subset: List[str] = None
) -> pd.DataFrame:
    """
    Remove duplicates based on subset of columns.
    """
    if subset is None:
        subset = ["lastTradeDate", "contractSymbol"]

    before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep="first")
    after = len(df_clean)

    logger.info(f"Removed {before - after} duplicates")

    return df_clean


def show_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return duplicated rows based on key columns.
    """
    subset = ["lastTradeDate", "contractSymbol"]

    dup = df[df.duplicated(subset=subset, keep=False)]

    logger.info(f"{len(dup)} duplicate rows found")

    return dup.sort_values(by=subset)


# =========================================================
# 3. SPLITTING
# =========================================================

def split_options_by_type(
    df: pd.DataFrame,
    column: str = "type"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into calls and puts.
    """
    df[column] = df[column].astype(str).str.lower()

    calls = df[df[column].str.contains("call")].copy()
    puts = df[df[column].str.contains("put")].copy()

    logger.info(f"{len(calls)} calls / {len(puts)} puts")

    return calls, puts


# =========================================================
# 4. PRICING MODEL (BINOMIAL)
# =========================================================

def american_binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call"
) -> float:
    """
    Price American option using binomial tree.
    """

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # terminal stock prices
    ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    if option_type == "call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    # backward induction
    for i in range(N - 1, -1, -1):
        ST = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])

        option_values = np.exp(-r * dt) * (
            p * option_values[1:i + 2] +
            (1 - p) * option_values[0:i + 1]
        )

        # early exercise
        if option_type == "call":
            option_values = np.maximum(option_values, ST - K)
        else:
            option_values = np.maximum(option_values, K - ST)

    return float(option_values[0])

def american_binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """
    Price an American option using a binomial tree model.

    Parameters
    ----------
    S : float
        Spot price of underlying asset
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annual, decimal)
    sigma : float
        Volatility (annual, decimal)
    N : int
        Number of steps in binomial tree
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price
    """

    # ----------------------------
    # 1. Model parameters
    # ----------------------------
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    u = np.exp(sigma * sqrt_dt)
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    discount = np.exp(-r * dt)

    # ----------------------------
    # 2. Terminal asset prices
    # ----------------------------
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))

    # Payoff at maturity
    if option_type == "call":
        option_values = np.maximum(ST - K, 0.0)
    else:
        option_values = np.maximum(K - ST, 0.0)

    # ----------------------------
    # 3. Backward induction
    # ----------------------------
    for i in range(N - 1, -1, -1):

        # asset prices at step i
        j = np.arange(i + 1)
        ST = S * (u ** j) * (d ** (i - j))

        # expected discounted value
        option_values = discount * (
            p * option_values[1:i + 2] +
            (1 - p) * option_values[0:i + 1]
        )

        # early exercise condition
        if option_type == "call":
            option_values = np.maximum(option_values, ST - K)
        else:
            option_values = np.maximum(option_values, K - ST)

    return float(option_values[0])