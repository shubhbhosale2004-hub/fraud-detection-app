"""
utils.py — Data preprocessing utilities for the Fraud Detection System.
Handles data loading, cleaning, and synthetic data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string


# ─────────────────────────────────────────────
#  Synthetic Data Generation
# ─────────────────────────────────────────────

COUNTRIES = [
    "India", "USA", "UK", "Germany", "France",
    "China", "Japan", "UAE", "Singapore", "Brazil"
]

FRAUD_COUNTRY_PAIRS = [
    ("India", "USA"), ("India", "UK"), ("India", "UAE"),
    ("USA", "China"), ("Germany", "Brazil"),
]


def _random_transaction_id() -> str:
    """Generate a random alphanumeric transaction ID."""
    return "TXN" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def generate_synthetic_data(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic transaction dataset for demo purposes.

    Parameters
    ----------
    n_rows : int
        Number of transaction rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: transaction_id, card_id, amount, timestamp, country.
    """
    random.seed(seed)
    np.random.seed(seed)

    card_ids = [f"CARD{str(i).zfill(4)}" for i in range(1, 31)]
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    records = []

    for _ in range(n_rows):
        card = random.choice(card_ids)
        country = random.choice(COUNTRIES)

        # ~5% chance of a very high-amount transaction
        if random.random() < 0.05:
            amount = round(random.uniform(52_000, 200_000), 2)
        else:
            amount = round(np.random.lognormal(mean=8.5, sigma=1.2), 2)
            amount = min(amount, 49_999)

        offset_minutes = random.randint(0, 60 * 24 * 30)
        timestamp = base_time + timedelta(minutes=offset_minutes)

        records.append({
            "transaction_id": _random_transaction_id(),
            "card_id": card,
            "amount": amount,
            "timestamp": timestamp,
            "country": country,
        })

    # Inject rapid-succession fraud (same card, <60 s apart)
    for _ in range(8):
        card = random.choice(card_ids)
        t = base_time + timedelta(minutes=random.randint(100, 40_000))
        for j in range(random.randint(2, 3)):
            records.append({
                "transaction_id": _random_transaction_id(),
                "card_id": card,
                "amount": round(random.uniform(500, 15_000), 2),
                "timestamp": t + timedelta(seconds=random.randint(5, 55)),
                "country": random.choice(COUNTRIES),
            })

    # Inject impossible-travel fraud (same card, different countries, <1 h)
    for _ in range(6):
        card = random.choice(card_ids)
        t = base_time + timedelta(minutes=random.randint(100, 40_000))
        c1, c2 = random.choice(FRAUD_COUNTRY_PAIRS)
        records.append({
            "transaction_id": _random_transaction_id(),
            "card_id": card,
            "amount": round(random.uniform(1_000, 30_000), 2),
            "timestamp": t,
            "country": c1,
        })
        records.append({
            "transaction_id": _random_transaction_id(),
            "card_id": card,
            "amount": round(random.uniform(1_000, 30_000), 2),
            "timestamp": t + timedelta(minutes=random.randint(5, 50)),
            "country": c2,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  Data Loading & Cleaning
# ─────────────────────────────────────────────

REQUIRED_COLUMNS = {"transaction_id", "card_id", "amount", "timestamp", "country"}


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, clean, and sort a raw transactions DataFrame.

    Steps
    -----
    1. Validate required columns exist.
    2. Parse `timestamp` to datetime.
    3. Coerce `amount` to float, drop unparseable rows.
    4. Drop rows with null values in key columns.
    5. Sort by card_id then timestamp.
    6. Reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from CSV or generated synthetically.

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted DataFrame ready for fraud detection.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Coerce amount to numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop rows with critical nulls
    df.dropna(subset=["card_id", "amount", "timestamp", "country"], inplace=True)

    # Strip whitespace from string columns
    for col in ["transaction_id", "card_id", "country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Sort for sequential rule evaluation
    df.sort_values(["card_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def load_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Read a Streamlit UploadedFile object into a cleaned DataFrame.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The file object from st.file_uploader.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction DataFrame.
    """
    df_raw = pd.read_csv(uploaded_file)
    return load_and_clean(df_raw)


def build_single_transaction(
    transaction_id: str,
    card_id: str,
    amount: float,
    timestamp: datetime,
    country: str,
) -> pd.DataFrame:
    """
    Wrap a single manually-entered transaction into a one-row DataFrame.

    Parameters
    ----------
    transaction_id : str
    card_id : str
    amount : float
    timestamp : datetime
    country : str

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame compatible with the fraud-detection pipeline.
    """
    return pd.DataFrame([{
        "transaction_id": transaction_id,
        "card_id": card_id,
        "amount": float(amount),
        "timestamp": pd.Timestamp(timestamp),
        "country": country,
    }])
