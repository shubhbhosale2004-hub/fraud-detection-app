"""
fraud_rules.py — Rule-based fraud detection engine.

Each rule is implemented as a standalone function that annotates a DataFrame
with fraud scores and reasons.  The main entry point is `apply_all_rules()`.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────
#  Thresholds & Weights
# ─────────────────────────────────────────────

HIGH_AMOUNT_THRESHOLD = 50_000       # INR
RAPID_SUCCESSION_WINDOW_SEC = 60     # seconds
IMPOSSIBLE_TRAVEL_WINDOW_MIN = 60    # minutes

# Fraud score contributions (sum → overall risk score)
SCORE_HIGH_AMOUNT = 40
SCORE_RAPID_SUCCESSION = 35
SCORE_IMPOSSIBLE_TRAVEL = 50

FRAUD_THRESHOLD_SCORE = 30           # minimum score to be flagged is_fraud=True


# ─────────────────────────────────────────────
#  Individual Rules
# ─────────────────────────────────────────────

def rule_high_amount(df: pd.DataFrame) -> pd.Series:
    """
    Rule 1 — High-Value Transaction.

    Flag any transaction whose `amount` exceeds HIGH_AMOUNT_THRESHOLD (INR).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions DataFrame.

    Returns
    -------
    pd.Series[bool]
        Boolean mask; True where the rule fires.
    """
    return df["amount"] > HIGH_AMOUNT_THRESHOLD


def rule_rapid_succession(df: pd.DataFrame) -> pd.Series:
    """
    Rule 2 — Rapid-Succession Transactions.

    Flag transactions that occur within RAPID_SUCCESSION_WINDOW_SEC seconds
    of *any other* transaction on the same card.

    Algorithm
    ---------
    For each card, compute the time delta to both the previous and the next
    transaction.  If either delta < threshold the row is flagged.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by (card_id, timestamp).

    Returns
    -------
    pd.Series[bool]
        Boolean mask; True where the rule fires.
    """
    df = df.copy()
    df["_prev_ts"] = df.groupby("card_id")["timestamp"].shift(1)
    df["_next_ts"] = df.groupby("card_id")["timestamp"].shift(-1)

    window = pd.Timedelta(seconds=RAPID_SUCCESSION_WINDOW_SEC)

    prev_close = (df["timestamp"] - df["_prev_ts"]).abs() < window
    next_close = (df["_next_ts"] - df["timestamp"]).abs() < window

    mask = (prev_close | next_close).fillna(False)
    return mask


def rule_impossible_travel(df: pd.DataFrame) -> pd.Series:
    """
    Rule 3 — Impossible Travel (Different Countries Within Short Window).

    Flag transactions where the same card makes purchases in *different* countries
    within IMPOSSIBLE_TRAVEL_WINDOW_MIN minutes.

    Algorithm
    ---------
    For each card, compare each transaction's country with the previous
    transaction's country.  If the country differs and the time gap is
    within the threshold, both transactions are flagged.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by (card_id, timestamp).

    Returns
    -------
    pd.Series[bool]
        Boolean mask; True where the rule fires.
    """
    df = df.copy()
    df["_prev_ts"] = df.groupby("card_id")["timestamp"].shift(1)
    df["_prev_country"] = df.groupby("card_id")["country"].shift(1)

    window = pd.Timedelta(minutes=IMPOSSIBLE_TRAVEL_WINDOW_MIN)

    time_ok = (df["timestamp"] - df["_prev_ts"]) <= window
    country_diff = df["country"] != df["_prev_country"]

    current_flagged = (time_ok & country_diff).fillna(False)

    # Also flag the *previous* transaction in each such pair
    df["_flagged"] = current_flagged
    df["_prev_flagged"] = df.groupby("card_id")["_flagged"].shift(-1).fillna(False)

    mask = (current_flagged | df["_prev_flagged"]).fillna(False)
    return mask


# ─────────────────────────────────────────────
#  Score Aggregator
# ─────────────────────────────────────────────

def apply_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all fraud-detection rules to a cleaned transactions DataFrame.

    Adds three new columns to the DataFrame:
    - ``fraud_score``  (int)  : cumulative risk score (0–125)
    - ``fraud_reason`` (str)  : human-readable explanation of triggered rules
    - ``is_fraud``     (bool) : True if fraud_score >= FRAUD_THRESHOLD_SCORE

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions DataFrame (output of utils.load_and_clean).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with fraud annotation columns appended.
    """
    df = df.copy()

    # Evaluate each rule
    high_amount_mask    = rule_high_amount(df)
    rapid_mask          = rule_rapid_succession(df)
    travel_mask         = rule_impossible_travel(df)

    # Build score and reason vectors
    scores  = np.zeros(len(df), dtype=int)
    reasons = [[] for _ in range(len(df))]

    scores[high_amount_mask.values]  += SCORE_HIGH_AMOUNT
    scores[rapid_mask.values]        += SCORE_RAPID_SUCCESSION
    scores[travel_mask.values]       += SCORE_IMPOSSIBLE_TRAVEL

    for i, flag in enumerate(high_amount_mask):
        if flag:
            reasons[i].append(
                f"High amount: ₹{df.iloc[i]['amount']:,.2f} exceeds ₹{HIGH_AMOUNT_THRESHOLD:,}"
            )
    for i, flag in enumerate(rapid_mask):
        if flag:
            reasons[i].append(
                f"Rapid succession: multiple transactions within {RAPID_SUCCESSION_WINDOW_SEC}s"
            )
    for i, flag in enumerate(travel_mask):
        if flag:
            reasons[i].append(
                f"Impossible travel: different countries within {IMPOSSIBLE_TRAVEL_WINDOW_MIN} min"
            )

    df["fraud_score"]  = scores
    df["fraud_reason"] = [" | ".join(r) if r else "None" for r in reasons]
    df["is_fraud"]     = df["fraud_score"] >= FRAUD_THRESHOLD_SCORE

    return df


# ─────────────────────────────────────────────
#  Single-Transaction Helper
# ─────────────────────────────────────────────

def check_single_transaction(
    single_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate a single new transaction against existing card history.

    The new transaction is temporarily appended to the history so that
    rapid-succession and impossible-travel rules can reference prior activity.

    Parameters
    ----------
    single_df : pd.DataFrame
        One-row DataFrame representing the new transaction.
    history_df : pd.DataFrame
        Full processed transaction history (already cleaned).

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with fraud annotation columns.
    """
    card_id = single_df.iloc[0]["card_id"]

    # Filter history to the same card for context
    card_history = history_df[history_df["card_id"] == card_id].copy()

    # Drop annotation columns if present
    drop_cols = [c for c in ["fraud_score", "fraud_reason", "is_fraud"] if c in card_history.columns]
    card_history.drop(columns=drop_cols, inplace=True)

    combined = pd.concat([card_history, single_df], ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    result = apply_all_rules(combined)

    # Return only the row corresponding to the new transaction
    tid = single_df.iloc[0]["transaction_id"]
    return result[result["transaction_id"] == tid].reset_index(drop=True)
