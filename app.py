"""
app.py — Streamlit GUI for the Rule-Based Credit Card Fraud Detection System.

Run with:
    streamlit run app.py
"""

import io
import random
import string
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import streamlit as st

from fraud_rules import (
    FRAUD_THRESHOLD_SCORE,
    HIGH_AMOUNT_THRESHOLD,
    IMPOSSIBLE_TRAVEL_WINDOW_MIN,
    RAPID_SUCCESSION_WINDOW_SEC,
    apply_all_rules,
    check_single_transaction,
)
from utils import (
    COUNTRIES,
    build_single_transaction,
    generate_synthetic_data,
    load_and_clean,
    load_from_upload,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page Config & Global Style
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FraudGuard · AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a polished dark-themed look
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    /* ── Root variables ── */
    :root {
        --bg:        #0d1117;
        --surface:   #161b22;
        --border:    #30363d;
        --accent:    #e84545;
        --accent2:   #f5a623;
        --safe:      #2ea043;
        --text:      #e6edf3;
        --muted:     #8b949e;
        --mono:      'IBM Plex Mono', monospace;
        --sans:      'Sora', sans-serif;
    }

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: var(--sans);
        background-color: var(--bg);
        color: var(--text);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 22px;
    }
    div[data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: var(--mono);
        font-size: 2rem !important;
        font-weight: 600;
    }

    /* ── DataFrames ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--accent);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-family: var(--mono);
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.5rem 1.6rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Download button (override) ── */
    .stDownloadButton > button {
        background: #238636;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-family: var(--mono);
        font-weight: 600;
        padding: 0.5rem 1.6rem;
    }
    .stDownloadButton > button:hover { opacity: 0.85; }

    /* ── Section headers ── */
    .section-header {
        font-family: var(--mono);
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--muted);
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* ── Fraud badge ── */
    .badge-fraud  { color: var(--accent);  font-weight: 700; font-family: var(--mono); }
    .badge-safe   { color: var(--safe);    font-weight: 700; font-family: var(--mono); }

    /* ── Alert boxes ── */
    .alert-fraud {
        background: rgba(232,69,69,0.10);
        border: 1px solid var(--accent);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    .alert-safe {
        background: rgba(46,160,67,0.10);
        border: 1px solid var(--safe);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-family: var(--mono);
        font-size: 0.82rem;
        letter-spacing: 0.06em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes for download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _risk_label(score: int) -> str:
    """Return a coloured HTML risk label for a given fraud score."""
    if score == 0:
        return '<span class="badge-safe">✔ Clean</span>'
    if score < FRAUD_THRESHOLD_SCORE:
        return f'<span style="color:#f5a623;font-weight:700;font-family:monospace">⚠ Low-Risk ({score})</span>'
    return f'<span class="badge-fraud">✘ FRAUD ({score})</span>'


@st.cache_data(show_spinner=False)
def _cached_synthetic() -> pd.DataFrame:
    """Cache the synthetic dataset so it doesn't regenerate on every rerun."""
    return generate_synthetic_data(n_rows=220)


def _rand_txn_id() -> str:
    return "TXN" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


# ─────────────────────────────────────────────────────────────────────────────
#  Matplotlib theme
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#161b22",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#8b949e",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — Data Source
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.82rem'>Rule-Based Credit Card Fraud Detection</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Data Source")
    uploaded = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        help="Required columns: transaction_id, card_id, amount, timestamp, country",
    )

    use_synthetic = st.checkbox("Use synthetic demo data", value=(uploaded is None))

    st.divider()
    st.markdown("### Rule Thresholds")
    st.markdown(
        f"""
        <div style='font-size:0.82rem;line-height:1.8;font-family:monospace;color:#8b949e'>
        💰 High amount   : <b style='color:#e6edf3'>₹{HIGH_AMOUNT_THRESHOLD:,}</b><br>
        ⚡ Rapid window  : <b style='color:#e6edf3'>{RAPID_SUCCESSION_WINDOW_SEC}s</b><br>
        ✈️ Travel window  : <b style='color:#e6edf3'>{IMPOSSIBLE_TRAVEL_WINDOW_MIN} min</b><br>
        🎯 Fraud score   : <b style='color:#e84545'>≥ {FRAUD_THRESHOLD_SCORE}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("v1.0 · Built with Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
#  Load & Process Data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="🔍 Processing transactions …")
def _process(df_raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = load_and_clean(df_raw)
    return apply_all_rules(cleaned)


if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        df = _process(raw_df)
        data_source_label = f"📂 Uploaded: `{uploaded.name}`"
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
elif use_synthetic:
    raw_df = _cached_synthetic()
    df = _process(raw_df)
    data_source_label = "🤖 Synthetic demo dataset (220 transactions)"
else:
    st.info("⬆️ Upload a CSV or enable synthetic demo data from the sidebar.")
    st.stop()

fraud_df  = df[df["is_fraud"]].copy()
clean_df  = df[~df["is_fraud"]].copy()
fraud_pct = len(fraud_df) / len(df) * 100 if len(df) else 0


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='font-family:"IBM Plex Mono",monospace;font-size:2.1rem;
               letter-spacing:-0.02em;margin-bottom:2px'>
        🛡️ FraudGuard
        <span style='font-size:1rem;color:#8b949e;font-weight:400;
                     letter-spacing:0.05em'> · Credit Card Fraud Detection</span>
    </h1>
    """,
    unsafe_allow_html=True,
)
st.caption(data_source_label)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  KPI Row
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Transactions", f"{len(df):,}")
k2.metric("Fraud Detected",     f"{len(fraud_df):,}", delta=f"{fraud_pct:.1f}% of total", delta_color="inverse")
k3.metric("Clean Transactions", f"{len(clean_df):,}")
k4.metric("Unique Cards",       f"{df['card_id'].nunique():,}")
k5.metric("Countries Seen",     f"{df['country'].nunique():,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_dash, tab_alerts, tab_viz, tab_manual = st.tabs([
    "📊  Dashboard",
    "🚨  Fraud Alerts",
    "📈  Visual Analytics",
    "🔎  Manual Checker",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    st.markdown('<p class="section-header">All Transactions</p>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns([1, 2, 2])
    with col_f1:
        show_fraud_only = st.checkbox("High-risk only", value=False)
    with col_f2:
        min_score = st.slider("Minimum fraud score", 0, 125, 0, step=5)
    with col_f3:
        countries_sel = st.multiselect(
            "Filter by country",
            options=sorted(df["country"].unique()),
            default=[],
        )

    view = df.copy()
    if show_fraud_only:
        view = view[view["is_fraud"]]
    if min_score > 0:
        view = view[view["fraud_score"] >= min_score]
    if countries_sel:
        view = view[view["country"].isin(countries_sel)]

    # Display
    display_cols = [
        "transaction_id", "card_id", "amount", "timestamp",
        "country", "fraud_score", "fraud_reason", "is_fraud",
    ]

    def _style_fraud(row):
        if row["is_fraud"]:
            return ["background-color: rgba(232,69,69,0.10)"] * len(row)
        return [""] * len(row)

    styled = (
        view[display_cols]
        .style
        .apply(_style_fraud, axis=1)
        .format({"amount": "₹{:,.2f}", "fraud_score": "{:d}"})
        .set_properties(**{"font-family": "monospace", "font-size": "0.82rem"})
    )

    st.dataframe(styled, use_container_width=True, height=440)
    st.caption(f"Showing {len(view):,} of {len(df):,} transactions")

    st.download_button(
        label="⬇️  Export filtered results as CSV",
        data=_df_to_csv_bytes(view[display_cols]),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Fraud Alerts
# ══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown('<p class="section-header">Flagged Fraudulent Transactions</p>', unsafe_allow_html=True)

    if fraud_df.empty:
        st.success("✅ No fraudulent transactions detected in this dataset.")
    else:
        st.markdown(
            f"<p style='color:#e84545;font-family:monospace;font-size:0.9rem'>"
            f"⚠️  <b>{len(fraud_df)}</b> transaction(s) flagged as fraudulent "
            f"({fraud_pct:.1f}% of total)</p>",
            unsafe_allow_html=True,
        )

        fraud_display = fraud_df[[
            "transaction_id", "card_id", "amount", "timestamp",
            "country", "fraud_score", "fraud_reason",
        ]].copy()

        fraud_display_styled = (
            fraud_display.style
            .background_gradient(subset=["fraud_score"], cmap="Reds", vmin=0, vmax=125)
            .format({"amount": "₹{:,.2f}", "fraud_score": "{:d}"})
            .set_properties(**{"font-family": "monospace", "font-size": "0.82rem"})
        )

        st.dataframe(fraud_display_styled, use_container_width=True, height=420)

        # Rule breakdown
        st.divider()
        st.markdown('<p class="section-header">Rule Trigger Breakdown</p>', unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        high_n   = fraud_df["fraud_reason"].str.contains("High amount").sum()
        rapid_n  = fraud_df["fraud_reason"].str.contains("Rapid succession").sum()
        travel_n = fraud_df["fraud_reason"].str.contains("Impossible travel").sum()

        b1.metric("💰 High Amount", high_n)
        b2.metric("⚡ Rapid Succession", rapid_n)
        b3.metric("✈️ Impossible Travel", travel_n)

        st.divider()
        st.download_button(
            label="⬇️  Export fraud alerts as CSV",
            data=_df_to_csv_bytes(fraud_display),
            file_name="fraud_alerts.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Visual Analytics
# ══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.markdown('<p class="section-header">Visual Analytics</p>', unsafe_allow_html=True)

    # ── Row 1: Amount distribution + Fraud vs Clean
    row1_c1, row1_c2 = st.columns(2)

    # Chart 1 — Amount Distribution
    with row1_c1:
        st.markdown("**Transaction Amount Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.set_facecolor("#161b22")
        fig1.patch.set_facecolor("#161b22")

        amounts_clean = clean_df["amount"].clip(upper=200_000)
        amounts_fraud = fraud_df["amount"].clip(upper=200_000)

        ax1.hist(amounts_clean, bins=40, color="#2ea043", alpha=0.7, label="Clean")
        ax1.hist(amounts_fraud, bins=40, color="#e84545", alpha=0.85, label="Fraud")
        ax1.axvline(HIGH_AMOUNT_THRESHOLD, color="#f5a623", lw=1.5, ls="--",
                    label=f"₹{HIGH_AMOUNT_THRESHOLD:,} threshold")
        ax1.set_xlabel("Amount (INR)")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=8)
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
        ax1.grid(True, axis="y")
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    # Chart 2 — Fraud vs Non-Fraud
    with row1_c2:
        st.markdown("**Fraud vs Non-Fraud Count**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.set_facecolor("#161b22")
        fig2.patch.set_facecolor("#161b22")

        categories = ["Clean", "Fraudulent"]
        counts     = [len(clean_df), len(fraud_df)]
        colors     = ["#2ea043", "#e84545"]
        bars = ax2.bar(categories, counts, color=colors, width=0.5, edgecolor="#30363d", linewidth=0.8)

        for bar, count in zip(bars, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(count),
                ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                fontfamily="monospace",
            )
        ax2.set_ylabel("Transaction Count")
        ax2.set_ylim(0, max(counts) * 1.18)
        ax2.grid(True, axis="y")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Row 2: Fraud by Country
    st.markdown("**Fraud Occurrences by Country**")
    fig3, ax3 = plt.subplots(figsize=(12, 3.8))
    ax3.set_facecolor("#161b22")
    fig3.patch.set_facecolor("#161b22")

    country_counts = (
        fraud_df.groupby("country")
        .size()
        .reset_index(name="fraud_count")
        .sort_values("fraud_count", ascending=False)
    )

    if country_counts.empty:
        ax3.text(0.5, 0.5, "No fraud data", ha="center", va="center", transform=ax3.transAxes)
    else:
        palette = sns.color_palette("Reds_r", n_colors=len(country_counts))
        bars3 = ax3.bar(
            country_counts["country"],
            country_counts["fraud_count"],
            color=palette,
            edgecolor="#30363d",
            linewidth=0.8,
        )
        for bar, val in zip(bars3, country_counts["fraud_count"]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(val),
                ha="center", va="bottom",
                fontsize=9, fontfamily="monospace",
            )
        ax3.set_ylabel("Fraud Count")
        ax3.set_xlabel("Country")
        ax3.grid(True, axis="y")

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── Row 3: Fraud score distribution
    st.markdown("**Fraud Score Distribution (Flagged Transactions)**")
    if not fraud_df.empty:
        fig4, ax4 = plt.subplots(figsize=(12, 3.2))
        ax4.set_facecolor("#161b22")
        fig4.patch.set_facecolor("#161b22")
        ax4.hist(fraud_df["fraud_score"], bins=20, color="#e84545", edgecolor="#30363d")
        ax4.axvline(FRAUD_THRESHOLD_SCORE, color="#f5a623", lw=1.5, ls="--",
                    label=f"Threshold ({FRAUD_THRESHOLD_SCORE})")
        ax4.set_xlabel("Fraud Score")
        ax4.set_ylabel("Count")
        ax4.legend(fontsize=8)
        ax4.grid(True, axis="y")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
    else:
        st.info("No fraud data to plot.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Manual Transaction Checker
# ══════════════════════════════════════════════════════════════════════════════

with tab_manual:
    st.markdown('<p class="section-header">Manual Transaction Checker</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:0.88rem'>"
        "Enter a new transaction below. It will be evaluated against the existing card history "
        "in the loaded dataset.</p>",
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        m_txn_id  = st.text_input("Transaction ID", value=_rand_txn_id())
        m_card_id = st.text_input("Card ID", value="CARD0001")
        m_amount  = st.number_input("Amount (₹ INR)", min_value=0.0, value=5000.0, step=100.0)

    with m2:
        m_country   = st.selectbox("Country", options=sorted(COUNTRIES))
        m_timestamp = st.date_input("Date", value=datetime(2024, 2, 1))
        m_time      = st.time_input("Time", value=datetime(2024, 2, 1, 12, 0).time())

    check_btn = st.button("🔎  Check Transaction")

    if check_btn:
        ts = datetime.combine(m_timestamp, m_time)
        single = build_single_transaction(m_txn_id, m_card_id, m_amount, ts, m_country)

        result = check_single_transaction(single, df)

        if result.empty:
            st.warning("Could not evaluate transaction — please verify the inputs.")
        else:
            row = result.iloc[0]
            score  = int(row["fraud_score"])
            reason = row["fraud_reason"]
            fraud  = bool(row["is_fraud"])

            if fraud:
                st.markdown(
                    f"""
                    <div class="alert-fraud">
                        <b style='color:#e84545;font-size:1.1rem'>🚨 FRAUD DETECTED</b><br>
                        <span style='font-family:monospace'>
                        Score : <b>{score}</b> / 125<br>
                        Reason: {reason}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="alert-safe">
                        <b style='color:#2ea043;font-size:1.1rem'>✅ Transaction Appears Clean</b><br>
                        <span style='font-family:monospace'>
                        Score : <b>{score}</b> / 125<br>
                        Reason: {reason}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("**Full Result**")
            st.dataframe(
                result.style.format({"amount": "₹{:,.2f}"}),
                use_container_width=True,
            )
