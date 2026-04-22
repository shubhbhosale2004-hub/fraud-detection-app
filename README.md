# 🛡️ FraudGuard — Rule-Based Credit Card Fraud Detection

A production-quality fraud detection system with a modern Streamlit GUI.

---

## 📁 Project Structure

```
fraud_detection/
├── app.py              # Streamlit GUI (main entry point)
├── fraud_rules.py      # Rule-based fraud detection engine
├── utils.py            # Data preprocessing & synthetic data generation
├── requirements.txt    # Python dependencies
└── data/
    └── sample_transactions.csv   # Demo dataset (253 rows)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🧠 Fraud Detection Rules

| Rule | Trigger | Score |
|------|---------|-------|
| 💰 High Amount | Transaction > ₹50,000 | +40 |
| ⚡ Rapid Succession | Same card, 2+ transactions within 60 seconds | +35 |
| ✈️ Impossible Travel | Same card, different countries within 60 minutes | +50 |

A transaction is flagged as **fraudulent** when `fraud_score ≥ 30`.

---

## 📊 App Sections

| Tab | Description |
|-----|-------------|
| 📊 Dashboard | All transactions with filters (risk score, country, fraud-only) |
| 🚨 Fraud Alerts | Flagged transactions + rule breakdown statistics |
| 📈 Visual Analytics | Amount distribution, fraud vs clean, fraud by country charts |
| 🔎 Manual Checker | Enter a single transaction and get instant fraud verdict |

---

## 📂 CSV Format

Upload your own CSV with these columns:

```
transaction_id, card_id, amount, timestamp, country
```

Example:
```csv
transaction_id,card_id,amount,timestamp,country
TXN001,CARD0001,5000.00,2024-01-15 10:30:00,India
TXN002,CARD0001,75000.00,2024-01-15 10:30:45,India
```
