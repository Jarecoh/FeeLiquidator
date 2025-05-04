# FeeLiquidator

**FeeLiquidator** is a Python-based GUI tool that rapidly executes small trades in a loop to quantify real-world slippage and fees on **Coinbase**. It supports both **market** and **post-only limit** orders, automatically hugging the spread with the best possible prices.

This drives up your market volume efficiently to help you **qualify for lower fee tiers**.

---

## ✅ Features

- 💸 Live trading via Coinbase Advanced Trade API  
- 🔁 Repeats trades automatically using market or tight limit orders  
- 📉 Uses post-only logic to avoid taker fees (with price retry logic)  
- 🧮 Tracks **real slippage** vs **estimated fees**  
- 🧾 Logs each trade in `logs/trade.log` with clear timestamps  
- 🖥️ Clean, visual Tkinter interface with toggle buttons and real-time labels  
- 🔄 Re-attempts unfilled SELL orders with tighter prices before falling back to market  
- 🧠 Prevents over-trading by auto-adjusting for fees and wallet balance  
- 🧰 Compatible with all Coinbase spot pairs, defaults to BTC-USD

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jarecoh/FeeLiquidator.git
cd FeeLiquidator
```

### 2. Set up API Credentials

Create a file named:

```
vars/trade_api_key.json
```

With the following contents:

```json
{
  "name": "YOUR_API_KEY",
  "privateKey": "YOUR_API_SECRET"
}
```

> ⚠️ Your API key must have **"trade"** permissions enabled in the **Coinbase Advanced Trade API**.  
> 🔑 You can create a new Coinbase Advanced API Key here:  
> 👉 [coinbase.com/settings/api](https://www.coinbase.com/settings/api)

---

### 3. Install Dependencies

Make sure you're using **Python 3.10 or higher**, then run:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is available, install manually:

```bash
pip install coinbase-advanced-trade numpy
```

> `tkinter` is built-in with most Python installations.

---

### 4. Launch the App

```bash
python FeeLiquidator.py
```

---

## 🖱️ Using the GUI

1. **Max Trade Amount**: Set the dollar amount used per trade pair (BUY + SELL).
2. **Iterations**: Choose how many trade pairs to run.
3. **Order Type Toggle**: Click 📉 (Limit) or ⚡ (Market) to switch.
4. **Fee Tier Dropdown**: Select your current Coinbase fee tier for accurate slippage estimates.
5. **“Let’s Go”**: Confirm total cost estimate, then launch the loop.
6. **Stop**: Halts all activity and cancels any open orders.
7. **Real Slippage**: Shows actual USD lost/gained including fees.
8. **Spot Volume**: See your Coinbase spot volume (30-day rolling).

---

## 📁 Output & Logs

All important activity is permanently logged to:

```
logs/trade.log
```

Includes:

- BUY / SELL orders and prices  
- Fill confirmations  
- Cancel events  
- Net gains/losses  
- Runtime summary  

---

## 🔄 Reposting Logic

For **limit orders**, FeeLiquidator:

- Posts hugging the bid/ask with a safe tick away  
- Retries limit price adjustments if **too close**  
- Cancels and **re-posts unfilled SELL orders** for up to 3 timeouts  
- Falls back to **market SELL** only if no fills after 3 attempts

---

## 🛑 Important Safety Notes

- This script places **live trades** on your Coinbase account.
- Start with **tiny amounts** like $1 to observe behavior.
- Consider testing in **sandbox or throwaway accounts** first.
- Always monitor open orders and wallet balances.

---

## 📄 License

MIT – Provided as-is. Use responsibly at your own risk.

---

Written by Jared Cohn with one assload of help from ChatGPT
