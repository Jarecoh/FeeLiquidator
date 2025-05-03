````markdown
# FeeLiquidator

FeeLiquidator is a Python-based GUI tool that rapidly executes small trades in a loop to quantify real-world slippage and fees on Coinbase. It supports both market and post-only limit orders.

## ✅ Features

- Live trading through Coinbase Advanced Trade API  
- Toggle between market and post-only limit orders  
- Set maximum trade amount and number of iterations  
- Displays both expected and real slippage  
- Permanent logging of each trade to `logs/trade.log`  
- Clean, readable Tkinter interface

---

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/jarecoh/FeeLiquidator.git
cd FeeLiquidator
````

### 2. Set up your API key

Create a file at:

```
vars/trade_api_key.json
```

With this format:

```json
{
  "name": "YOUR_API_KEY",
  "privateKey": "YOUR_API_SECRET"
}
```

⚠️ **Your API key must have "trade" permissions enabled** in the Coinbase Advanced Trade API settings.

### 3. Install dependencies

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

*(Or manually install: `tk`, `numpy`, `coinbase-advanced-trade`)*

### 4. Launch the app

```bash
python FeeLiquidator.py
```

---

## 📁 Output

* All trades are recorded to:

  ```
  logs/trade.log
  ```

* Log includes:

  * Order direction (BUY/SELL)
  * Fill confirmations
  * Iteration stats
  * Slippage summary

---

## 🛑 Safety Notice

This script **places live orders** on your Coinbase account. Use small amounts and test in a sandbox if available.

---

## 📄 License

MIT – Use at your own risk.
