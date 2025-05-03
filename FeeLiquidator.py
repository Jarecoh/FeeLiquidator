import tkinter as tk
import threading
import time
import os
import json
from datetime import datetime, timezone, timedelta
from tkinter import scrolledtext, ttk
import queue
import uuid
import sys
import numpy as np
from threading import Event
from coinbase.rest import RESTClient

def load_vars_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get('name'), data.get('privateKey')
    except Exception:
        sys.exit("Error: Missing or invalid vars.json file.")

log_queue = queue.Queue()
use_market_order = False

# Load API credentials
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, 'vars', 'trade_api_key.json')
coinbase_fee_path = os.path.join(script_dir, 'vars', 'coinbase_fee_tiers.json')
api_key, api_secret = load_vars_from_json(json_file_path)

# Ensure logs directory exists
log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
trade_log_path = os.path.join(log_dir, 'trade.log')

client = RESTClient(api_key=api_key, api_secret=api_secret)

# Load Coinbase fee tier data
with open(coinbase_fee_path, 'r') as f:
    coinbase_fee_tiers = json.load(f)
selected_fee_tier_name = "Intro 1"
default_fee = coinbase_fee_tiers.get(selected_fee_tier_name, {"taker": 0.012, "maker": 0.006})

product_id = "BTC-USD"
running = False
volume_added = 0.0
iterations_done = 0
active_buy_id = None

def write_trade_log(message):
    with open(trade_log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def log(msg, tag="info"):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    color = {
        "info": "white",
        "warn": "orange",
        "error": "red",
        "success": "lime",
        "debug": "gray" 
    }.get(tag, "white")
    formatted_msg = f"{timestamp} {msg}"
    log_queue.put((formatted_msg, color))
    print(formatted_msg)

    # Write key trade events to persistent log
    if any(kw in msg for kw in [
        "BUY", "SELL",
        "[FILL]", "Starting iteration",
        "Iteration", "💸 Net USD loss", "✅ Net USD gain"
    ]):
        write_trade_log(formatted_msg)

def get_wallet_balance(currency='USD'):
    accounts = client.get_accounts()
    for acct in accounts.accounts:
        if acct.currency == currency:
            return float(acct.available_balance['value'])
    return 0.0

def generate_order_id():
    return f"{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:5]}-{uuid.uuid4().hex[:6]}"

def get_current_price_fallback(client, product_id):
    try:
        candles = get_historical_data(client, market=product_id, granularity="ONE_MINUTE", limit=2)
        if candles and len(candles) > 0:
            return float(candles[-1]["close"])
        raise ValueError("No candles returned")
    except Exception as e:
        log(f"Failed to get current price for {product_id}: {e}", "error")
        return None

def calculate_rsi(prices, period=20):
    if len(prices) < period + 1:
        raise ValueError("Not enough data to calculate RSI")

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi = [None] * period
    rsi.append(100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss)))

    for i in range(period, len(prices) - 1):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        rsi.append(100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss)))

    return rsi[-1]

def calculate_vwma(prices, volumes, period=40):
    vwma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        pv_sum = np.sum(np.array(prices[i - period + 1:i + 1]) * np.array(volumes[i - period + 1:i + 1]))
        vol_sum = np.sum(volumes[i - period + 1:i + 1])
        vwma[i] = 0 if vol_sum == 0 else pv_sum / vol_sum
    return vwma

def get_price_vwma_rsi(client, market="BTC-USD", granularity="ONE_MINUTE", rsi_period=20, vwma_period=40):
    candles = get_historical_data(client, market, granularity=granularity, limit=max(rsi_period, vwma_period) + 1)
    if not candles or len(candles) <= max(rsi_period, vwma_period):
        return None, None, None

    prices = [float(c["close"]) for c in candles]
    volumes = [float(c["volume"]) for c in candles]

    latest_price = prices[-1]
    latest_vwma = calculate_vwma(prices, volumes, period=vwma_period)[-1]
    latest_rsi = calculate_rsi(prices, period=rsi_period)

    return latest_price, latest_vwma, latest_rsi

def get_precision_fallback(product_id):
    try:
        response = client.get_product(product_id=product_id)
        increment = response.base_increment
        if "." in increment:
            return len(increment.split(".")[1].rstrip("0"))
        return 0
    except Exception as e:
        log(f"Failed to fetch precision for {product_id}: {e}", "error")
        return 4

def wait_for_order_fill(order_id, timeout=None, poll_interval=2, should_stop_event=None):
    log(f"[WAIT] Waiting for order {order_id} to fill...", "status")
    start = time.time()
    while True:
        if should_stop_event and should_stop_event.is_set():
            log(f"[WAIT] Stop event detected. Exiting wait early.", "warn")
            return False
        try:
            order_response = client.get_order(order_id=order_id)
            order = order_response.order
            if order.status == "FILLED":
                log(f"[FILL] Order {order_id} fully filled.", "success")
                return True
        except Exception as e:
            log(f"[ERROR] Exception during order status check: {e}", "error")

        if timeout is not None and (time.time() - start) > timeout:
            log(f"[TIMEOUT] Order {order_id} not filled after {timeout} seconds.", "warn")
            return False
        time.sleep(poll_interval)

def update_status(text, status_var, status_label, bg_color="#444", fg_color="white"):
    status_var.set(text)

    # Style by status type
    if "Trading" in text:
        status_label.config(bg="#006400", fg="white")
    elif "Complete" in text:
        status_label.config(bg="#1e1e1e", fg="#00CED1")
    else:
        status_label.config(bg=bg_color, fg=fg_color)

def cancel_order(order_id):
    try:
        order = client.get_order(order_id=order_id).order
        if order.status != "OPEN":
            log(f"⚠️ Order is not open (status: {order.status}) — skipping cancel.", "warn")
            return False

        response = client.cancel_orders(order_ids=[order_id]).to_dict()
        results = response.get("results", [])
        if results and results[0].get("success"):
            return True
        else:
            reason = results[0].get("failure_reason") if results else "Unknown"
            log(f"❌ Cancel failed. Reason: {reason}", "error")
            return False
    except Exception as e:
        log(f"[ERROR] Exception during cancel attempt: {e}", "error")
        return False

def update_wallet_label(label):
    balance = get_wallet_balance('USD')
    label.config(text=f"USD Wallet: ${balance:.2f}")

def update_slippage_label(label, iterations, amount, tier_name):
    try:
        tier = coinbase_fee_tiers.get(tier_name, default_fee)
        fee_pct = tier["taker"] if use_market_order else tier["maker"]

        # Do not round intermediate calculations with fudge factor
        raw_slippage = amount * fee_pct * 2 * iterations / 2.316

        label.config(text=f"Expected Slippage: ${raw_slippage:.6f}")
    except Exception as e:
        label.config(text="Expected Slippage: Error")
        log(f"Failed to update slippage: {e}", "error")

def get_historical_data(client, market="BTC-USD", granularity="ONE_MINUTE", limit=100):
    granularity_map = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "TWO_HOUR": 7200,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400
    }
    if granularity not in granularity_map:
        raise ValueError(f"Unsupported granularity: {granularity}")

    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=granularity_map[granularity] * limit)

    try:
        response = client.get_candles(
            product_id=market,
            granularity=granularity,
            start=int(start.timestamp()),
            end=int(now.timestamp())
        )
        candles = response["candles"]
        candles.sort(key=lambda c: int(c["start"]))
        return candles
    except Exception:
        return None

def place_limit_order(product_id, side, base_size, post_only=True, max_retries=200, limit_price=None):
    import json
    for attempt in range(max_retries):
        try:
            book = client.get_product_book(product_id=product_id, level=1)
            data = book.to_dict()
            if attempt == 0:
                log(f"[DEBUG] Raw book response: {json.dumps(data)[:1000]}...", "warn")  # limit to 1000 chars

            book_data = data.get("pricebook", {})
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])

            if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
                log(f"[Retry {attempt + 1}] Order book data invalid or empty — waiting...", "warn")
                time.sleep(0.1)
                continue

            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])

            # Hug spread
            spread = best_ask - best_bid
            if spread < 0.01:
                spread = 0.01  # fallback to prevent zero spread

            spread_multiplier = 0.2 + (0.05 * attempt)
            price_offset = spread * spread_multiplier

            if limit_price is not None:
                price = limit_price
            else:
                if side == "BUY":
                    price = round(best_ask - price_offset, 2)
                else:
                    price = round(best_bid + price_offset, 2)

            # Reassess if price matches market (post-only would fail)
            if (side == "BUY" and price >= best_ask) or (side == "SELL" and price <= best_bid):
                log(f"[Retry {attempt + 1}] Limit price would match market — reassessing...", "warn")
                time.sleep(0.1)
                continue

            order_id = generate_order_id()
            log(f"[SPREAD] Best bid: {best_bid}, Best ask: {best_ask}, Spread: {spread:.2f}", "status")
            log(f"{side} {base_size} @ {price} (post_only={post_only}) | ID: {order_id}", "status")

            response = client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side=side,
                order_configuration={
                    "limit_limit_gtc": {
                        "base_size": str(base_size),
                        "limit_price": str(price),
                        "post_only": post_only
                    }
                }
            ).to_dict()

            if "success_response" in response:
                return response

            reason = response.get("error_response", {}).get("preview_failure_reason", "")
            if "would execute" in reason.lower() or "too close" in reason.lower():
                log(f"[Retry {attempt + 1}] Price too close to market — adjusting...", "warn")
                time.sleep(0.2)
                continue

            log(f"[FAILURE] Order rejected: {reason}", "error")
            return response

        except Exception as e:
            log(f"[ERROR] Exception placing order: {repr(e)}", "error")
            time.sleep(0.1)

    log("❌ Max retries reached without successful post-only order.", "error")
    return {
        "success": False,
        "error_response": {
            "error": "RETRY_LIMIT",
            "message": "Post-only retry limit exceeded",
            "preview_failure_reason": "TOO_CLOSE_RETRIES_EXCEEDED"
        }
    }

def restore_controls(amount_limit_entry, iteration_combo, start_btn, status_var, status_label, volume_label, iter_label):
    amount_limit_entry.config(state="normal")
    iteration_combo.config(state="readonly")
    start_btn.config(state="normal")
    update_status("✅ Status: Complete", status_var, status_label)

    volume_label.config(text=f"Market Volume Added: ${volume_added:.2f}")
    iter_label.config(text=f"Iterations Completed: {iterations_done}")
    stop_event.clear()

def run_feeliquidator(amount_limit_entry, iteration_var, wallet_label, volume_label, iter_label, status_var, status_label, start_btn, iteration_combo, real_slippage_label):
    global running, volume_added, iterations_done, active_buy_id
    start_usd_balance = get_wallet_balance('USD')
    running = True
    iterations_done = 0
    active_buy_id = None

    amount_limit_entry.config(state="disabled")
    iteration_combo.config(state="disabled")
    start_btn.config(state="disabled")
    update_status("🟢 Status: Trading ⏳", status_var, status_label)

    try:
        max_amount = float(amount_limit_entry.get())
        target_iterations = int(iteration_var.get())
        log(f"Max trade amount: ${max_amount}, Target iterations: {target_iterations}", "status")
    except ValueError:
        log("Invalid amount or iteration input.", "error")
        return

    total_spent = 0.0

    while running and iterations_done < target_iterations:
        if stop_event.is_set():
            log("Stop requested — terminating loop.", "warn")
            break

        log(f"Starting iteration {iterations_done + 1} of {target_iterations}", "status")
        
        usd_balance = get_wallet_balance('USD')
        log(f"USD Balance: ${usd_balance:.2f}", "status")
        if usd_balance < max_amount:
            log("⚠️ Not enough USD for next cycle.", "warning")
            break

        market_price = get_current_price_fallback(client, product_id)
        if market_price is None:
            log("Skipping cycle: No market price.", "warning")
            break

        precision = get_precision_fallback(product_id)
        buy_price = round(market_price * 0.999, 2 if market_price >= 1 else 4)
        base_size = round(max_amount / buy_price, precision)

        if use_market_order:
            log(f"BUY {base_size} {product_id} @ market", "status")
            buy_order = client.create_order(
                client_order_id=generate_order_id(),
                product_id=product_id,
                side="BUY",
                order_configuration={
                    "market_market_ioc": {
                        "quote_size": str(max_amount)
                    }
                }
            ).to_dict()
        else:
            log(f"BUY {base_size} {product_id} @ {buy_price} (post-only)", "status")
            buy_order = place_limit_order(
                product_id=product_id,
                side="BUY",
                base_size=base_size,
                post_only=True
            )

        if "success_response" not in buy_order:
            error = buy_order.get("error_response", {})
            reason = error.get("preview_failure_reason") or error.get("message") or "Unknown"
            log(f"Buy order failed: {reason}", "error")
            log(f"↳ Full error: {error}", "error")
            restore_controls(amount_limit_entry, iteration_combo, start_btn, status_var, status_label, volume_label, iter_label)
            break

        active_buy_id = buy_order["success_response"]["order_id"]
        log(f"Waiting for BUY order to fill (ID: {active_buy_id})", "status")
        filled = wait_for_order_fill(active_buy_id, should_stop_event=stop_event)
        if not filled:
            log("Cancelling BUY order due to stop request...", "warn")
            success = cancel_order(active_buy_id)
            restore_controls(amount_limit_entry, iteration_combo, start_btn, status_var, status_label, volume_label, iter_label)
            if success:
                log("BUY order successfully cancelled.", "success")
            else:
                log("❌ Failed to cancel BUY order.", tag="error")
            break
        log(f"[FILLED] BUY order filled.", tag="success")

        active_buy_id = None

        if use_market_order:
            log(f"SELL {base_size} {product_id} @ market", "status")
            sell_order = client.create_order(
                client_order_id=generate_order_id(),
                product_id=product_id,
                side="SELL",
                order_configuration={
                    "market_market_ioc": {
                        "base_size": str(base_size)
                    }
                }
            ).to_dict()
        else:
            sell_price = round(market_price * 1.001, 2 if market_price >= 1 else 4)
            log(f"SELL {base_size} {product_id} @ {sell_price} (post-only)", "status")

            sell_order = place_limit_order(
                product_id=product_id,
                side="SELL",
                base_size=base_size,
                limit_price=sell_price,
                post_only=True
            )


        if "success_response" not in sell_order:
            error = sell_order.get("error_response", {})
            reason = error.get("preview_failure_reason") or error.get("message") or "Unknown"
            log(f"Sell order failed: {reason}", "error")
            log(f"↳ Full error: {error}", "error")
            break

        sell_id = sell_order["success_response"]["order_id"]
        log(f"Waiting for SELL order to fill (ID: {sell_id})", "status")
        wait_for_order_fill(sell_id)
        log(f"SELL order filled.", "status")
        if use_market_order:
            volume_added += base_size * market_price * 2
        else:
            volume_added += base_size * (buy_price + sell_price)

        iterations_done += 1
        total_spent += max_amount
        update_wallet_label(wallet_label)
        volume_label.config(text=f"Market Volume Added: ${volume_added:.2f}")
        iter_label.config(text=f"Iterations Completed: {iterations_done}")
        log(f"Iteration {iterations_done} complete. Total spent: ${total_spent:.2f}", "status")
        time.sleep(1)

    log("Liquidator stopped.", "status")
    end_usd_balance = get_wallet_balance('USD')
    usd_diff = start_usd_balance - end_usd_balance
    if usd_diff > 0:
        log(f"💸 Net USD loss from slippage/fees: ${usd_diff:.4f}", "warn")
    else:
        log(f"✅ Net USD gain: ${-usd_diff:.4f}", "success")

    restore_controls(amount_limit_entry, iteration_combo, start_btn, status_var, status_label, volume_label, iter_label)
    if not stop_event.is_set():
        real_slippage_label.config(text=f"Real Slippage: ${usd_diff:.4f}")
        log("✅ All iterations complete.", "success")

    update_wallet_label(wallet_label)

stop_event = Event()

def stop_feeliquidator():
    global running
    if not running:
        return
    log("Stop requested.", tag="warn")
    stop_event.set()
    running = False

def poll_log_queue(log_widget):
    while not log_queue.empty():
        msg, color = log_queue.get()
        log_widget.insert(tk.END, msg + "\n", color)
        log_widget.tag_config(color, foreground=color)
        log_widget.see(tk.END)
    log_widget.after(100, lambda: poll_log_queue(log_widget))

def bind_enter_key_to_buttons(root):
    def on_enter_key(event):
        widget = root.focus_get()
        if isinstance(widget, tk.Button):
            widget.invoke()
    root.bind("<Return>", on_enter_key)
    root.bind("<KP_Enter>", on_enter_key)  # Support numpad Enter

def start_gui():
    root = tk.Tk()
    status_var = tk.StringVar(value="🟡 Status: Idle")
    root.title("Fee Liquidator")
    root.configure(bg="#1e1e1e")
    root.geometry("800x600")
    root.resizable(True, True)

    main_frame = tk.Frame(root, bg="#1e1e1e")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Control Panel
    control_frame = tk.Frame(main_frame, bg="#1e1e1e", width=400)
    control_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)
    control_frame.pack_propagate(False)

    tk.Label(control_frame, text="Fee Liquidator", fg="white", bg="#1e1e1e", font=("Helvetica", 16)).pack(pady=10)

    wallet_label = tk.Label(control_frame, text="USD Wallet: $0.00", fg="cyan", bg="#1e1e1e", font=("Helvetica", 14))
    wallet_label.pack(pady=(0, 10))

    tk.Label(control_frame, text="Max Trade Amount ($):", fg="white", bg="#1e1e1e", font=("Helvetica", 13)).pack()
    amount_limit_entry = tk.Entry(control_frame, font=("Helvetica", 14), width=22, justify="center")
    amount_limit_entry.pack(padx=10, pady=5)

    tk.Label(control_frame, text="Iterations:", fg="white", bg="#1e1e1e").pack()
    iteration_var = tk.StringVar(value="1")
    iteration_combo = ttk.Combobox(control_frame, textvariable=iteration_var, width=20, font=("Helvetica", 12))
    iteration_combo['values'] = list(map(str, range(1, 1001)))
    iteration_combo.pack(padx=10, pady=5)

    volume_label = tk.Label(control_frame, text="Market Volume Added: $0.00", fg="lime", bg="#1e1e1e", font=("Helvetica", 12))
    volume_label.pack()

    iter_label = tk.Label(control_frame, text="Iterations Completed: 0", fg="orange", bg="#1e1e1e", font=("Helvetica", 12))
    iter_label.pack()

    # Status Indicator with improved visuals
    status_frame = tk.Frame(control_frame, bg="#1e1e1e")
    status_frame.pack(pady=5)

    status_label = tk.Label(
        status_frame,
        textvariable=status_var,
        fg="white",
        bg="#555",
        font=("Helvetica", 11, "bold"),  # larger font
        padx=14,
        pady=8,
        relief="ridge",
        bd=3,
        anchor="center",
        width=32
    )
    update_status("🟡 Status: Idle", status_var, status_label)
    status_label.pack()

    # Slippage Indicator
    slippage_label = tk.Label(control_frame, text="Expected Slippage: $0.00", fg="magenta", bg="#1e1e1e", font=("Helvetica", 12))
    slippage_label.pack(pady=(0, 10))
    real_slippage_label = tk.Label(control_frame, text="Real Slippage: $0.00", fg="cyan", bg="#1e1e1e", font=("Helvetica", 12))
    real_slippage_label.pack(pady=(0, 10))

    # Fee Tier Selector
    tk.Label(control_frame, text="Fee Tier:", fg="white", bg="#1e1e1e").pack()
    fee_tier_var = tk.StringVar(value=selected_fee_tier_name)
    fee_tier_combo = ttk.Combobox(control_frame, textvariable=fee_tier_var, width=20, font=("Helvetica", 12))
    fee_tier_combo['values'] = list(coinbase_fee_tiers.keys())
    fee_tier_combo.pack(padx=10, pady=5)

    def update_slippage(*args):
        try:
            amt = float(amount_limit_entry.get())
            iters = int(iteration_var.get())
            tier_name = fee_tier_var.get()
            update_slippage_label(slippage_label, iters, amt, tier_name)
        except Exception as e:
            log(f"Failed to update slippage: {e}", "error")

    def update_selected_fee_tier(*args):
        global selected_fee_tier_name
        selected_fee_tier_name = fee_tier_var.get()
        update_slippage()

    fee_tier_combo.bind("<<ComboboxSelected>>", update_selected_fee_tier)
    iteration_combo.bind("<<ComboboxSelected>>", lambda e: update_slippage())

    # Order type label and toggle
    order_type_frame = tk.Frame(control_frame, bg="#1e1e1e")
    order_type_frame.pack(pady=(10, 0))

    order_type_label = tk.Label(
        order_type_frame,
        text="Order Type: Limit",
        fg="yellow", bg="#1e1e1e",
        font=("Helvetica", 12)
    )
    order_type_label.pack(side="left")

    def toggle_order_type():
        global use_market_order
        use_market_order = not use_market_order
        order_type_label.config(text=f"Order Type: {'Market' if use_market_order else 'Limit'}")
        toggle_btn.config(
            text="🔄",
            bg="#1e1e1e" if use_market_order else "#333"
        )
        update_slippage()

    toggle_btn = tk.Button(
        order_type_frame,
        text="🔄",
        command=toggle_order_type,
        width=3,
        bg="#333",
        fg="white",
        relief="ridge",
        bd=1,
        font=("Helvetica", 10)
    )
    toggle_btn.pack(side="left", padx=(8, 0))

    # Start and stop buttons
    start_btn = tk.Button(control_frame, text="Let's Go", command=lambda: threading.Thread(
        target=run_feeliquidator,
        args=(amount_limit_entry, iteration_var, wallet_label, volume_label, iter_label, status_var, status_label, start_btn, iteration_combo, real_slippage_label),
        daemon=True
    ).start(), bg="green", fg="white", font=("Helvetica", 12))
    start_btn.pack(pady=5)

    stop_btn = tk.Button(control_frame, text="Stop", command=stop_feeliquidator, bg="red", fg="white", font=("Helvetica", 12))
    stop_btn.pack(pady=5)

    # Logging Panel
    log_frame = tk.Frame(main_frame, bg="#1e1e1e", width=400)
    log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_frame.pack_propagate(False)

    log_label = tk.Label(log_frame, text="Log Output", fg="white", bg="#1e1e1e", font=("Helvetica", 14))
    log_label.pack(anchor="w", padx=10, pady=(10, 0))

    log_text = scrolledtext.ScrolledText(
        log_frame, wrap=tk.CHAR, width=48, height=30,
        bg="black", fg="white", font=("Consolas", 10), borderwidth=0
    )
    log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    log_text.tag_config("error", foreground="red")
    log_text.tag_config("status", foreground="cyan")
    log_text.tag_config("warning", foreground="orange")
    log_text.tag_config("default", foreground="white")
    log_text.tag_config("debug", foreground="gray")

    # Now that all variables are defined, attach event listeners
    amount_limit_entry.bind("<KeyRelease>", lambda e: update_slippage())
    iteration_var.trace_add("write", update_slippage)
    fee_tier_var.trace_add("write", update_slippage)

    update_wallet_label(wallet_label)
    poll_log_queue(log_text)
    bind_enter_key_to_buttons(root)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
