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
from itertools import cycle
import numpy as np
from threading import Event
from coinbase.rest import RESTClient
from decimal import Decimal, ROUND_DOWN, getcontext, ROUND_UP

def load_vars_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get('name'), data.get('privateKey')
    except Exception:
        sys.exit("Error: Missing or invalid vars.json file.")

log_queue = queue.Queue()
use_market_order = False
amount_limit_entry = None
iteration_combo = None
start_btn = None
CUTOFF_DAYS = 30

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
        "Iteration", "💸 Net USD loss", "✅ Net USD gain",
        "Total runtime"
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

def get_price_precision(product_id):
    try:
        product = client.get_product(product_id=product_id)
        increment = product.quote_increment
        if "." in increment:
            return len(increment.split(".")[1].rstrip("0"))
        return 0
    except Exception as e:
        log(f"Failed to fetch price precision for {product_id}: {e}", "error")
        return 2  # Safe fallback

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

def get_filled_base_size_safe(order_id):
    try:
        order = client.get_order(order_id=order_id).order
        return float(order.filled_size) if order.filled_size else None
    except Exception as e:
        log(f"Error getting filled size: {e}", "error")
        return None

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

def get_spot_volume():
    from datetime import datetime, timedelta, timezone

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)
        spot_volume = 0.0
        cursor = None
        done = False  # <-- Flag to indicate cutoff reached

        while not done:
            response = client.list_orders(limit=100, cursor=cursor).to_dict()
            orders = response.get("orders", [])

            for order in orders:
                created_at = datetime.fromisoformat(order["created_time"].replace("Z", "+00:00"))
                if created_at < cutoff:
                    done = True
                    break  # exit the inner for-loop, outer while checks `done`

                filled_value = order.get("filled_value")
                if not filled_value:
                    size = float(order.get("filled_size") or 0)
                    avg_price = float(order.get("average_filled_price") or 0)
                    filled_value = size * avg_price
                else:
                    filled_value = float(filled_value)

                order_type = order.get("order_type", "").lower()
                product_type = order.get("product_type", "").lower()

                if order_type in ["futures", "futures_contract"] or product_type in ["futures", "futures_contract"]:
                    continue

                product_id = order.get("product_id", "")
                parts = product_id.split("-")
                if len(parts) < 2:
                    continue
                base, quote = parts[0], parts[-1]

                if quote != "USD" or base in ["USDC", "USDT", "DAI"]:
                    continue

                spot_volume += filled_value

            cursor = response.get("cursor")
            if not orders or not cursor:
                break

        return round(spot_volume, 2)

    except Exception as e:
        log(f"[WARN] Failed to get {CUTOFF_DAYS}d spot volume: {e}")
        return 0.0

def update_spot_volume_label(label):
    from datetime import datetime, timedelta, timezone

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=45)
        spot_volume = 0.0
        cursor = None

        while True:
            response = client.list_orders(limit=100, cursor=cursor).to_dict()
            orders = response.get("orders", [])

            for order in orders:
                created_at = datetime.fromisoformat(order["created_time"].replace("Z", "+00:00"))
                if created_at < cutoff:
                    continue

                if order["status"] != "FILLED":
                    continue

                filled_value = order.get("filled_value")
                if not filled_value:
                    size = float(order.get("filled_size") or 0)
                    avg_price = float(order.get("average_filled_price") or 0)
                    filled_value = size * avg_price
                else:
                    filled_value = float(filled_value)

                order_type = order.get("order_type", "").lower()
                product_type = order.get("product_type", "").lower()

                if order_type in ["futures", "futures_contract"] or product_type in ["futures", "futures_contract"]:
                    futures_volume += filled_value
                    continue

                product_id = order.get("product_id", "")
                parts = product_id.split("-")
                if len(parts) < 2:
                    continue
                base, quote = parts[0], parts[-1]

                if quote != "USD" or base in ["USDC", "USDT", "DAI"]:
                    continue

                spot_volume += filled_value

            cursor = response.get("cursor")
            if not orders or not cursor:
                break

        label.config(text=f"{CUTOFF_DAYS}-Day Spot Volume: ${spot_volume:,.2f}")
        log(f"📊 {CUTOFF_DAYS}-Day Spot Volume: ${spot_volume:,.2f}")

    except Exception as e:
        log(f"[WARN] Failed to update 30-day spot volume: {e}")
        label.config(text=f"{CUTOFF_DAYS}-Day Spot Volume: $ERR")

def update_slippage_label(label, iterations, amount, tier_name):
    try:
        tier = coinbase_fee_tiers.get(tier_name, default_fee)
        fee_pct = tier["taker"] if use_market_order else tier["maker"]

        # Do not round intermediate calculations with fudge factor
        raw_slippage = amount * fee_pct * 2 * iterations

        label.config(text=f"Estimated Slippage: ${raw_slippage:.6f}")
    except Exception as e:
        label.config(text="Estimated Slippage: Error")
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
    for attempt in range(max_retries):
        try:
            price_precision = get_price_precision(product_id)
            getcontext().prec = 20
            book = client.get_product_book(product_id=product_id, level=1)
            data = book.to_dict()

            book_data = data.get("pricebook", {})
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])

            if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
                log(f"[Retry {attempt + 1}] Order book data invalid or empty — waiting...", "warn")
                time.sleep(0.1)
                continue

            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            spread = max(best_ask - best_bid, 0.01)  # enforce minimum spread

            tick = Decimal("1") / Decimal("10")**price_precision
            price = None

            if side == "BUY":
                ref_price = Decimal(str(best_bid))
                price = (ref_price - tick).quantize(tick, rounding=ROUND_DOWN)
                if float(price) >= best_ask:
                    log(f"[Retry {attempt + 1}] Limit price would match market — reassessing...", "warn")
                    time.sleep(0.1)
                    continue

            elif side == "SELL":
                ref_price = Decimal(str(best_ask))
                price = (ref_price + tick).quantize(tick, rounding=ROUND_UP)
                if float(price) <= best_bid:
                    log(f"[Retry {attempt + 1}] Limit price would match market — reassessing...", "warn")
                    time.sleep(0.1)
                    continue

            order_id = generate_order_id()
            if attempt == 0 or attempt % 50 == 49:
                log(f"[SPREAD] Best bid: {best_bid}, Best ask: {best_ask}, Spread: {spread:.2f}", "status")
                log(f"{side} {base_size} @ {price} (post_only={post_only}) | ID: {order_id}", "status")

            response = client.create_order(
                client_order_id=order_id,
                product_id=product_id,
                side=side,
                order_configuration={
                    "limit_limit_gtc": {
                        "limit_price": format(price, 'f'),
                        "base_size": format(Decimal(str(base_size)).normalize(), 'f'),
                        "post_only": post_only
                    }
                }
            ).to_dict()

            if "success_response" in response:
                return response

            reason = response.get("error_response", {}).get("preview_failure_reason", "")
            if "would execute" in reason.lower() or "too close" in reason.lower():
                if attempt % 50 == 49:
                    log(f"[Retry {attempt + 1}] Still too close to market. Reason: {reason}", "warn")
                time.sleep(0.2)
                continue

            log(f"[FAILURE] Order rejected: {reason}", "error")
            return response

        except Exception as e:
            log(f"[ERROR] Exception placing order: {repr(e)}", "error")
            time.sleep(0.1)

    log(f"❌ Max retries reached placing {side} on {product_id}. Gave up hugging market.", "error")
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

def run_feeliquidator(amount_limit_entry, iteration_var, wallet_label, volume_label, iter_label, status_var, status_label, start_btn, iteration_combo, real_slippage_label, spot_volume_label):
    global running, volume_added, iterations_done, active_buy_id
    start_time = time.time()
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
    tier = coinbase_fee_tiers.get(selected_fee_tier_name, default_fee)
    fee_pct = tier["maker"] if not use_market_order else tier["taker"]
    reserved_fee = max_amount * 2 * fee_pct
    adjusted_max_amount = max(max_amount - reserved_fee, 1.00)

    while running and iterations_done < target_iterations:
        if stop_event.is_set():
            log("Stop requested — terminating loop.", "warn")
            break

        log(f"Starting iteration {iterations_done + 1} of {target_iterations}", "status")

        usd_balance = get_wallet_balance('USD')
        log(f"USD Balance: ${usd_balance:.2f}", "status")
        if usd_balance < adjusted_max_amount:
            log("⚠️ Not enough USD for next cycle.", "warning")
            break

        market_price = get_current_price_fallback(client, product_id)
        if market_price is None:
            log("Skipping cycle: No market price.", "warning")
            break

        precision = get_precision_fallback(product_id)
        base_size = round(adjusted_max_amount / market_price, precision)

        retry_limit = 3
        for attempt in range(retry_limit):
            if stop_event.is_set():
                break

            if use_market_order:
                log(f"BUY {base_size} {product_id} @ market", "status")
                quote_precision = get_price_precision(product_id)
                quote_unit = 10 ** (-quote_precision)
                floored_quote = (adjusted_max_amount // quote_unit) * quote_unit
                floored_quote = max(floored_quote, 1.00)
                quote_str = f"{floored_quote:.{quote_precision}f}"

                min_quote_threshold = 1.00
                if float(quote_str) < min_quote_threshold:
                    log(f"❌ Skipping order: quote_size ${quote_str} below Coinbase minimum of ${min_quote_threshold:.2f}", "error")
                    break

                buy_order = client.create_order(
                    client_order_id=generate_order_id(),
                    product_id=product_id,
                    side="BUY",
                    order_configuration={
                        "market_market_ioc": {
                            "quote_size": quote_str
                        }
                    }
                ).to_dict()

            else:
                log(f"BUY {base_size} {product_id} (post-only)", "status")
                buy_order = place_limit_order(
                    product_id=product_id,
                    side="BUY",
                    base_size=base_size,
                    post_only=True
                )

            if "success_response" in buy_order:
                active_buy_id = buy_order["success_response"]["order_id"]
                break
            else:
                error = buy_order.get("error_response", {})
                reason = error.get("preview_failure_reason") or error.get("message") or "Unknown"
                log(f"Buy order attempt {attempt+1}/{retry_limit} failed: {reason}", "error")
                if attempt + 1 < retry_limit:
                    log("Retrying BUY order...", "warn")
                    time.sleep(0.5)

        if not active_buy_id:
            log("❌ All BUY attempts failed. Skipping iteration.", "error")
            continue

        log(f"Waiting for BUY order to fill (ID: {active_buy_id})", "status")
        if not wait_for_order_fill(active_buy_id, timeout=90, should_stop_event=stop_event):
            log("BUY not filled. Cancelling...", "warn")
            if active_buy_id:
                try:
                    log(f"Cancelling active BUY order {active_buy_id}...", "warn")
                    cancel_order(active_buy_id)
                except Exception as e:
                    log(f"[ERROR] Exception during cancel attempt: {e}", "error")
            else:
                log("[SKIP] No active BUY order to cancel.", "warn")
            continue

        actual_base_size = get_filled_base_size_safe(active_buy_id)
        if actual_base_size is None or actual_base_size == 0:
            log("❌ Could not determine filled size. Skipping iteration.", "error")
            continue

        retry_limit = 50
        sell_id = None
        for attempt in range(retry_limit):
            if stop_event.is_set():
                break

            if use_market_order:
                log(f"SELL {actual_base_size} {product_id} @ market", "status")
                sell_order = client.create_order(
                    client_order_id=generate_order_id(),
                    product_id=product_id,
                    side="SELL",
                    order_configuration={
                        "market_market_ioc": {
                            "base_size": str(actual_base_size)
                        }
                    }
                ).to_dict()
            else:
                log(f"SELL {actual_base_size} {product_id} (post-only)", "status")
                # 🆕 Re-fetch BTC balance to avoid preview rejection
            try:
                btc_account = next((a for a in client.get_accounts().accounts if a.currency == "BTC"), None)
                if not btc_account:
                    raise ValueError("BTC account not found")
                available_btc = float(btc_account.available_balance['value'])
                if available_btc < actual_base_size:
                    log(f"❌ Failed to repost SELL limit order: Insufficient balance ({available_btc:.8f} BTC available, needed {actual_base_size})", "error")
                    continue
            except Exception as e:
                log(f"❌ Failed to fetch BTC balance before SELL retry: {e}", "error")
                continue

            sell_order = place_limit_order(
                product_id=product_id,
                side="SELL",
                base_size=available_btc,
                post_only=True
            )

            if "success_response" in sell_order:
                sell_id = sell_order["success_response"]["order_id"]
                break
            else:
                reason = sell_order.get("error_response", {}).get("message", "Unknown")
                log(f"SELL order attempt {attempt+1}/{retry_limit} failed: {reason}", "error")
                time.sleep(0.5)

        if not sell_id:
            log("❌ All SELL attempts failed. Skipping iteration.", "error")
            continue

        log(f"Waiting for SELL order to fill (ID: {sell_id})", "status")
        sell_fill_success = False
        max_sell_timeouts = 50

        for sell_timeout_attempt in range(max_sell_timeouts):
            if wait_for_order_fill(sell_id, timeout=90, should_stop_event=stop_event):
                sell_fill_success = True
                break

            log(f"SELL not filled (attempt {sell_timeout_attempt + 1}/{max_sell_timeouts}). Cancelling and retrying...", "warn")
            cancel_order(sell_id)

            sell_order = place_limit_order(
                product_id=product_id,
                side="SELL",
                base_size=actual_base_size,
                post_only=True
            )

            if "success_response" in sell_order:
                sell_id = sell_order["success_response"]["order_id"]
                log(f"[REPOST] New SELL order ID: {sell_id}", "status")
            else:
                reason = sell_order.get("error_response", {}).get("message", "Unknown")
                log(f"❌ Failed to repost SELL limit order: {reason}", "error")
                break

        if not sell_fill_success:
            log("SELL limit retries exhausted. Executing fallback market sell...", "warn")
            cancel_order(sell_id)
            fallback_sell = client.create_order(
                client_order_id=generate_order_id(),
                product_id=product_id,
                side="SELL",
                order_configuration={
                    "market_market_ioc": {
                        "base_size": str(actual_base_size)
                    }
                }
            ).to_dict()
            if "success_response" in fallback_sell:
                log("✅ Fallback market SELL succeeded.", "success")
            else:
                log("❌ Fallback market SELL failed.", "error")

        buy_value = None
        sell_value = None

        try:
            buy_order_data = client.get_order(order_id=active_buy_id).order
            buy_value = float(buy_order_data.filled_value or 0)
        except Exception as e:
            log(f"[WARN] Could not fetch BUY filled value: {e}", "warn")

        try:
            sell_order_data = client.get_order(order_id=sell_id).order
            sell_value = float(sell_order_data.filled_value or 0)
        except Exception as e:
            log(f"[WARN] Could not fetch SELL filled value: {e}", "warn")

        if buy_value is not None and sell_value is not None:
            volume_added += buy_value + sell_value
        else:
            log("[WARN] Skipping volume update due to missing fill value.", "warn")

        iterations_done += 1
        total_spent += adjusted_max_amount
        update_wallet_label(wallet_label)
        update_spot_volume_label(spot_volume_label)
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

        elapsed = time.time() - start_time
        minutes, seconds = divmod(elapsed, 60)
        runtime_str = f"⏱️ Total runtime: {int(minutes)}m {seconds:.1f}s"
        log(runtime_str, "success")

    update_wallet_label(wallet_label)

stop_event = Event()

def stop_feeliquidator():
    global running, active_buy_id
    if not running:
        return
    log("Stop requested.", tag="warn")
    stop_event.set()
    running = False

    if active_buy_id:
        log(f"Cancelling active BUY order {active_buy_id}...", "warn")
        cancel_order(active_buy_id)
        active_buy_id = None

    # Cancel all open SELL orders and fallback to market sell only once
    try:
        accounts = client.get_accounts().accounts
        btc_account = next((a for a in accounts if a.currency == "BTC"), None)
        if btc_account:
            btc_balance = float(btc_account.available_balance['value'])
            if btc_balance > 0:
                log(f"Fallback: Market selling {btc_balance:.8f} BTC due to stop...", "warn")
                fallback_sell = client.create_order(
                    client_order_id=generate_order_id(),
                    product_id="BTC-USD",
                    side="SELL",
                    order_configuration={
                        "market_market_ioc": {
                            "base_size": f"{btc_balance:.{get_precision_fallback(product_id)}f}"
                        }
                    }
                ).to_dict()
                if "success_response" in fallback_sell:
                    log("✅ Emergency fallback market SELL completed.", "success")
                else:
                    log("❌ Emergency fallback market SELL failed.", "error")
    except Exception as e:
        log(f"[ERROR] Exception during emergency SELL: {e}", "error")

    # Try UI control restore safely
    try:
        for control in (start_btn, iteration_combo, amount_limit_entry):
            if control and hasattr(control, "config"):
                control.config(state="normal")
    except Exception as e:
        log(f"[WARN] Failed to re-enable controls: {e}", "warn")

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
    global amount_limit_entry, iteration_combo, start_btn, stop_btn
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

    title_frame = tk.Frame(control_frame, bg="#1e1e1e")
    title_frame.pack(pady=(15, 5))

    title_label = tk.Label(
        control_frame,
        text="Fee Liquidator",
        fg="#3dc9a3",
        bg="#1e1e1e",
        font=("Consolas", 22, "bold"),
        pady=10
    )
    title_label.pack(pady=(15, 5))

    # Optional: subtle flicker (you can remove this too if you want it static)
    def flicker():
        import random
        if random.random() < 0.05:  # 5% chance
            title_label.config(fg="#1e1e1e")
            title_label.after(80, lambda: title_label.config(fg="#3dc9a3"))
        title_label.after(4000, flicker)

    flicker()

    wallet_label = tk.Label(control_frame, text="USD Wallet: $0.00", fg="cyan", bg="#1e1e1e", font=("Helvetica", 14))
    wallet_label.pack(pady=(0, 10))

    spot_volume_label = tk.Label(
        control_frame,
        text=f"{CUTOFF_DAYS}-Day Spot Volume: $0.00",
        fg="#FFD700",  # gold
        bg="#1e1e1e",
        font=("Helvetica", 14, "bold"),
        relief="ridge",
        bd=2,
        padx=6,
        pady=2
    )
    spot_volume_label.pack(pady=(0, 10))

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

    status_frame = tk.Frame(control_frame, bg="#1e1e1e")
    status_frame.pack(pady=5)

    status_label = tk.Label(
        status_frame,
        textvariable=status_var,
        fg="white",
        bg="#555",
        font=("Helvetica", 11, "bold"),
        padx=14,
        pady=8,
        relief="ridge",
        bd=3,
        anchor="center",
        width=32
    )
    update_status("🟡 Status: Idle", status_var, status_label)
    status_label.pack()

    slippage_label = tk.Label(control_frame, text="Expected Slippage: $0.00", fg="magenta", bg="#1e1e1e", font=("Helvetica", 12))
    slippage_label.pack(pady=(0, 10))
    real_slippage_label = tk.Label(control_frame, text="Real Slippage: $0.00", fg="cyan", bg="#1e1e1e", font=("Helvetica", 12))
    real_slippage_label.pack(pady=(0, 10))

    tk.Label(control_frame, text="Estimate Fee Tier:", fg="white", bg="#1e1e1e", font=("Helvetica", 12)).pack(pady=(0, 2))
    fee_tier_var = tk.StringVar(value=selected_fee_tier_name)
    fee_tier_combo = ttk.Combobox(control_frame, textvariable=fee_tier_var, width=20, font=("Helvetica", 12))
    fee_tier_combo['values'] = list(coinbase_fee_tiers.keys())
    fee_tier_combo.pack(padx=10, pady=5)

    def update_slippage(*args):
        try:
            amt_raw = amount_limit_entry.get().strip()
            iters_raw = iteration_var.get().strip()
            if not amt_raw or not iters_raw:
                return
            amt = float(amt_raw)
            iters = int(iters_raw)
            tier_name = fee_tier_var.get()
            update_slippage_label(slippage_label, iters, amt, tier_name)
        except Exception:
            pass

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

    def update_toggle_appearance():
        if use_market_order:
            toggle_btn.config(
                text="⚡",
                bg="#1e90ff",
                fg="white",
                activebackground="#4682b4",
                activeforeground="white"
            )
        else:
            toggle_btn.config(
                text="📉",
                bg="#2e8b57",
                fg="white",
                activebackground="#3cb371",
                activeforeground="white"
            )

    def toggle_order_type():
        global use_market_order
        use_market_order = not use_market_order
        order_type_label.config(text=f"Order Type: {'Market' if use_market_order else 'Limit'}")
        update_toggle_appearance()
        update_slippage()

    toggle_btn = tk.Button(
        order_type_frame,
        text="",
        command=toggle_order_type,
        width=3,
        relief="ridge",
        bd=1,
        font=("Helvetica", 12)
    )
    toggle_btn.pack(side="left", padx=(8, 0))
    update_toggle_appearance()

    def confirm_run(amount, iterations, expected_slippage, proceed_callback):
        popup = tk.Toplevel()
        popup.title("Confirm Trade")
        popup.configure(bg="#1e1e1e")
        popup.geometry("420x200")
        popup.resizable(False, False)

        msg_text = tk.Text(
            popup,
            height=6,
            width=55,
            bg="#1e1e1e",
            fg="white",
            font=("Helvetica", 12),
            relief="flat",
            wrap="none"
        )
        msg_text.insert("1.0", "This will likely cost ≈ ")
        msg_text.insert("end", f"${expected_slippage:.4f}", "bold_red")
        msg_text.insert("end", " in fees/slippage.\n\n")

        msg_text.insert("end", "It will add ≈ ")
        msg_text.insert("end", f"${amount * 2 * iterations:.2f}", "bold_green")
        msg_text.insert("end", " in spot volume.\n\n")

        msg_text.insert("end", "Continue?")
        msg_text.tag_configure("bold_red", foreground="red", font=("Helvetica", 12, "bold"))
        msg_text.tag_configure("bold_green", foreground="lime", font=("Helvetica", 12, "bold"))
        msg_text.tag_configure("center", justify="center")
        msg_text.tag_add("center", "1.0", "end")
        msg_text.configure(state="disabled")
        msg_text.pack(pady=10, padx=20)

        def on_continue(event=None):
            popup.destroy()
            proceed_callback()

        def on_cancel(event=None):
            popup.destroy()
            restore_controls(amount_limit_entry, iteration_combo, start_btn, status_var, status_label, volume_label, iter_label)
            update_status("🟡 Status: Idle", status_var, status_label)

        btn_frame = tk.Frame(popup, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        continue_btn = tk.Button(btn_frame, text="Continue", command=on_continue, bg="green", fg="white", font=("Helvetica", 12), width=10)
        continue_btn.grid(row=0, column=0, padx=10)

        cancel_btn = tk.Button(btn_frame, text="Cancel", command=on_cancel, bg="red", fg="white", font=("Helvetica", 12), width=10)
        cancel_btn.grid(row=0, column=1, padx=10)

        # Key bindings for Enter/Escape
        popup.bind("<Return>", on_continue)
        popup.bind("<Escape>", on_cancel)

        # Center the popup over the main window
        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        root_width = root.winfo_width()
        root_height = root.winfo_height()
        popup.update_idletasks()
        popup_width = popup.winfo_width()
        popup_height = popup.winfo_height()
        x = root_x + (root_width // 2) - (popup_width // 2)
        y = root_y + (root_height // 2) - (popup_height // 2)
        popup.geometry(f"+{x}+{y}")

        popup.transient()
        popup.grab_set()
        popup.focus_set()
        popup.wait_window()

    def start_with_confirmation():
        try:
            amt = float(amount_limit_entry.get())
            iters = int(iteration_combo.get())
            tier = coinbase_fee_tiers.get(fee_tier_var.get(), default_fee)
            fee_pct = tier["taker"] if use_market_order else tier["maker"]
            est_slip = amt * fee_pct * 4 * iters / 2.25

            def proceed():
                threading.Thread(
                    target=run_feeliquidator,
                    args=(amount_limit_entry, iteration_combo, wallet_label, volume_label, iter_label, status_var, status_label, start_btn, iteration_combo, real_slippage_label, spot_volume_label),
                    daemon=True
                ).start()

            confirm_run(amt, iters, est_slip, proceed)

        except Exception as e:
            log(f"Invalid input: {e}", "error")

    button_row = tk.Frame(control_frame, bg="#1e1e1e")
    button_row.pack(pady=5)

    start_btn = tk.Button(button_row, text="Let's Go", command=start_with_confirmation, bg="green", fg="white", font=("Helvetica", 12), width=10)
    start_btn.pack(side=tk.LEFT, padx=(0, 10))

    stop_btn = tk.Button(button_row, text="Stop", command=stop_feeliquidator, bg="red", fg="white", font=("Helvetica", 12), width=10)
    stop_btn.pack(side=tk.LEFT)

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

    amount_limit_entry.bind("<KeyRelease>", lambda e: update_slippage())
    iteration_var.trace_add("write", update_slippage)
    fee_tier_var.trace_add("write", update_slippage)

    update_wallet_label(wallet_label)
    poll_log_queue(log_text)
    bind_enter_key_to_buttons(root)

    # Fetch and update 30-day spot volume on startup
    def update_spot_volume_on_start():
        try:
            volume = get_spot_volume()
            if volume is not None:
                spot_volume_label.config(text=f"{CUTOFF_DAYS}-Day Spot Volume: ${volume:,.2f}")
                #log(f"📊 {CUTOFF_DAYS}-Day Spot Volume (all pairs): ${volume:,.2f}")
        except Exception as e:
            log(f"[WARN] Could not fetch spot volume: {e}")

    threading.Thread(target=update_spot_volume_on_start, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    start_gui()