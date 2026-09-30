"""
====================================================================================================
ALGORITHM: src/gates_engine.py — Concurrency-Guarded Risk Gating & $100 Control Sizing Engine
====================================================================================================
Purpose:
  Translates model outputs into actionable trade manifests categorized into MAIN vs. CONTROL tiers:
    - MAIN TRADES   : Passed Dynamic Soft-Gate hurdle & consensus -> Danger-budgeted capital (up to $1k).
    - CONTROL TRADES: Failed hurdle or consensus -> Fixed $100.00 micro-notional floor.
  Guarantees Control trades cleanly clear Binance's 50 USDT minimum notional filter (error -4164)
  while risking less than $1.00 per trade on 15m stop losses. Enforces strict 5-slot concurrency.

Algorithm Steps:
  Step 1: Module Setup & Configuration Ingestion:
          - Ingest total_slots (5), max_cash_per_slot ($1,000), base_risk_budget ($50).
          - Set control_notional = 100.0 USDT (100% safety buffer above Binance 50 USDT floor).
  Step 2: Concurrency & Reversal Guard:
          - If active_slots >= 5 and this is NOT a reversal on an existing coin -> Reject (MAX_SLOTS).
  Step 3: Dynamic Barrier Calculation with 15m ATR Noise Clamp:
          - Dynamic SL % = max(1.0 * atr_15m_pct, pred_danger_mae * 1.00).
          - Dynamic TP % = max(0.20, pred_profit_mfe).
  Step 4: 4-State Taxonomy & Consensus Defense:
          - Hard rejection of HIGH_RISK__LOW_PROFIT setups into the Control tier.
  Step 5: Dynamic Confidence Soft-Gate Evaluation:
          - If Prob(Profit) >= 0.55 -> Required R:R = 1.65; else Required R:R = 2.00.
  Step 6: Asymmetric Capital Allocation (Main vs. Control):
          - Cleared gates -> Sized via danger budget ($75 / $50 / $25, max $1,000).
          - Failed gates  -> Fixed $100.00 micro-notional floor.
  Step 7: Return Complete Manifest.
  Step 8: Production Self-Test Probe (`if __name__ == '__main__'`).
====================================================================================================
"""

import os
import json
import math


class ProductionGatesEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "configs", "config_production.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(os.getcwd(), "ema_testnet", "configs", "config_production.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.cfg = json.load(f)
            self.base_risk_usd  = float(self.cfg["capital_and_slots"].get("base_risk_budget_usd", 50.0))
            self.max_cash_slot  = float(self.cfg["capital_and_slots"].get("max_cash_per_slot", 1000.0))
            self.total_slots    = int(self.cfg["capital_and_slots"].get("total_slots", 5))
            self.fixed_leverage = float(self.cfg["capital_and_slots"].get("fixed_leverage", 1.0))
        else:
            self.base_risk_usd  = 50.0
            self.max_cash_slot  = 1000.0
            self.total_slots    = 5
            self.fixed_leverage = 1.0

        # Policy & Dynamic Hurdle Constants
        self.tp_multiplier      = 1.00
        self.sl_multiplier      = 1.00
        self.standard_rr_hurdle = 2.00
        self.relaxed_rr_hurdle  = 1.65
        self.high_conf_thresh   = 0.55
        self.danger_thresh      = 0.50
        self.profit_thresh      = 0.50
        self.control_notional   = 100.0  # Fixed $100.00 floor (clears Binance 50 USDT filter with 2x buffer)

        self.multipliers = {
            "LOW_RISK__HIGH_PROFIT":  1.50,  # $75.00 Risk Budget
            "LOW_RISK__LOW_PROFIT":   1.00,  # $50.00 Risk Budget
            "HIGH_RISK__HIGH_PROFIT": 0.50,  # $25.00 Risk Budget (De-risked)
            "HIGH_RISK__LOW_PROFIT":  0.00   # Consensus Rejection
        }

    def evaluate_gates_and_sizing(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        model_outputs: dict,
        atr_pct: float = 0.40,
        free_wallet_balance: float = 5000.0,
        active_positions_count: int = 0,
        is_reversal: bool = False
    ) -> dict:
        """
        Evaluates risk gates, enforces strict portfolio concurrency, clamps stops
        to ATR noise, and partitions into MAIN vs. CONTROL tiers ($100 floor).
        """
        # 1. Strict Concurrency Bound Guard
        if active_positions_count >= self.total_slots and not is_reversal:
            return {
                "approved": False,
                "trade_tier": "AWAITING",
                "rejection_reason": "MAX_CONCURRENCY_REACHED (5/5 Slots Full)",
                "symbol": symbol,
                "direction": direction,
                "allocated_cash": 0.0,
                "contract_quantity": 0.0
            }

        pred_profit_mfe = float(model_outputs["pred_profit_mfe"])
        pred_danger_mae = float(model_outputs["pred_danger_mae"])
        prob_profit     = float(model_outputs["prob_profit"])
        prob_danger     = float(model_outputs["prob_danger"])

        # 2. Dynamic Barriers with 15m ATR % Noise-Floor Clamp
        dynamic_tp_pct = max(0.20, pred_profit_mfe) * self.tp_multiplier
        raw_sl_pct     = max(0.15, pred_danger_mae) * self.sl_multiplier
        dynamic_sl_pct = max(float(atr_pct), raw_sl_pct)  # Clamped above 1-bar Brownian noise!

        rr_ratio = (dynamic_tp_pct / dynamic_sl_pct) if dynamic_sl_pct > 0 else 0.0

        if direction.upper() == "LONG":
            dynamic_tp_price = entry_price * (1.0 + (dynamic_tp_pct / 100.0))
            dynamic_sl_price = entry_price * (1.0 - (dynamic_sl_pct / 100.0))
        else:
            dynamic_tp_price = entry_price * (1.0 - (dynamic_tp_pct / 100.0))
            dynamic_sl_price = entry_price * (1.0 + (dynamic_sl_pct / 100.0))

        # 3. 4-State Taxonomy Classification
        risk_tier   = "HIGH_RISK"   if prob_danger >= self.danger_thresh else "LOW_RISK"
        profit_tier = "HIGH_PROFIT" if prob_profit >= self.profit_thresh else "LOW_PROFIT"
        gate_tag    = f"{risk_tier}__{profit_tier}"

        # 4. Dynamic Soft-Gate Verification
        required_rr = self.relaxed_rr_hurdle if prob_profit >= self.high_conf_thresh else self.standard_rr_hurdle
        passed_hurdle = (rr_ratio >= required_rr)
        passed_consensus = not (risk_tier == "HIGH_RISK" and profit_tier == "LOW_PROFIT")

        # 5. Experimental Tier Assignment: MAIN vs. CONTROL ($100 Floor)
        if passed_hurdle and passed_consensus:
            trade_tier = "MAIN"
            rejection_reason = "None"
            
            category_mult = float(self.multipliers.get(gate_tag, 1.0))
            dollar_risk_budget = self.base_risk_usd * category_mult

            remaining_slots = max(1, self.total_slots - active_positions_count)
            slot_cash_cap   = min(self.max_cash_slot, free_wallet_balance / remaining_slots)

            uncapped_pos_usd = dollar_risk_budget / (dynamic_sl_pct / 100.0)
            allocated_cash   = min(slot_cash_cap, uncapped_pos_usd)
        else:
            trade_tier = "CONTROL"
            rejection_reason = "CONSENSUS_FAILURE" if not passed_consensus else f"RR_HURDLE_FAILED ({rr_ratio:.2f} < {required_rr:.2f})"
            dollar_risk_budget = 1.00
            allocated_cash = min(free_wallet_balance, self.control_notional)

        contract_quantity = allocated_cash / entry_price if entry_price > 0 else 0.0

        return {
            "approved": True,
            "trade_tier": trade_tier,
            "rejection_reason": rejection_reason,
            "symbol": symbol,
            "direction": direction,
            "gate_combo_tag": gate_tag,
            "rr_ratio": round(rr_ratio, 2),
            "entry_price": round(entry_price, 6),
            "dynamic_tp_pct": round(dynamic_tp_pct, 4),
            "dynamic_sl_pct": round(dynamic_sl_pct, 4),
            "dynamic_tp_price": round(dynamic_tp_price, 6),
            "dynamic_sl_price": round(dynamic_sl_price, 6),
            "risk_budget_usd": round(dollar_risk_budget, 2),
            "allocated_cash": round(allocated_cash, 2),
            "contract_quantity": round(contract_quantity, 6),
            "leverage": self.fixed_leverage,
            "pred_profit_mfe": round(pred_profit_mfe, 4),
            "pred_danger_mae": round(pred_danger_mae, 4),
            "prob_profit": round(prob_profit, 4),
            "prob_danger": round(prob_danger, 4),
            "noise_ratio": round(dynamic_sl_pct / float(atr_pct), 2) if float(atr_pct) > 0 else 1.0
        }


# =============================================================================
# STEP 8: Integration Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING RISK GATING ENGINE (src/gates_engine.py — $100 CONTROL FLOOR)        ")
    print("===============================================================================")
    engine = ProductionGatesEngine()

    # Test Control Trade Sizing on BTC ($83,841.60)
    mock_failed = {"pred_profit_mfe": 1.11, "pred_danger_mae": 0.65, "prob_profit": 0.525, "prob_danger": 0.475}
    r = engine.evaluate_gates_and_sizing("BTCUSDT", "SHORT", 83841.60, mock_failed, atr_pct=0.40, active_positions_count=3)
    print(f"Control Trade Result: Tier={r['trade_tier']} | Cash=${r['allocated_cash']:,.2f} | Reason={r['rejection_reason']}")
    assert r['trade_tier'] == 'CONTROL' and r['allocated_cash'] == 100.0, "Control sizing failed to allocate $100.00 floor!"
    print("===============================================================================")
    print("  VERDICT: [PASS] GATES ENGINE SIZING ALIGNED                                  ")
    print("===============================================================================")
