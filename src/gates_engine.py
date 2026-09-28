"""
====================================================================================================
ALGORITHM: src/gates_engine.py — Experimental Gating Engine (Main vs. Control Trade Tiers)
====================================================================================================
Purpose:
  Translates raw model outputs (CatBoost MFE/MAE and Funnel GRU probabilities) into trade manifests
  categorized into two live experimental tiers: MAIN TRADES (cleared all gates, allocated standard
  danger-budgeted capital up to $1,000) and CONTROL TRADES (failed gates, allocated fixed $50.00
  micro-notional). Enforces the Dynamic Soft-Gate hurdle and the 15m ATR % noise-floor clamp.

Algorithm Steps:
  Step 1: Module Setup, Configuration Ingestion & Threshold Loading.
  Step 2: Class Definition — ProductionGatesEngine.
  Step 3: Dynamic Barrier Calculation with 15m ATR Noise Clamp:
          - Dynamic SL % = max(1.0 * atr_15m_pct, pred_danger_mae * 1.00).
  Step 4: 4-State Taxonomy Classification & Consensus Defense:
          - Flags HIGH_RISK__LOW_PROFIT as consensus failure.
  Step 5: Dynamic Confidence Soft-Gate Evaluation:
          - If Prob(Profit) >= 0.55 -> Required R:R = 1.65; else Required R:R = 2.00.
  Step 6: Trade Tier & Capital Allocation (Main vs. Control):
          - If passed: trade_tier = 'MAIN', sized via danger budget ($75/$50/$25, max $1,000).
          - If failed: trade_tier = 'CONTROL', sized at fixed $50.00 micro-notional floor.
  Step 7: Return Complete Trade Manifest.
  Step 8: Built-in Integration Self-Test (`if __name__ == '__main__'`).
====================================================================================================
"""

import os
import json
import math

class ProductionGatesEngine:
    """
    Evaluates risk gates, assigns trade tiers (MAIN vs. CONTROL),
    clamps stop losses to 15m ATR noise, and sizes capital accordingly.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "configs", "config_production.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(os.getcwd(), "Ema_testnet", "configs", "config_production.json")
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
        self.control_notional   = 50.0  # Fixed $50.00 micro-notional floor for Control trades

        self.multipliers = {
            "LOW_RISK__HIGH_PROFIT":  1.50,  # $75.00 Risk Budget
            "LOW_RISK__LOW_PROFIT":   1.00,  # $50.00 Risk Budget
            "HIGH_RISK__HIGH_PROFIT": 0.50,  # $25.00 Risk Budget (De-risked)
            "HIGH_RISK__LOW_PROFIT":  0.00
        }

    def evaluate_gates_and_sizing(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        model_outputs: dict,
        atr_pct: float = 0.40,
        free_wallet_balance: float = 5000.0,
        active_positions_count: int = 0
    ) -> dict:
        pred_profit_mfe = float(model_outputs["pred_profit_mfe"])
        pred_danger_mae = float(model_outputs["pred_danger_mae"])
        prob_profit     = float(model_outputs["prob_profit"])
        prob_danger     = float(model_outputs["prob_danger"])

        # 1. Dynamic Barriers with 15m ATR % Noise-Floor Clamp
        dynamic_tp_pct = max(0.20, pred_profit_mfe) * self.tp_multiplier
        raw_sl_pct     = max(0.15, pred_danger_mae) * self.sl_multiplier
        dynamic_sl_pct = max(float(atr_pct), raw_sl_pct)  # Clamped above 1-bar noise!

        rr_ratio = (dynamic_tp_pct / dynamic_sl_pct) if dynamic_sl_pct > 0 else 0.0

        if direction.upper() == "LONG":
            dynamic_tp_price = entry_price * (1.0 + (dynamic_tp_pct / 100.0))
            dynamic_sl_price = entry_price * (1.0 - (dynamic_sl_pct / 100.0))
        else:
            dynamic_tp_price = entry_price * (1.0 - (dynamic_tp_pct / 100.0))
            dynamic_sl_price = entry_price * (1.0 + (dynamic_sl_pct / 100.0))

        # 2. 4-State Taxonomy Classification
        risk_tier   = "HIGH_RISK"   if prob_danger >= self.danger_thresh else "LOW_RISK"
        profit_tier = "HIGH_PROFIT" if prob_profit >= self.profit_thresh else "LOW_PROFIT"
        gate_tag    = f"{risk_tier}__{profit_tier}"

        # 3. Dynamic Soft-Gate Verification
        required_rr = self.relaxed_rr_hurdle if prob_profit >= self.high_conf_thresh else self.standard_rr_hurdle
        passed_hurdle = (rr_ratio >= required_rr)
        passed_consensus = not (risk_tier == "HIGH_RISK" and profit_tier == "LOW_PROFIT")

        # 4. Experimental Tier Assignment: MAIN vs. CONTROL
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
            dollar_risk_budget = 0.50
            # Control trades allocate fixed $50.00 notional floor
            allocated_cash = min(free_wallet_balance, self.control_notional)

        contract_quantity = allocated_cash / entry_price if entry_price > 0 else 0.0

        return {
            "approved": True,  # 100% of crossovers are executed on Binance in Version 2.0!
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


if __name__ == "__main__":
    engine = ProductionGatesEngine()
    mock_passed = {"pred_profit_mfe": 1.80, "pred_danger_mae": 0.40, "prob_profit": 0.60, "prob_danger": 0.20}
    r1 = engine.evaluate_gates_and_sizing("BTCUSDT", "LONG", 75000.0, mock_passed, atr_pct=0.35)
    print(f"Main Trade: Tier={r1['trade_tier']} | Cash=${r1['allocated_cash']} | R:R={r1['rr_ratio']}")
    assert r1['trade_tier'] == 'MAIN'

    mock_failed = {"pred_profit_mfe": 0.80, "pred_danger_mae": 0.90, "prob_profit": 0.30, "prob_danger": 0.40}
    r2 = engine.evaluate_gates_and_sizing("SOLUSDT", "SHORT", 100.0, mock_failed, atr_pct=0.45)
    print(f"Control Trade: Tier={r2['trade_tier']} | Cash=${r2['allocated_cash']} (Fixed $50 Floor) | Reason={r2['rejection_reason']}")
    assert r2['trade_tier'] == 'CONTROL' and r2['allocated_cash'] == 50.0
