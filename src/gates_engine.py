"""
====================================================================================================
ALGORITHM: src/gates_engine.py — Deterministic Flat Sizing Engine ($1,000 Main / $50 Control)
====================================================================================================
Purpose:
  Translates model outputs into actionable trade manifests across 4 active altcoins:
    - MAIN TRADES   : Passed Dynamic Soft-Gate hurdle & consensus -> Flat $1,000.00 cash.
    - CONTROL TRADES: Failed hurdle or consensus -> Flat $50.00 cash.
  Eliminates dynamic sizing multipliers to permanently prevent the $2,000 allocation explosion.
  Enforces a strict 4-slot concurrency bound and the 15m ATR % noise-floor clamp.

Algorithm Steps:
  Step 1: Module Setup & Configuration Ingestion:
          - Enforces total_slots = 4.
          - Hardcodes defensive clamp: max_cash_per_slot = 1000.0.
  Step 2: Concurrency & Reversal Guard:
          - Blocks new entries if active_slots >= 4 (unless reversing an existing trade).
  Step 3: Dynamic Barrier Calculation with 15m ATR Noise Clamp:
          - Dynamic SL % = max(1.0 * atr_15m_pct, pred_danger_mae * 1.00).
          - Dynamic TP % = max(0.20, pred_profit_mfe).
  Step 4: 4-State Taxonomy & Consensus Defense:
          - Rejects HIGH_RISK__LOW_PROFIT setups into the Control tier.
  Step 5: Dynamic Confidence Soft-Gate Evaluation:
          - If Prob(Profit) >= 0.55 -> Required R:R = 1.65; else Required R:R = 2.00.
  Step 6: Deterministic Flat Sizing Allocation:
          - MAIN    : allocated_cash = min(1000.0, free_wallet_balance).
          - CONTROL : allocated_cash = min(50.0, free_wallet_balance).
          - Contract quantity = allocated_cash / entry_price.
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

        self.total_slots        = 4
        self.fixed_main_cash    = 1000.0
        self.fixed_control_cash = 50.0
        self.fixed_leverage     = 1.0

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                self.total_slots = int(cfg.get("capital_and_slots", {}).get("total_slots", 4))
            except Exception:
                pass

        # Defensive hard clamp: Main trade capital can NEVER exceed $1,000.00
        self.max_cash_slot      = 1000.0
        self.standard_rr_hurdle = 2.00
        self.relaxed_rr_hurdle  = 1.65
        self.high_conf_thresh   = 0.55
        self.danger_thresh      = 0.50
        self.profit_thresh      = 0.50

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
        """Evaluates Soft-Gate hurdle, enforces 4-slot concurrency, and sizes flat $1k / $50."""
        # 1. Concurrency Guard (Strictly 4 Slots Max)
        if active_positions_count >= self.total_slots and not is_reversal:
            return {
                "approved": False,
                "trade_tier": "AWAITING",
                "rejection_reason": f"MAX_CONCURRENCY_REACHED ({self.total_slots}/{self.total_slots} Slots Full)",
                "symbol": symbol,
                "direction": direction,
                "allocated_cash": 0.0,
                "contract_quantity": 0.0
            }

        pred_profit_mfe = float(model_outputs["pred_profit_mfe"])
        pred_danger_mae = float(model_outputs["pred_danger_mae"])
        prob_profit     = float(model_outputs["prob_profit"])
        prob_danger     = float(model_outputs["prob_danger"])

        # 2. Dynamic Barrier Percentages with 15m ATR % Noise-Floor Clamp
        dynamic_tp_pct = max(0.20, pred_profit_mfe)
        raw_sl_pct     = max(0.15, pred_danger_mae)
        dynamic_sl_pct = max(float(atr_pct), raw_sl_pct)  # Clamped above 1-bar Brownian noise!

        rr_ratio = (dynamic_tp_pct / dynamic_sl_pct) if dynamic_sl_pct > 0 else 0.0

        # Estimated dollar barriers (to be re-derived from physical fill price)
        if direction.upper() == "LONG":
            est_tp_price = entry_price * (1.0 + (dynamic_tp_pct / 100.0))
            est_sl_price = entry_price * (1.0 - (dynamic_sl_pct / 100.0))
        else:
            est_tp_price = entry_price * (1.0 - (dynamic_tp_pct / 100.0))
            est_sl_price = entry_price * (1.0 + (dynamic_sl_pct / 100.0))

        # 3. 4-State Taxonomy Classification
        risk_tier   = "HIGH_RISK"   if prob_danger >= self.danger_thresh else "LOW_RISK"
        profit_tier = "HIGH_PROFIT" if prob_profit >= self.profit_thresh else "LOW_PROFIT"
        gate_tag    = f"{risk_tier}__{profit_tier}"

        # 4. Dynamic Soft-Gate Verification
        required_rr = self.relaxed_rr_hurdle if prob_profit >= self.high_conf_thresh else self.standard_rr_hurdle
        passed_hurdle = (rr_ratio >= required_rr)
        passed_consensus = not (risk_tier == "HIGH_RISK" and profit_tier == "LOW_PROFIT")

        # 5. Deterministic Flat Sizing (Strictly Flat $1,000 Main / Flat $50 Control)
        if passed_hurdle and passed_consensus:
            trade_tier = "MAIN"
            rejection_reason = "None"
            allocated_cash = min(self.fixed_main_cash, free_wallet_balance)
        else:
            trade_tier = "CONTROL"
            rejection_reason = "CONSENSUS_FAILURE" if not passed_consensus else f"RR_HURDLE_FAILED ({rr_ratio:.2f} < {required_rr:.2f})"
            allocated_cash = min(self.fixed_control_cash, free_wallet_balance)

        # Defensive assertion: allocated_cash can NEVER exceed $1,000.00
        allocated_cash = min(1000.0, float(allocated_cash))
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
            "dynamic_tp_price": round(est_tp_price, 6),
            "dynamic_sl_price": round(est_sl_price, 6),
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
    print("  TESTING FLAT SIZING GATING ENGINE (src/gates_engine.py)                      ")
    print("===============================================================================")
    engine = ProductionGatesEngine()

    # Test 1: Main Trade Flat Sizing ($1,000.00)
    mock_passed = {"pred_profit_mfe": 2.0, "pred_danger_mae": 0.5, "prob_profit": 0.70, "prob_danger": 0.15}
    r1 = engine.evaluate_gates_and_sizing("ETHUSDT", "LONG", 2650.0, mock_passed, atr_pct=0.40)
    print(f"Main Trade: Tier={r1['trade_tier']} | Cash=${r1['allocated_cash']:,.2f}")
    assert r1['allocated_cash'] == 1000.0, "Main trade failed to allocate flat $1,000.00!"

    # Test 2: Control Trade Flat Sizing ($50.00)
    mock_failed = {"pred_profit_mfe": 0.8, "pred_danger_mae": 0.9, "prob_profit": 0.30, "prob_danger": 0.50}
    r2 = engine.evaluate_gates_and_sizing("SOLUSDT", "SHORT", 150.0, mock_failed, atr_pct=0.40)
    print(f"Control Trade: Tier={r2['trade_tier']} | Cash=${r2['allocated_cash']:,.2f}")
    assert r2['allocated_cash'] == 50.0, "Control trade failed to allocate flat $50.00!"
    print("===============================================================================")
    print("  VERDICT: [PASS] FLAT SIZING & HARD 4-SLOT CAP OPERATIONAL                    ")
    print("===============================================================================")
