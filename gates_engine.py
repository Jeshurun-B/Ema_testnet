"""
====================================================================================================
ALGORITHM: src/gates_engine.py — Production Decision Policy, Sizing & Parity Hurdle
====================================================================================================
Purpose:
  Translate raw model outputs (MFE %, MAE %, Prob(Profit), Prob(Danger)) into a definitive,
  actionable trading decision ('APPROVED' vs 'REJECTED') with exact dynamic barriers,
  unleveraged 1.0x position sizing, and risk-budgeted allocations.

Key Policy Enforcements:
  1. Regression Prediction Multipliers = 1.00:
     - Dynamic TP % = pred_profit_mfe * 1.00
     - Dynamic SL % = pred_danger_mae * 1.00
  2. The 2.0:1 Asymmetrical Parity Hurdle:
     - R:R = (Dynamic TP % / Dynamic SL %) >= 2.00
     - Rejects any trade where expected reward does not offer at least 2x the downside danger.
  3. 4-State Consensus Classification:
     - Danger Gate: Prob(Danger) >= 0.50 flags 'HIGH_RISK'.
     - Profit Gate: Prob(Profit) >= 0.50 flags 'HIGH_PROFIT'.
     - Consensus Rejection: Immediately rejects 'HIGH_RISK__LOW_PROFIT' setups.
  4. Unleveraged Volatility-Targeted Position Sizing:
     - Base Risk Budget = $50.00 (0.50% of $10,000 portfolio).
     - Category Multipliers:
         * LOW_RISK__HIGH_PROFIT  : 1.5x Multiplier ($75.00 Risk Budget)
         * LOW_RISK__LOW_PROFIT   : 1.0x Multiplier ($50.00 Risk Budget)
         * HIGH_RISK__HIGH_PROFIT : 0.5x Multiplier ($25.00 Risk Budget, De-risked)
     - Dynamic Slot Cash Cap: Available Free Cash / Remaining Unoccupied Slots (Capped at $2,000).
     - Position Size ($) = min( Slot Cash Cap, Dollar Risk Budget / (Dynamic SL % / 100) ).
     - Contract Quantity = Position Size ($) / Entry Price.
     - Leverage: Fixed strictly at 1.0x (unleveraged cash allocation).

Algorithm Steps:
  1. Module Setup & Configuration Ingestion:
     - Load `configs/config_production.json` to extract hurdle thresholds, multipliers, and sizing caps.
  2. Class Definition — ProductionGatesEngine:
     - Encapsulate policy evaluation inside clean, verifiable methods.
  3. Barrier & Hurdle Verification:
     - Calculate Dynamic TP % and Dynamic SL %.
     - Evaluate R:R ratio against the min_rr_hurdle (2.00).
  4. 4-State Taxonomy Assignment:
     - Map risk_tier and profit_tier into gate_combo_tag.
     - Enforce consensus rejection filter.
  5. Unleveraged Risk-Budgeted Sizing Calculation:
     - Apply category multiplier to calculate Dollar Risk Budget.
     - Calculate maximum cash permitted for this slot based on available free balance.
     - Compute exact position size ($) and contract quantity.
  6. Return Decision Manifest:
     - Return clean dictionary containing approval status, barriers, quantity, and sizing metadata.
  7. Built-in Integration Self-Test (if __name__ == '__main__'):
     - Test mock trade setups across the 4 states and assert mathematical invariants.
====================================================================================================
"""

# =============================================================================
# STEP 1: Module Setup & Configuration Ingestion
# =============================================================================
import os
import json
import math

class ProductionGatesEngine:
    """
    Evaluates institutional risk gates, enforces the R:R >= 2.0 hurdle,
    and calculates unleveraged danger-budgeted position sizes.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "configs", "config_production.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(os.getcwd(), "ema_testnet", "configs", "config_production.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[Gates Engine] Missing configuration file: {config_path}")

        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        # Ingest Policy Constants
        self.min_rr_hurdle   = float(self.cfg["risk_and_hurdle_policy"]["min_rr_hurdle"])          # 2.00
        self.tp_multiplier   = float(self.cfg["risk_and_hurdle_policy"]["dynamic_tp_multiplier"])  # 1.00
        self.sl_multiplier   = float(self.cfg["risk_and_hurdle_policy"]["dynamic_sl_multiplier"])  # 1.00
        self.danger_thresh   = float(self.cfg["risk_and_hurdle_policy"]["danger_cls_threshold"])   # 0.50
        self.profit_thresh   = float(self.cfg["risk_and_hurdle_policy"]["profit_cls_threshold"])   # 0.50
        self.multipliers     = self.cfg["risk_and_hurdle_policy"]["category_multipliers"]

        # Sizing & Slot Constants
        self.base_risk_usd   = float(self.cfg["capital_and_slots"]["base_risk_budget_usd"])        # $50.00
        self.max_cash_slot   = float(self.cfg["capital_and_slots"]["max_cash_per_slot"])           # $2,000.00
        self.total_slots     = int(self.cfg["capital_and_slots"]["total_slots"])                   # 5 slots
        self.fixed_leverage  = float(self.cfg["capital_and_slots"]["fixed_leverage"])              # 1.0x

    # =========================================================================
    # STEP 2–5: Policy Evaluation & Position Sizing Engine
    # =========================================================================
    def evaluate_gates_and_sizing(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        model_outputs: dict,
        free_wallet_balance: float = 10000.0,
        active_positions_count: int = 0
    ) -> dict:
        """
        Translates raw model outputs into an approved trade manifest or rejection notice.
        
        Args:
            symbol (str): Asset pair (e.g. 'BTCUSDT')
            direction (str): 'LONG' or 'SHORT'
            entry_price (float): 9/15 EMA crossover candle close price
            model_outputs (dict): Output from ProductionModelRegistry.predict_trade_setup()
            free_wallet_balance (float): Current available USDT cash on Binance
            active_positions_count (int): Count of currently open positions (0 to 5)
            
        Returns:
            dict containing decision ('APPROVED' vs 'REJECTED'), pricing barriers, and sizing.
        """
        pred_profit_mfe = model_outputs["pred_profit_mfe"]
        pred_danger_mae = model_outputs["pred_danger_mae"]
        prob_profit     = model_outputs["prob_profit"]
        prob_danger     = model_outputs["prob_danger"]

        # 1. Dynamic Barriers (1.00x Multipliers)
        dynamic_tp_pct = pred_profit_mfe * self.tp_multiplier
        dynamic_sl_pct = pred_danger_mae * self.sl_multiplier
        rr_ratio       = (dynamic_tp_pct / dynamic_sl_pct) if dynamic_sl_pct > 0 else 0.0

        # Calculate Barrier Exit Prices
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

        # 3. Filter Rejections
        # Rule A: Consensus Rejection (High Risk + Low Profit)
        if risk_tier == "HIGH_RISK" and profit_tier == "LOW_PROFIT":
            return {
                "approved": False,
                "rejection_reason": "CONSENSUS_FAILURE: HIGH_RISK + LOW_PROFIT",
                "gate_combo_tag": gate_tag,
                "rr_ratio": round(rr_ratio, 2)
            }

        # Rule B: Asymmetrical Parity Hurdle (R:R >= 2.00)
        if rr_ratio < self.min_rr_hurdle:
            return {
                "approved": False,
                "rejection_reason": f"RR_HURDLE_FAILED: R:R = {rr_ratio:.2f} < {self.min_rr_hurdle:.2f}",
                "gate_combo_tag": gate_tag,
                "rr_ratio": round(rr_ratio, 2)
            }

        # 4. Unleveraged Volatility-Targeted Position Sizing
        category_mult = float(self.multipliers.get(gate_tag, 1.0))
        dollar_risk_budget = self.base_risk_usd * category_mult  # $75, $50, or $25

        # Dynamic Slot Cash Cap: Divides free cash across remaining slots (never starves coins)
        remaining_unoccupied_slots = max(1, self.total_slots - active_positions_count)
        slot_cash_cap = min(self.max_cash_slot, free_wallet_balance / remaining_unoccupied_slots)

        # Danger-Driven Position Size: Sized inversely to Stop-Loss width
        uncapped_position_usd = dollar_risk_budget / (dynamic_sl_pct / 100.0)
        allocated_cash = min(slot_cash_cap, uncapped_position_usd)

        # Quantize contract quantity
        contract_quantity = allocated_cash / entry_price if entry_price > 0 else 0.0

        return {
            "approved": True,
            "rejection_reason": "None",
            "symbol": symbol,
            "direction": direction,
            "gate_combo_tag": gate_tag,
            "rr_ratio": round(rr_ratio, 2),
            "entry_price": round(entry_price, 8),
            "dynamic_tp_pct": round(dynamic_tp_pct, 4),
            "dynamic_sl_pct": round(dynamic_sl_pct, 4),
            "dynamic_tp_price": round(dynamic_tp_price, 8),
            "dynamic_sl_price": round(dynamic_sl_price, 8),
            "risk_budget_usd": round(dollar_risk_budget, 2),
            "allocated_cash": round(allocated_cash, 2),
            "contract_quantity": round(contract_quantity, 6),
            "leverage": self.fixed_leverage  # Strictly 1.0x
        }


# =============================================================================
# STEP 6: Built-in Integration Self-Test
# =============================================================================
RUN_GATES_ENGINE_SELF_TEST = True

if __name__ == "__main__" and RUN_GATES_ENGINE_SELF_TEST:
    print("===============================================================================")
    print("  TESTING PRODUCTION POLICY & GATES ENGINE (src/gates_engine.py)               ")
    print("  Rules: Multipliers=1.0 | R:R >= 2.0 Hurdle | Danger Sizing | Unleveraged 1.0x")
    print("===============================================================================\n")

    engine = ProductionGatesEngine()

    # Test Case 1: Prime Setup (LOW_RISK__HIGH_PROFIT) -> Expect Approval, 1.5x Multiplier
    mock_prime = {"pred_profit_mfe": 2.80, "pred_danger_mae": 0.60, "prob_profit": 0.65, "prob_danger": 0.35}
    res1 = engine.evaluate_gates_and_sizing("BTCUSDT", "LONG", 64000.0, mock_prime, free_wallet_balance=8000.0, active_positions_count=1)
    print(f"Test 1 [Prime Setup]:")
    print(f"  Approved: {res1['approved']} | Tag: {res1['gate_combo_tag']} | R:R: {res1['rr_ratio']}")
    print(f"  Allocated Cash: ${res1['allocated_cash']} (Slot Cap: ${engine.max_cash_slot}) | Quantity: {res1['contract_quantity']} BTC")
    assert res1['approved'] == True and res1['risk_budget_usd'] == 75.0, "Test 1 failed!"

    # Test Case 2: Volatile Breakout (HIGH_RISK__HIGH_PROFIT) -> Expect Danger-Shrunk Size
    mock_volatile = {"pred_profit_mfe": 5.00, "pred_danger_mae": 2.20, "prob_profit": 0.60, "prob_danger": 0.60}
    res2 = engine.evaluate_gates_and_sizing("SOLUSDT", "LONG", 140.0, mock_volatile, free_wallet_balance=6000.0, active_positions_count=2)
    print(f"\nTest 2 [Volatile Altcoin Breakout]:")
    print(f"  Approved: {res2['approved']} | Tag: {res2['gate_combo_tag']} | R:R: {res2['rr_ratio']}")
    print(f"  Allocated Cash: ${res2['allocated_cash']} (Danger automatically shrunk size below $2k cap!)")
    assert res2['approved'] == True and res2['allocated_cash'] < 1200.0, "Test 2 failed danger shrinkage!"

    # Test Case 3: Hurdle Failure (R:R < 2.0) -> Expect Rejection
    mock_low_rr = {"pred_profit_mfe": 1.50, "pred_danger_mae": 1.10, "prob_profit": 0.55, "prob_danger": 0.40}
    res3 = engine.evaluate_gates_and_sizing("ETHUSDT", "SHORT", 3400.0, mock_low_rr, free_wallet_balance=10000.0, active_positions_count=0)
    print(f"\nTest 3 [Hurdle Rejection (R:R = {res3['rr_ratio']} < 2.0)]:")
    print(f"  Approved: {res3['approved']} | Reason: {res3['rejection_reason']}")
    assert res3['approved'] == False and "RR_HURDLE_FAILED" in res3['rejection_reason'], "Test 3 hurdle filter failed!"

    # Test Case 4: Consensus Failure (HIGH_RISK__LOW_PROFIT) -> Expect Rejection
    mock_consensus_fail = {"pred_profit_mfe": 1.20, "pred_danger_mae": 1.80, "prob_profit": 0.40, "prob_danger": 0.70}
    res4 = engine.evaluate_gates_and_sizing("DOGEUSDT", "LONG", 0.10, mock_consensus_fail)
    print(f"\nTest 4 [Consensus Failure]:")
    print(f"  Approved: {res4['approved']} | Reason: {res4['rejection_reason']}")
    assert res4['approved'] == False and "CONSENSUS_FAILURE" in res4['rejection_reason'], "Test 4 consensus rejection failed!"

    print("\n===============================================================================")
    print("  VERDICT: [PASS] PRODUCTION POLICY & GATES ENGINE FULLY OPERATIONAL           ")
    print("===============================================================================")
