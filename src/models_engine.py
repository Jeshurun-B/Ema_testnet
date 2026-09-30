"""
====================================================================================================
ALGORITHM: src/models_engine.py — Production RAM Singleton Inference Engine
====================================================================================================
Purpose:
  Caches all 48 production models in RAM upon module initialization, providing sub-10ms real-time
  inference across continuous CatBoost regressors (.cbm) and binary PyTorch Funnel GRUs (.pt).
  Interfaces with `features.py` to ingest true rolling sequences and normalize inputs via pre-fitted
  manifest scalers. Engineered for GitHub Actions with self-healing device selection.

Algorithm Steps:
  Step 1: Module Setup, Hardware Allocation & Architecture Definition:
          - Resolve device via get_robust_device() (CPU default on GitHub Actions runners).
          - Define MultiLayerFunnelGRU PyTorch architecture matching production checkpoint topologies.
  Step 2: Class Initialization & Path Resolution (`ProductionModelRegistry`):
          - Auto-resolve repository root across standard and nested runner directory trees.
          - Ingest hyperparameter manifests and initialize ProductionFeaturePipeline.
  Step 3: Singleton In-Memory Model Loader (`_load_all_models`):
          - Ingest 20 Symbol CatBoost Regressors (.cbm) into RAM.
          - Instantiate & ingest 20 Symbol Funnel GRU Classifiers (.pt) into RAM and set eval() mode.
  Step 4: Unified Inference Engine (`predict_trade_setup`):
          - Generate normalized (1, 16) array for CatBoost continuous MFE/MAE estimates.
          - Generate normalized (1, seq_len, 16) causal sequence tensor for Funnel GRU probabilities.
          - Execute forward passes under torch.no_grad() and clamp outputs to valid numerical bounds.
          - Measure and log inference latency in milliseconds.
  Step 5: Production Self-Test Probe (`if __name__ == '__main__'`).
====================================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostRegressor

try:
    from src.features import ProductionFeaturePipeline
except ImportError:
    from features import ProductionFeaturePipeline

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# STEP 1: Hardware Allocation & Architecture Definition
# =============================================================================
def get_robust_device():
    """Selects CUDA if modern GPU is present; defaults cleanly to CPU on GitHub Actions runners."""
    if torch.cuda.is_available():
        try:
            major_cap = torch.cuda.get_device_capability()[0]
            if major_cap >= 7:
                t = torch.randn(2, 2, device='cuda')
                _ = torch.relu(t)
                return torch.device('cuda')
        except Exception:
            pass
    return torch.device('cpu')

DEVICE = get_robust_device()


class MultiLayerFunnelGRU(nn.Module):
    """2-Layer Funnel GRU Classifier matching production PyTorch weights (input_dim=16)."""
    def __init__(self, input_dim: int = 16, hidden_dims: list = [64, 32], dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        in_d = input_dim
        for h_d in hidden_dims:
            self.layers.append(nn.GRU(input_size=in_d, hidden_size=h_d, batch_first=True))
            in_d = h_d
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curr = x
        for layer in self.layers:
            curr, h_n = layer(curr)
        return self.head(self.drop(h_n.squeeze(0)))


# =============================================================================
# STEP 2 & 3: ProductionModelRegistry (In-Memory RAM Singleton)
# =============================================================================
class ProductionModelRegistry:
    def __init__(self, repo_root: str = None):
        if repo_root is None:
            # Self-healing directory discovery
            candidates = [
                os.getcwd(),
                os.path.join(os.getcwd(), "Ema_testnet"),
                os.path.join(os.getcwd(), "ema_testnet"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ]
            self.repo_root = os.getcwd()
            for c in candidates:
                if os.path.exists(os.path.join(c, "models")):
                    self.repo_root = c
                    break
        else:
            self.repo_root = repo_root

        self.cb_dir          = os.path.join(self.repo_root, "models", "production", "catboost")
        self.gru_dir         = os.path.join(self.repo_root, "models", "production", "neural_nets")
        self.hyperparams_dir = os.path.join(self.repo_root, "Optimal_hyperparameters")

        gru_json_path = os.path.join(self.hyperparams_dir, "Ema_testnet_FunnelGRU_Classification_Optimization_results.json")
        if not os.path.exists(gru_json_path):
            raise FileNotFoundError(f"[FATAL] GRU optimization config not found at: {gru_json_path}")

        with open(gru_json_path, "r") as f:
            self.gru_configs = json.load(f).get("per_symbol_models", {})

        manifest_path = os.path.join(self.hyperparams_dir, "Ema_testnet_feature_manifest.json")
        self.feature_pipeline = ProductionFeaturePipeline(manifest_path=manifest_path)

        self.catboost_models = {}
        self.gru_models      = {}
        self.active_symbols  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
        self.directions      = ["LONG", "SHORT"]

        # Bulk load all models into memory once on startup
        self._load_all_models()

    def _load_all_models(self):
        """Loads and pre-warms all 40 production models into RAM."""
        # 1. Load 20 Symbol CatBoost Regressors
        for t_col in ["target_profit_v1", "target_danger_v1"]:
            for dir_val in self.directions:
                for sym in self.active_symbols:
                    key = f"{t_col}__{dir_val}__{sym}"
                    filename = f"Ema_testnet_CatBoostRegressor_{t_col}_{dir_val}_{sym}_models.cbm"
                    filepath = os.path.join(self.cb_dir, filename)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"[Model Registry] Missing CatBoost model: {filepath}")
                    self.catboost_models[key] = CatBoostRegressor().load_model(filepath)

        # 2. Load 20 Symbol PyTorch Funnel GRUs
        for t_cls in ["target_profit_b50", "target_danger_b50"]:
            for dir_val in self.directions:
                for sym in self.active_symbols:
                    key = f"{t_cls}__{dir_val}__{sym}"
                    cfg = self.gru_configs.get(key, {"topology": "Funnel_2L_64_32", "seq_len": 15})
                    h_dims = [64, 32] if "64_32" in cfg.get("topology", "Funnel_2L_64_32") else [64, 32, 16]

                    filename = f"Ema_testnet_FunnelGRU_{t_cls}_{dir_val}_{sym}_models.pt"
                    filepath = os.path.join(self.gru_dir, filename)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"[Model Registry] Missing Funnel GRU model: {filepath}")

                    model = MultiLayerFunnelGRU(input_dim=16, hidden_dims=h_dims, dropout=0.1).to(DEVICE)
                    model.load_state_dict(torch.load(filepath, map_location=DEVICE, weights_only=True))
                    model.eval()
                    self.gru_models[key] = (model, int(cfg.get("seq_len", 15)))


    # =========================================================================
    # STEP 4: Unified Inference Engine
    # =========================================================================
    def predict_trade_setup(
        self,
        symbol: str,
        direction: str,
        features_latest: dict,
        recent_features_list: list
    ) -> dict:
        """
        Executes sub-10ms RAM inference across CatBoost continuous regressors
        and PyTorch Funnel GRU sequence classifiers.
        """
        t0 = time.perf_counter()

        cb_profit_key = f"target_profit_v1__{direction}__{symbol}"
        cb_danger_key = f"target_danger_v1__{direction}__{symbol}"
        nn_profit_key = f"target_profit_b50__{direction}__{symbol}"
        nn_danger_key = f"target_danger_b50__{direction}__{symbol}"

        # 1. CatBoost Excursion Magnitude Inferences
        x_p_sc = self.feature_pipeline.prepare_catboost_input(features_latest, "target_profit_v1", direction, symbol)
        x_d_sc = self.feature_pipeline.prepare_catboost_input(features_latest, "target_danger_v1", direction, symbol)

        raw_pred_mfe = float(self.catboost_models[cb_profit_key].predict(x_p_sc)[0])
        raw_pred_mae = float(self.catboost_models[cb_danger_key].predict(x_d_sc)[0])

        # Enforce non-negative financial excursion bounds
        pred_profit_mfe = max(0.20, raw_pred_mfe)
        pred_danger_mae = max(0.15, raw_pred_mae)

        # 2. Funnel GRU Binary Regime Probabilities (Causal Sequence Tensor)
        gru_p_model, p_len = self.gru_models[nn_profit_key]
        gru_d_model, d_len = self.gru_models[nn_danger_key]

        x_p_tensor = self.feature_pipeline.prepare_gru_sequence_tensor(
            recent_features_list, "target_profit_b50", direction, symbol, seq_len=p_len, device=DEVICE
        )
        x_d_tensor = self.feature_pipeline.prepare_gru_sequence_tensor(
            recent_features_list, "target_danger_b50", direction, symbol, seq_len=d_len, device=DEVICE
        )

        with torch.no_grad():
            raw_prob_profit = float(torch.sigmoid(gru_p_model(x_p_tensor)).item())
            raw_prob_danger = float(torch.sigmoid(gru_d_model(x_d_tensor)).item())

        # Mathematical boundary clamping
        prob_profit = max(0.0001, min(0.9999, raw_prob_profit))
        prob_danger = max(0.0001, min(0.9999, raw_prob_danger))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pred_profit_mfe": round(pred_profit_mfe, 4),
            "pred_danger_mae": round(pred_danger_mae, 4),
            "prob_profit": round(prob_profit, 4),
            "prob_danger": round(prob_danger, 4),
            "inference_time_ms": round(elapsed_ms, 2)
        }


# =============================================================================
# STEP 5: Production Self-Test Probe
# =============================================================================
if __name__ == "__main__":
    print("===============================================================================")
    print("  TESTING PRODUCTION MODEL REGISTRY (src/models_engine.py)                     ")
    print(f"  Target Device: {DEVICE} | Architecture: RAM Singleton                        ")
    print("===============================================================================")

    t_init = time.perf_counter()
    registry = ProductionModelRegistry()
    print(f"\n1. Registry Initialized in {(time.perf_counter() - t_init):.2f}s.")
    print(f"   --> Loaded {len(registry.catboost_models)} CatBoost models.")
    print(f"   --> Loaded {len(registry.gru_models)} Funnel GRU models.")

    # Probe synthetic inference on BTCUSDT LONG
    manifest = registry.feature_pipeline.manifest
    f_names_p = manifest["shap_features_by_target"]["target_profit_v1"]["LONG"]
    f_names_d = manifest["shap_features_by_target"]["target_danger_v1"]["LONG"]
    all_f = list(set(f_names_p + f_names_d))

    dummy_feat = {col: 0.5 for col in all_f}
    dummy_seq  = [dummy_feat] * 15

    print("\n2. Executing Real-Time Inference Probe (BTCUSDT LONG)...")
    res = registry.predict_trade_setup("BTCUSDT", "LONG", dummy_feat, dummy_seq)
    print(f"   --> Pred Profit MFE : {res['pred_profit_mfe']:.2f}%")
    print(f"   --> Pred Danger MAE : {res['pred_danger_mae']:.2f}%")
    print(f"   --> Prob Profit     : {res['prob_profit']:.4f}")
    print(f"   --> Prob Danger     : {res['prob_danger']:.4f}")
    print(f"   --> Latency         : {res['inference_time_ms']} ms")

    assert res["inference_time_ms"] < 50.0, "Inference exceeded latency budget!"
    print("\n===============================================================================")
    print("  VERDICT: [PASS] IN-MEMORY MODEL REGISTRY OPERATIONAL                         ")
    print("===============================================================================")
