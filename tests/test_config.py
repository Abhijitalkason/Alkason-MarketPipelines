"""Config: required keys present; no obviously dead top-level sections."""

from __future__ import annotations

from src.intraday import load_config

REQUIRED = {
    "universe": ["membership_file", "min_median_1min_volume", "max_median_spread_bps"],
    "data": ["provider", "bars_path", "timezone", "live_poll_seconds",
             "max_flow_missing_share", "max_skip_share"],
    "screen": ["time", "top_n", "gap_band", "lookback_days", "weights"],
    "geometry": ["target_atr", "stop_atr", "time_barrier_hours", "squareoff", "entry_window"],
    "gate": ["fire_threshold_init", "aci_gamma", "target_win_rate", "model_agreement_floor",
             "calibration_window_days", "flow_model_active", "flow_activation_share"],
    "training": ["train_min_months", "test_fold_months", "min_folds", "embargo_days",
                 "tune", "optuna_trials", "inner_calib_days"],
    "costs": ["brokerage_per_order_inr", "stt_sell_pct", "slippage_half_spread_pct",
              "impact_coeff", "slippage_stress_multiplier"],
    "risk": ["risk_per_trade_pct", "max_positions", "max_per_sector", "daily_loss_limit_pct"],
    "drift": ["reference_artifact", "current_window_sessions", "share_drifted_columns_alert",
              "consecutive_breaches_to_halt"],
    "gates": ["min_win_rate", "min_expectancy_pct", "min_signals_per_day",
              "leakage_alarm_win_rate", "calibration_max_gap", "paper_days"],
    "mlflow": ["tracking_uri", "experiment_name", "registry_model_name"],
    "api": ["host", "port", "api_key_env"],
    "paths": ["reports", "signals_log", "journal", "alerts", "gates", "run_registry", "state"],
}


def test_required_keys_present(repo):
    cfg = load_config()
    for section, keys in REQUIRED.items():
        assert section in cfg, f"missing config section {section}"
        for k in keys:
            assert k in cfg[section], f"missing config key {section}.{k}"
