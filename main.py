"""
main.py — CLI entry point for the AI-MLOps Stock Prediction System.

Usage examples:
  python main.py --mode ingest
  python main.py --mode train --trainer manual
  python main.py --mode train --trainer mlflow --symbols RELIANCE.NS TCS.NS
  python main.py --mode predict --symbol RELIANCE.NS
  python main.py --mode full-pipeline --trainer mlflow
  python main.py --mode serve
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_symbols(args, cfg) -> list:
    """Resolve symbol list: CLI override → config default."""
    if hasattr(args, "symbols") and args.symbols:
        return [s.upper() for s in args.symbols]
    if hasattr(args, "symbol") and args.symbol:
        return [args.symbol.upper()]
    return cfg["stocks"]["symbols"]


def cmd_ingest(args, cfg):
    from src.data.ingestion import download_all
    symbols = _get_symbols(args, cfg)
    logger.info("Ingesting data for: %s", symbols)
    download_all(symbols=symbols)


def cmd_validate(args, cfg):
    from src.data.ingestion import download_all
    from src.data.validation import validate_all_raw
    symbols = _get_symbols(args, cfg)
    raw = download_all(symbols=symbols)
    validate_all_raw(raw)
    logger.info("Validation complete.")


def cmd_features(args, cfg):
    from src.data.ingestion import download_all
    from src.data.validation import validate_all_raw
    from src.data.features import engineer_all
    symbols = _get_symbols(args, cfg)
    raw = download_all(symbols=symbols)
    raw = validate_all_raw(raw)
    engineer_all(raw)
    logger.info("Feature engineering complete.")


def cmd_train(args, cfg):
    symbols = _get_symbols(args, cfg)
    skip_ag = getattr(args, "skip_autogluon", False)
    if args.trainer == "mlflow":
        from src.training.mlflow_trainer import train_all_mlflow
        train_all_mlflow(symbols=symbols, skip_autogluon=skip_ag)
    else:
        from src.training.manual_trainer import train_all
        train_all(symbols=symbols, skip_autogluon=skip_ag)


def cmd_compare(args, cfg):
    from src.evaluation.evaluator import compare_all
    symbols = _get_symbols(args, cfg)
    compare_all(symbols=symbols)


def cmd_monitor(args, cfg):
    from src.evaluation.monitoring import run_drift_report
    symbols = _get_symbols(args, cfg)
    paths = run_drift_report(symbols=symbols)
    for sym, path in paths.items():
        logger.info("%s drift report: %s", sym, path)


def cmd_predict(args, cfg):
    from src.data.features import load_processed
    from src.slm.sentiment import get_live_sentiment
    from src.slm.explainer import explain_prediction
    import pickle, numpy as np

    symbol = (args.symbol or cfg["stocks"]["symbols"][0]).upper()
    model_path = Path(cfg["paths"]["manual_models"]) / f"{symbol}_ensemble.pkl"
    if not model_path.exists():
        logger.error("No trained model found for %s. Run: python main.py --mode train --trainer manual", symbol)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_processed(symbol)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    latest_X = X[-1:].reshape(1, -1)

    proba = model.predict_proba(latest_X)[0]
    signal_class = int(np.argmax(proba))
    confidence = float(proba[signal_class])
    signal_map = {4: "Strong Buy", 3: "Buy", 2: "Hold", 1: "Sell", 0: "Strong Sell"}

    sent = get_live_sentiment(symbol)
    try:
        top_feat = model.top_features(X[-50:], n=5)
    except Exception:
        top_feat = {}

    explanation = explain_prediction(
        symbol=symbol,
        signal_class=signal_class,
        confidence=confidence,
        top_features=top_feat,
        sentiment_score=sent.get("sentiment_score", 0.0),
    )

    print(f"\n{'='*60}")
    print(f"  PREDICTION: {symbol}")
    print(f"{'='*60}")
    print(f"  Signal:      {signal_map[signal_class]}")
    print(f"  Confidence:  {confidence:.1%}")
    print(f"  Data as of:  {df.index[-1].date()}")
    print(f"\n  Probabilities:")
    labels = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    for i, (label, p) in enumerate(zip(labels, proba)):
        bar = "█" * int(p * 30)
        marker = " ◄" if i == signal_class else ""
        print(f"    {label:<12} {p:6.1%}  {bar}{marker}")
    print(f"\n  Sentiment:   score={sent.get('sentiment_score', 0):.2f}  news={int(sent.get('news_count_7d', 0))} headlines")
    print(f"\n  Explanation: {explanation}")
    print(f"{'='*60}\n")


def cmd_serve(args, cfg):
    import uvicorn
    host = cfg["api"]["host"]
    port = cfg["api"]["port"]
    logger.info("Starting FastAPI server at http://%s:%d", host, port)
    uvicorn.run("src.api.app:app", host=host, port=port, reload=cfg["api"]["reload"])


def cmd_full_pipeline(args, cfg):
    logger.info("Running full pipeline ...")
    cmd_ingest(args, cfg)
    cmd_validate(args, cfg)
    cmd_features(args, cfg)
    cmd_train(args, cfg)
    cmd_compare(args, cfg)
    logger.info("Full pipeline complete.")


# ── PLAN v3 — intraday selective signal system (PLAN_v3.md) ──────────

def cmd_backfill(args, cfg):
    from src.intraday.data_feed import backfill
    backfill(symbols=args.symbols, years=args.years)


def cmd_bhavcopy(args, cfg):
    from datetime import date, timedelta
    from src.intraday.bhavcopy import backfill_bhavcopies, fetch_fii_dii
    from src.intraday import load_config as v3cfg
    years = args.years or v3cfg()["data"]["backfill_years"]
    start = date.today() - timedelta(days=int(years * 365.25))
    backfill_bhavcopies(start, date.today())
    fetch_fii_dii()


def cmd_record(args, cfg):
    from src.intraday.recorder import run
    run()


def cmd_screen(args, cfg):
    from datetime import date
    from src.intraday.screener import screen_day
    day = date.fromisoformat(args.date) if args.date else date.today()
    top = screen_day(day)
    print(top.to_string(index=False))


def cmd_backtest(args, cfg):
    import json
    from datetime import date, timedelta
    from src.intraday.backtester import run_backtest
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=3 * 365)
    report = run_backtest(start, end, stress=args.stress)
    print(json.dumps(report, indent=2, default=str))


def cmd_paper(args, cfg):
    from src.intraday.paper_runner import run
    run(capital_inr=args.capital)


def main():
    import yaml
    cfg_path = Path(__file__).parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="AI-MLOps Stock Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode ingest
  python main.py --mode ingest --symbols RELIANCE.NS TCS.NS
  python main.py --mode validate
  python main.py --mode features
  python main.py --mode train --trainer manual
  python main.py --mode train --trainer mlflow
  python main.py --mode train --trainer manual --symbols TCS.NS
  python main.py --mode compare
  python main.py --mode monitor
  python main.py --mode predict --symbol RELIANCE.NS
  python main.py --mode serve
  python main.py --mode full-pipeline --trainer manual
  python main.py --mode full-pipeline --trainer mlflow
        """,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["ingest", "validate", "features", "train", "compare",
                 "monitor", "predict", "serve", "full-pipeline",
                 # PLAN v3 intraday modes
                 "backfill", "bhavcopy", "record", "screen", "backtest", "paper"],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--trainer",
        choices=["manual", "mlflow"],
        default="manual",
        help="Training mode: manual (local artifacts) or mlflow (tracked)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single stock symbol (e.g. RELIANCE.NS)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Multiple stock symbols (e.g. --symbols RELIANCE.NS TCS.NS)",
    )
    parser.add_argument(
        "--skip-autogluon",
        action="store_true",
        default=False,
        help="Skip AutoGluon training (faster, especially on first run)",
    )
    # ── PLAN v3 arguments ────────────────────────────────────────────
    parser.add_argument("--years", type=int, default=None,
                        help="[v3 backfill/bhavcopy] years of history to download")
    parser.add_argument("--date", type=str, default=None,
                        help="[v3 screen] date YYYY-MM-DD (default today)")
    parser.add_argument("--start", type=str, default=None,
                        help="[v3 backtest] start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="[v3 backtest] end date YYYY-MM-DD")
    parser.add_argument("--stress", action="store_true", default=False,
                        help="[v3 backtest] 2x slippage stress mode (Gate 3)")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="[v3 paper] paper-trading capital in INR")

    args = parser.parse_args()

    dispatch = {
        "ingest":        cmd_ingest,
        "validate":      cmd_validate,
        "features":      cmd_features,
        "train":         cmd_train,
        "compare":       cmd_compare,
        "monitor":       cmd_monitor,
        "predict":       cmd_predict,
        "serve":         cmd_serve,
        "full-pipeline": cmd_full_pipeline,
        # PLAN v3 intraday modes
        "backfill":      cmd_backfill,
        "bhavcopy":      cmd_bhavcopy,
        "record":        cmd_record,
        "screen":        cmd_screen,
        "backtest":      cmd_backtest,
        "paper":         cmd_paper,
    }

    dispatch[args.mode](args, cfg)


if __name__ == "__main__":
    main()
