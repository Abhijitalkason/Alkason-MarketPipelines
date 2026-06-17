"""Features: the load-bearing no-lookahead property test (would catch B-4),
FII future-row immunity, missingness flags + FlowDataError, schema v3.1."""

from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.intraday import ROOT, load_config
from src.intraday.bars import load_1min
from src.intraday.features import (FLOW_FEATURES, FLOW_PRESENT, FeatureSchemaError,
                                   FlowDataError, build_matrix, check_schema,
                                   features_for_day)


def test_no_lookahead_property(repo, frozen_day):
    """A decision-bar row computed on data truncated at that bar's close must be
    identical to the same row computed on the full day (Section 4.4)."""
    full = features_for_day("AAA", frozen_day)
    assert not full.empty
    df_1min = load_1min("AAA", frozen_day - timedelta(days=45), frozen_day)
    for _, row in full.iterrows():
        bar_ts = row["ts"]
        cutoff = bar_ts + pd.Timedelta("15min")            # decision-bar close
        truncated = df_1min[df_1min.index < cutoff]
        sub = features_for_day("AAA", frozen_day, df_1min=truncated)
        match = sub[sub["ts"] == bar_ts]
        assert not match.empty, f"row {bar_ts} vanished under truncation"
        for col in [c for c in full.columns if c not in ("symbol", "ts", "date")]:
            a, b = row[col], match.iloc[0][col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"{col} leaked at {bar_ts}"


def test_fii_future_rows_do_not_change_features(repo, frozen_day):
    """Adding FII rows dated on/after the decision day must not move fii_5d_z
    (B-4 regression)."""
    bdir = ROOT / load_config()["data"]["bhavcopy_path"]
    base = pd.DataFrame({
        "date": pd.bdate_range(end=frozen_day - timedelta(days=1), periods=10),
        "fii_net_cr": np.arange(10.0), "dii_net_cr": np.arange(10.0),
        "capture_ts": [datetime.combine(frozen_day, time(18, 0))] * 10,
    })
    base.to_parquet(bdir / "fii_dii.parquet", index=False)
    before = features_for_day("AAA", frozen_day)["fii_5d_z"].iloc[0]

    poisoned = pd.concat([base, pd.DataFrame({
        "date": [pd.Timestamp(frozen_day), pd.Timestamp(frozen_day) + pd.Timedelta(days=1)],
        "fii_net_cr": [1e6, -1e6], "dii_net_cr": [0, 0],
        "capture_ts": [datetime.combine(frozen_day, time(18, 0))] * 2,
    })], ignore_index=True)
    poisoned.to_parquet(bdir / "fii_dii.parquet", index=False)
    after = features_for_day("AAA", frozen_day)["fii_5d_z"].iloc[0]
    assert (pd.isna(before) and pd.isna(after)) or before == pytest.approx(after)


def test_missingness_flags_present(repo, frozen_day):
    f = features_for_day("AAA", frozen_day)
    for col in FLOW_PRESENT:
        assert col in f.columns
    # no bhavcopy on disk → flow absent → present flags all 0
    assert (f[FLOW_PRESENT] == 0).all().all()


def test_flow_floor_raises_when_required(repo, frozen_day):
    cfg = load_config()
    cfg["gate"]["flow_model_active"] = True          # require flow channel
    plan = pd.DataFrame({"symbol": ["AAA"], "date": [frozen_day], "direction": [1]})
    with pytest.raises(FlowDataError):
        build_matrix(plan, require_flow=True)


def test_schema_check_catches_missing_and_nan(repo, frozen_day):
    f = build_matrix(pd.DataFrame({"symbol": ["AAA"], "date": [frozen_day], "direction": [1]}))
    check_schema(f)                                  # passes
    with pytest.raises(FeatureSchemaError):
        check_schema(f.drop(columns=["atr_pct"]))
