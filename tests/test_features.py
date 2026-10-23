import numpy as np
import pandas as pd
import pytest

from stockpredictor.features import (
    FEATURE_COLUMNS, TARGET, compute_rsi, features_after_close,
    make_samples, sequence_inputs,
)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_exact_tradeable_target_and_tail(prices, horizon):
    samples = make_samples(prices, horizon)
    d = 250
    assert samples.iloc[d][TARGET] == pytest.approx(
        prices.Close.iloc[d + horizon - 1] / prices.Open.iloc[d] - 1)
    assert samples.iloc[d].feature_asof == prices.index[d - 1]
    assert samples.iloc[d].label_end == prices.index[d + horizon - 1]
    assert samples[TARGET].isna().sum() == horizon - 1
    assert samples.label_end.isna().sum() == horizon - 1


def test_all_features_lagged_and_future_changes_invariant(prices):
    cutoff = 270
    original = make_samples(prices, 3)
    altered = prices.copy()
    altered.iloc[cutoff:, :] *= 3.0
    changed = make_samples(altered, 3)
    pd.testing.assert_frame_equal(
        original.loc[:prices.index[cutoff], list(FEATURE_COLUMNS)],
        changed.loc[:prices.index[cutoff], list(FEATURE_COLUMNS)])
    pd.testing.assert_series_equal(
        original.loc[prices.index[cutoff], list(FEATURE_COLUMNS)].astype(float),
        features_after_close(prices).iloc[cutoff - 1], check_names=False)
    assert original.iloc[cutoff][TARGET] != original.iloc[cutoff - 1][TARGET]


def test_trailing_features_can_be_precomputed(prices):
    full = features_after_close(prices)
    prefix = features_after_close(prices.iloc[:270])
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)


@pytest.mark.parametrize("values, expected", [
    (np.repeat(100.0, 25), 50.0),
    (np.arange(1.0, 26.0), 100.0),
    (np.arange(25.0, 0.0, -1), 0.0),
])
def test_rsi_edges(values, expected):
    actual = compute_rsi(pd.Series(values))
    assert actual.iloc[:14].isna().all()
    assert (actual.iloc[14:] == expected).all()


def test_sequences_keep_exact_context_and_do_not_compact_warmup(prices):
    samples = make_samples(prices, 3)
    X = sequence_inputs(samples, 20)
    assert X.shape == (len(prices), 20, len(FEATURE_COLUMNS))
    assert not np.isfinite(X[218]).all()
    assert np.isfinite(X[219]).all()
    np.testing.assert_allclose(
        X[250], samples.iloc[231:251][list(FEATURE_COLUMNS)].to_numpy())
    assert TARGET not in FEATURE_COLUMNS
    assert "entry_price" not in FEATURE_COLUMNS


@pytest.mark.parametrize("horizon", [0, 4, -1, True, 1.5])
def test_reject_invalid_horizon(prices, horizon):
    with pytest.raises(ValueError, match="horizon"):
        make_samples(prices, horizon)
