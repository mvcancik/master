import numpy as np
import pandas as pd
import pytest

from conftest import MeanRecorder
from stockpredictor.backtest import forecast_next_open, walk_forward
from stockpredictor.features import FEATURE_COLUMNS, TARGET, features_after_close, make_samples


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_purge_unavailable_labels_and_exact_training_rows(prices, horizon):
    fitted = []
    def factory():
        model = MeanRecorder()
        fitted.append(model)
        return model

    result = walk_forward(prices, horizon=horizon, window=40, min_train=30,
                          estimator_factory=factory)
    samples = make_samples(prices, horizon)
    predictions = result.predictions
    assert predictions.index.is_unique
    assert (predictions.train_max_label_end <= predictions.fit_asof).all()
    assert (predictions.fit_asof <= predictions.feature_asof).all()
    assert (predictions.feature_asof < predictions.index).all()
    assert predictions.index[-1] == prices.index[-horizon]
    for (_, row), model in zip(predictions.iterrows(), fitted):
        eligible = samples.loc[
            (samples.label_end <= row.fit_asof)
            & samples[list(FEATURE_COLUMNS)].notna().all(axis=1)
        ].tail(40)
        X, y = model.fit_calls[0]
        np.testing.assert_allclose(X, eligible[list(FEATURE_COLUMNS)])
        np.testing.assert_allclose(y, eligible[TARGET])
        assert len(y) == row.train_rows


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("model_type", ["ridge", "lightgbm"])
def test_future_mutation_cannot_change_earlier_predictions(prices, horizon, model_type):
    if model_type == "lightgbm":
        pytest.importorskip("lightgbm")
    kwargs = dict(horizon=horizon, window=50, min_train=30,
                  refit_every=7, model_type=model_type)
    result = walk_forward(prices, **kwargs)
    cutoff = prices.index[300]
    changed = prices.copy()
    changed.loc[cutoff:, :] *= 1.7
    other = walk_forward(changed, **kwargs)
    cols = ["prediction", "mean_prediction", "fit_asof", "train_max_label_end"]
    pd.testing.assert_frame_equal(
        result.predictions.loc[:cutoff, cols], other.predictions.loc[:cutoff, cols])


def test_appending_rows_preserves_existing_predictions(prices):
    kwargs = dict(horizon=3, min_train=30, window=50, refit_every=5, model_type="ridge")
    short = walk_forward(prices.iloc[:310], **kwargs)
    long = walk_forward(prices, **kwargs)
    pd.testing.assert_frame_equal(short.predictions, long.predictions.loc[short.predictions.index])


def test_lstm_gets_one_context_sequence_per_target(prices):
    fitted = []
    def factory():
        model = MeanRecorder()
        fitted.append(model)
        return model
    result = walk_forward(prices, model_type="lstm", horizon=3, lookback=20,
                          min_train=30, window=40, refit_every=9, estimator_factory=factory)
    assert len(result.predictions) > 0
    assert all(m.fit_calls[0][0].shape[1:] == (20, len(FEATURE_COLUMNS)) for m in fitted)
    assert all(x.shape == (1, 20, len(FEATURE_COLUMNS))
               for m in fitted for x in m.predict_calls)
    np.testing.assert_allclose(
        fitted[0].predict_calls[0][0, -1],
        features_after_close(prices).loc[result.predictions.iloc[0].feature_asof])


@pytest.mark.parametrize("model_type", ["ridge", "lstm"])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_live_next_open_matches_first_new_walk_forward_prediction(prices, model_type, horizon):
    kwargs = dict(horizon=horizon, model_type=model_type, min_train=30,
                  window=50, estimator_factory=MeanRecorder)
    forecast = forecast_next_open(prices.iloc[:300], **kwargs)
    historical = walk_forward(prices, **kwargs)
    assert forecast["predicted_gross_return"] == historical.predictions.loc[
        prices.index[300], "prediction"]


def test_metrics_match_hand_calculation(prices):
    result = walk_forward(prices, min_train=30, window=40, estimator_factory=MeanRecorder)
    actual = result.predictions.actual_return.to_numpy()
    pred = result.predictions.prediction.to_numpy()
    assert result.metrics["return_rmse"] == pytest.approx(np.sqrt(np.mean((pred - actual)**2)))
    assert result.metrics["zero_return_rmse"] == pytest.approx(np.sqrt(np.mean(actual**2)))
    assert result.metrics["past_mean_rmse"] == result.metrics["return_rmse"]


@pytest.mark.parametrize("model_type", ["ridge", "lightgbm"])
def test_real_live_and_historical_forecast_agree(prices, model_type):
    if model_type == "lightgbm":
        pytest.importorskip("lightgbm")
    kwargs = dict(horizon=3, model_type=model_type, min_train=30, window=50)
    live = forecast_next_open(prices.iloc[:300], **kwargs)
    historical = walk_forward(prices.iloc[:303], **kwargs)
    assert live["predicted_gross_return"] == pytest.approx(
        historical.predictions.loc[prices.index[300], "prediction"], abs=1e-12)


def test_insufficient_history_fails_clearly(prices):
    with pytest.raises(ValueError, match="Insufficient history"):
        walk_forward(prices.iloc[:220])


@pytest.mark.parametrize("kwargs", [
    {"refit_every": 0}, {"min_train": 253}, {"window": 0},
    {"model_type": "typo"}, {"model_type": "lstm", "lookback": 1},
    {"cost_bps": -1}, {"cost_bps": float("nan")}, {"threshold": -0.1},
])
def test_invalid_config(prices, kwargs):
    with pytest.raises(ValueError):
        walk_forward(prices, **kwargs)


@pytest.mark.parametrize("returned", [[float("nan")], [1, 2]])
def test_invalid_predictions_fail(prices, returned):
    class Invalid(MeanRecorder):
        def predict(self, X):
            return returned
    with pytest.raises(ValueError, match="one finite prediction"):
        walk_forward(prices, min_train=30, estimator_factory=Invalid)
