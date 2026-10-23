"""Command-line research evaluation with an explicit completed-data cutoff."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import forecast_next_open, walk_forward
from .data import prepare_dataset, validate_prices


def _load(args):
    if args.csv:
        prices = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        prices = prices.loc[prices.index < pd.Timestamp(args.end)]
        return validate_prices(prices)
    return prepare_dataset(args.ticker, end=args.end)


def run_train(args):
    forecast = forecast_next_open(
        _load(args), horizon=args.horizon, model_type=args.model,
        window=args.window, min_train=args.min_train,
        lookback=args.lookback, epochs=args.epochs)
    print(json.dumps(forecast, indent=2, allow_nan=False))


def run_backtest(args):
    result = walk_forward(
        _load(args), horizon=args.horizon, model_type=args.model,
        window=args.window, min_train=args.min_train, refit_every=args.refit_every,
        lookback=args.lookback, epochs=args.epochs,
        cost_bps=args.cost_bps, threshold=args.threshold)
    print(json.dumps(result.metrics, indent=2, allow_nan=False))
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(output / "predictions.csv")
        result.trades.to_csv(output / "trades.csv", index=False)
        result.equity.to_csv(output / "equity.csv")
        (output / "metrics.json").write_text(
            json.dumps(result.metrics, indent=2, allow_nan=False) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Short-horizon stock research")
    commands = parser.add_subparsers(dest="command")
    for command in ("train", "backtest"):
        sub = commands.add_parser(command)
        sub.add_argument("ticker", help="Single ticker (also a label with --csv)")
        sub.add_argument("--end", required=True,
                         help="Exclusive YYYY-MM-DD cutoff; all earlier bars must be complete")
        sub.add_argument("--csv", help="Frozen daily OHLCV CSV; first column is session date")
        sub.add_argument("--model", choices=["lightgbm", "ridge", "lstm"], default="lightgbm")
        sub.add_argument("--horizon", type=int, choices=[1, 2, 3], default=1)
        sub.add_argument("--window", type=int, default=252)
        sub.add_argument("--min-train", type=int, default=126)
        sub.add_argument("--lookback", type=int, default=20)
        sub.add_argument("--epochs", type=int, default=10)
        if command == "backtest":
            sub.add_argument("--refit-every", type=int, default=1)
            sub.add_argument("--cost-bps", type=float, default=10.0,
                             help="Assumed execution cost in basis points PER SIDE")
            sub.add_argument("--threshold", type=float, default=0.0,
                             help="Minimum predicted NET return, decimal units")
            sub.add_argument("--output", help="Write predictions, trades, daily equity and metrics")
            sub.set_defaults(func=run_backtest)
        else:
            sub.set_defaults(func=run_train)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
