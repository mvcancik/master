import argparse

from .data import prepare_dataset
from .features import add_technical_features, add_calendar_features
from .models import train_lightgbm, train_lstm
from .backtest import rolling_window_backtest


def run_train(args):
    df = prepare_dataset(args.ticker)
    df = add_technical_features(df)
    df = add_calendar_features(df)
    if args.model == "lightgbm":
        model, rmse = train_lightgbm(df)
    else:
        model, rmse = train_lstm(df, epochs=args.epochs)
    print(f"Training completed. RMSE={rmse:.4f}")


def run_backtest(args):
    _, rmse = rolling_window_backtest(args.ticker, model_type=args.model)
    print(f"Backtest RMSE={rmse:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock prediction toolkit")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("ticker", help="Ticker symbol")
    train_p.add_argument("--model", choices=["lightgbm", "lstm"], default="lightgbm")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.set_defaults(func=run_train)

    back_p = subparsers.add_parser("backtest", help="Run rolling window backtest")
    back_p.add_argument("ticker", help="Ticker symbol")
    back_p.add_argument("--model", choices=["lightgbm", "lstm"], default="lightgbm")
    back_p.set_defaults(func=run_backtest)

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
