# train_models.py
import sys
from train import train_model_for_symbol, STOCK_LIST

def show_usage():
    print("""
Usage:
    python train_models.py all
    python train_models.py AAPL
    python train_models.py AAPL GOOGL TSLA
""")

if __name__ == "__main__":
    # No arguments → show help
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(0)

    args = sys.argv[1:]

    # Train all stocks
    if args[0].lower() == "all":
        print("🚀 Training ALL stocks...")
        for symbol in STOCK_LIST:
            train_model_for_symbol(symbol)
        print("✅ Training completed for all symbols")
        sys.exit(0)

    # Train selected stocks
    else:
        print("🚀 Training selected stocks:", args)
        for symbol in args:
            try:
                train_model_for_symbol(symbol.upper())
            except Exception as e:
                print(f"❌ Error training {symbol}: {e}")
        print("✅ Training completed for selected symbols")
