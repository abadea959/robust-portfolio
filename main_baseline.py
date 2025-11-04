from src.config import load_config

def main():
    cfg = load_config()
    print("Config loaded. Tickers:", cfg["tickers"])

if __name__ == "__main__":
    main()
