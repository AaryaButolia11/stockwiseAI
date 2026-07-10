import yfinance as yf

for symbol in [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
]:
    print(f"\n{symbol}")
    print(yf.Ticker(symbol).history(period="5d"))