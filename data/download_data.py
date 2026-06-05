import yfinance as yf

ticker = "AAPL"

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2025-12-31",
    auto_adjust=True
)

print(df.head())
print("Rows:", len(df))

df.to_csv("data/AAPL.csv")

print("Saved successfully.")