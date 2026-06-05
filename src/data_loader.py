import yfinance as yf

def get_stock_data(
    ticker,
    start
):

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True
    )

    df.reset_index(inplace=True)

    if hasattr(df.columns, "levels"):

        df.columns = [
            col[0]
            if isinstance(col, tuple)
            else col
            for col in df.columns
        ]

    return df