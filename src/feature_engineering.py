def create_features(df):

    df['lag_1'] = df['Close'].shift(1)

    df['lag_7'] = df['Close'].shift(7)

    df['lag_30'] = df['Close'].shift(30)

    df['rolling_mean_7'] = (
        df['Close'].rolling(7).mean()
    )

    df['rolling_std_7'] = (
        df['Close'].rolling(7).std()
    )

    df['MA_10'] = (
        df['Close']
        .rolling(10)
        .mean()
    )

    df['MA_50'] = (
        df['Close']
        .rolling(50)
        .mean()
    )

    df['EMA_10'] = (
        df['Close']
        .ewm(span=10)
        .mean()
    )

    df['Volatility'] = (
        df['Close']
        .rolling(20)
        .std()
    )

    return df.dropna()