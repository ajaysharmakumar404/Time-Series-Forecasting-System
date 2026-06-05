import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df.set_index(
        "Date",
        inplace=True
    )

    return df


def scale_data(df):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        df[["Close"]]
    )

    return scaled, scaler