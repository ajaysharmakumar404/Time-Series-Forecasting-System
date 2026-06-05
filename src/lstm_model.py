import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

def create_sequences(
    data,
    window=60
):

    X = []
    y = []

    for i in range(window, len(data)):

        X.append(
            data[i-window:i]
        )

        y.append(
            data[i]
        )

    return np.array(X), np.array(y)


def build_lstm(
    input_shape
):

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(64)
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_lstm(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
):

    model = build_lstm(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model


def predict_lstm(
    model,
    X_test
):

    predictions = model.predict(
        X_test,
        verbose=0
    )

    return predictions