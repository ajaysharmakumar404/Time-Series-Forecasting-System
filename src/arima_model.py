from pmdarima import auto_arima
import pandas as pd

def train_arima(train):

    model = auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=True
    )

    return model


def forecast_arima(model, steps):
    forecast = model.predict(
        n_periods=steps
    )

    forecast = pd.Series(
        forecast
    )

    forecast = forecast.fillna(
        method="ffill"
    )

    return forecast