import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

def evaluate(y_true, y_pred):

    y_true = pd.Series(y_true)

    y_pred = pd.Series(y_pred)

    mask = (
        ~y_true.isna()
        &
        ~y_pred.isna()
    )

    y_true = y_true[mask]

    y_pred = y_pred[mask]

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        mean_absolute_percentage_error(
            y_true,
            y_pred
        ) * 100
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return rmse, mape, r2