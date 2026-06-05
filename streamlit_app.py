import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

from src.data_loader import get_stock_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.arima_model import train_arima, forecast_arima
from src.evaluation import evaluate

from src.lstm_model import (
    create_sequences,
    train_lstm,
    predict_lstm
)

from src.data_preprocessing import (
    preprocess_data,
    scale_data
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Market Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 Stock Market Forecasting Dashboard")

st.markdown("""
Forecast stock prices using historical Yahoo Finance data
and ARIMA time-series forecasting.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Configuration")

ticker = st.sidebar.selectbox(
    "Select Stock",
    [
        "AAPL",
        "MSFT",
        "TSLA",
        "GOOGL"
    ]
)

model_type = st.sidebar.selectbox(
    "Forecast Model",
    [
        "ARIMA",
        "LSTM"
    ]
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2015, 1, 1)
)

run_button = st.sidebar.button(
    "Run Forecast"
)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if run_button:

    try:

        # ------------------------------------------
        # LSTM Placeholder
        # ------------------------------------------



        # ------------------------------------------
        # DATA LOADING
        # ------------------------------------------

        with st.spinner(
            f"Downloading {ticker} stock data..."
        ):

            df = get_stock_data(
                ticker=ticker,
                start=start_date
            )

        st.success(
            f"{ticker} data loaded successfully"
        )

        # ------------------------------------------
        # DATA PREVIEW
        # ------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head()
        )

        # ------------------------------------------
        # HISTORICAL PRICE CHART
        # ------------------------------------------

        st.subheader(
            f"{ticker} Historical Stock Price"
        )

        fig = px.line(
            df,
            x="Date",
            y="Close",
            title=f"{ticker} Historical Price Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ------------------------------------------
        # PREPROCESSING
        # ------------------------------------------

        df = preprocess_data(df)

        df = create_features(df)

        if model_type == "LSTM":
            with st.spinner(
                    "Training LSTM model..."
            ):
                scaled_data, scaler = scale_data(df)

                X, y = create_sequences(
                    scaled_data,
                    window=60
                )

                split = int(
                    len(X) * 0.8
                )

                X_train = X[:split]
                X_test = X[split:]

                y_train = y[:split]
                y_test = y[split:]

                model = train_lstm(
                    X_train,
                    y_train
                )

                predictions = predict_lstm(
                    model,
                    X_test
                )

                predictions = scaler.inverse_transform(
                    predictions
                )

                y_test = scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                )

                rmse, mape, r2 = evaluate(
                    y_test.flatten(),
                    predictions.flatten()
                )

                st.subheader(
                    "LSTM Model Performance"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "RMSE",
                    f"{rmse:.2f}"
                )

                c2.metric(
                    "MAPE",
                    f"{mape:.2f}%"
                )

                c3.metric(
                    "R² Score",
                    f"{r2:.4f}"
                )

                lstm_dates = df.index[-len(y_test):]

                results = pd.DataFrame({
                    "Date": lstm_dates,
                    "Actual": y_test.flatten(),
                    "Predicted": predictions.flatten()
                })

                # Save LSTM forecasts
                results.to_csv(
                    "outputs/lstm_forecasts.csv",
                    index=False
                )

                # Save LSTM metrics
                metrics_df = pd.DataFrame({
                    "RMSE": [rmse],
                    "MAPE": [mape],
                    "R2": [r2]
                })

                metrics_df.to_csv(
                    "outputs/lstm_metrics.csv",
                    index=False
                )

                comparison_df = pd.DataFrame({
                    "Model": ["LSTM"],
                    "RMSE": [rmse],
                    "MAPE": [mape],
                    "R2": [r2]
                })

                comparison_df.to_csv(
                    "outputs/model_comparison.csv",
                    index=False
                )

                lstm_fig = px.line(
                    results,
                    x="Date",
                    y=["Actual", "Predicted"],
                    title="LSTM Actual vs Predicted"
                )

                st.plotly_chart(
                    lstm_fig,
                    use_container_width=True
                )

                csv = results.to_csv(index=False)

                st.download_button(
                    label="📥 Download LSTM Forecast CSV",
                    data=csv,
                    file_name=f"{ticker}_lstm_forecast.csv",
                    mime="text/csv"
                )



                st.stop()

        # ------------------------------------------
        # TRAIN TEST SPLIT
        # ------------------------------------------

        train_size = int(
            len(df) * 0.8
        )

        train = df["Close"][:train_size]

        test = df["Close"][train_size:]

        st.info(
            f"Training Samples: {len(train)} | "
            f"Test Samples: {len(test)}"
        )

        # ------------------------------------------
        # ARIMA MODEL
        # ------------------------------------------

        with st.spinner(
            "Training ARIMA model..."
        ):

            model = train_arima(
                train
            )

            predictions = forecast_arima(
                model,
                len(test)
            )

        # ------------------------------------------
        # CLEAN PREDICTIONS
        # ------------------------------------------

        predictions = pd.Series(
            predictions
        )

        predictions = predictions.ffill()

        predictions = pd.Series(
            predictions.values,
            index=test.index
        )

        if predictions.isna().sum() > 0:

            st.error(
                "Predictions contain NaN values."
            )

            st.stop()

        # ------------------------------------------
        # EVALUATION
        # ------------------------------------------

        rmse, mape, r2 = evaluate(
            test,
            predictions
        )

        st.subheader(
            "ARIMA Model Performance"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        col2.metric(
            "MAPE",
            f"{mape:.2f}%"
        )

        col3.metric(
            "R² Score",
            f"{r2:.4f}"
        )

        # ------------------------------------------
        # FORECAST RESULTS
        # ------------------------------------------

        results = pd.DataFrame({
            "Date": test.index,
            "Actual": test,
            "Predicted": predictions
        })

        st.info(
            f"Forecast Horizon: {len(test)} trading days"
        )

        st.subheader(
            "Actual vs Predicted"
        )

        forecast_fig = px.line(
            results,
            x="Date",
            y=["Actual", "Predicted"],
            title="Actual vs Predicted"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        # ------------------------------------------
        # SAVE OUTPUTS
        # ------------------------------------------

        results.to_csv(
            "outputs/forecasts.csv",
            index=False
        )

        metrics_df = pd.DataFrame({
            "RMSE": [rmse],
            "MAPE": [mape],
            "R2": [r2]
        })

        metrics_df.to_csv(
            "outputs/metrics.csv",
            index=False
        )

        comparison_df = pd.DataFrame({
            "Model": ["ARIMA"],
            "RMSE": [rmse],
            "MAPE": [mape],
            "R2": [r2]
        })

        comparison_df.to_csv(
            "outputs/model_comparison.csv",
            index=False
        )

        # ------------------------------------------
        # DOWNLOAD BUTTON
        # ------------------------------------------

        csv = results.to_csv(index=False)

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.exception(e)
