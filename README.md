# 📈 Time Series Forecasting System

A Machine Learning and Deep Learning based Stock Market Forecasting System that predicts stock prices using historical market data. The project implements both **ARIMA** and **LSTM** forecasting models and provides an interactive **Streamlit Dashboard** for visualization and performance comparison.

---

## 🚀 Features

* Historical stock data collection using Yahoo Finance
* Data preprocessing and feature engineering
* ARIMA-based time series forecasting
* LSTM-based deep learning forecasting
* Performance evaluation using RMSE, MAPE, and R² Score
* Interactive Streamlit dashboard
* Actual vs Predicted visualization
* Download forecast results as CSV
* Model comparison between ARIMA and LSTM

---

## 📂 Project Structure

```text
Time-Series-Forecasting-System
│
├── outputs/
│   ├── forecasts.csv
│   ├── lstm_forecasts.csv
│   ├── metrics.csv
│   ├── lstm_metrics.csv
│   └── model_comparison.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── arima_model.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── lstm_model.py
│   └── visualization.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* TensorFlow / Keras
* Plotly
* Streamlit
* yfinance
* Matplotlib

---

## 📊 Data Preprocessing

The following preprocessing techniques are applied:

* Missing value handling
* Date parsing and indexing
* Data normalization using MinMaxScaler
* Lag feature generation
* Rolling mean and rolling standard deviation
* Moving averages (MA)
* Exponential moving averages (EMA)
* Volatility calculation

---

## 📈 Forecasting Models

### ARIMA Model

ARIMA is used to capture linear temporal dependencies in stock prices.

Features:

* Automatic parameter selection using Auto ARIMA
* Time-series forecasting
* Performance evaluation

### LSTM Model

Long Short-Term Memory (LSTM) neural network is used to capture long-term temporal patterns.

Architecture:

* LSTM Layer (64 units)
* Dropout Layer
* LSTM Layer (64 units)
* Dropout Layer
* Dense Output Layer

---

## 📉 Evaluation Metrics

The forecasting models are evaluated using:

### RMSE

Root Mean Squared Error

### MAPE

Mean Absolute Percentage Error

### R² Score

Coefficient of Determination

---

## 🖥 Dashboard Features

* Select stock ticker
* Select forecasting model (ARIMA or LSTM)
* Interactive stock price visualization
* Model performance metrics
* Actual vs Predicted comparison chart
* Forecast CSV download

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Time-Series-Forecasting-System.git
cd Time-Series-Forecasting-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser.

---

## 📋 Supported Stocks

Currently supported:

* AAPL
* MSFT
* TSLA
* GOOGL

Additional tickers can easily be added.

---

## 📸 Sample Outputs

### Historical Stock Price Visualization

Interactive Plotly chart displaying historical trends.

### Model Performance

| Metric | Description      |
| ------ | ---------------- |
| RMSE   | Prediction Error |
| MAPE   | Percentage Error |
| R²     | Goodness of Fit  |

### Forecast Export

Generated files:

```text
forecasts.csv
lstm_forecasts.csv
metrics.csv
lstm_metrics.csv
model_comparison.csv
```

---

## 🎯 Future Improvements

* GRU Forecasting Model
* Prophet Integration
* Hyperparameter Optimization
* Multi-stock Portfolio Forecasting
* Real-time Market Data Streaming
* ARIMA vs LSTM Comparative Dashboard
* Deployment on Streamlit Cloud

---

## 👨‍💻 Author

Ajay Kumar

BCA Graduate | Networking & Machine Learning Enthusiast

---

## 📄 License

This project is intended for educational and research purposes.
