import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">Stock Market Predictor</p>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

# Model loading
@st.cache_resource
def load_ml_model():
    try:
        return load_model('C:\Python\Stock\Stock Predictions Model.keras')
    except FileNotFoundError:
        st.error("Error: The model file 'Stock Predictions Model.keras' could not be found. Please check the file path and ensure the model exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.stop()

model = load_ml_model()

# User inputs
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
today = datetime.today().strftime('%Y-%m-%d')
start = st.sidebar.date_input('Start date', datetime(2012, 1, 1))
end = st.sidebar.date_input('End date', datetime.strptime(today, '%Y-%m-%d'))

if start > end:
    st.sidebar.error('Error: End date must fall after start date.')
    st.stop()

# Data loading
@st.cache_data
def load_data(stock, start, end):
    try:
        data = yf.download(stock, start, end)
        if data.empty:
            st.error(f"Error: No data found for the stock symbol '{stock}'. Please check the symbol and try again.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error downloading stock data: {str(e)}")
        st.stop()

data = load_data(stock, start, end)

if len(data) < 200:
    st.error("Error: Insufficient data for analysis. The selected date range must contain at least 200 trading days.")
    st.stop()

# Data preparation
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Stock Data", "📊 Moving Averages", "🔮 Price Prediction"])

with tab1:
    st.header('Stock Data')
    st.dataframe(data.style.highlight_max(axis=0))

with tab2:
    st.header('Moving Averages')
    
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=ma_50_days, name="MA50"))
    fig.add_trace(go.Scatter(x=data.index, y=ma_100_days, name="MA100"))
    fig.add_trace(go.Scatter(x=data.index, y=ma_200_days, name="MA200"))

    fig.update_layout(title=f"{stock} Stock Price and Moving Averages",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      legend_title="Indicators")
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header('Price Prediction')

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x, y = np.array(x), np.array(y)

    try:
        predict = model.predict(x)
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.stop()

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_test.index[-len(y):], y=y, name="Original Price"))
    fig.add_trace(go.Scatter(x=data_test.index[-len(predict):], y=predict.flatten(), name="Predicted Price"))

    fig.update_layout(title=f"{stock} Original vs Predicted Stock Price",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      legend_title="Price Type")
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Created by StrawHats")