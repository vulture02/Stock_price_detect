import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------ Load Model ------------------ #
model = load_model(r'D:\5th sem notes\MINORPROJECTSENGINERRING\stockprice\stock_predictions Model.keras')

# ------------------ Page Config ------------------ #
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide",
    page_icon="📈"
)

# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.title("📊 Menu")
    stock = st.text_input('Enter Stock Ticker Symbol:', 'GOOG')
    start = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
    end = st.date_input('End Date', pd.to_datetime('2023-10-01'))
    st.markdown("---")
    st.info("🔍 Predicting stock prices using a deep learning model (LSTM).")

# ------------------ Header ------------------ #
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #004488;
            text-align: center;
            margin-bottom: 20px;
        }

        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f0f0;
            color: #666;
            text-align: center;
            font-size: 0.9rem;
            padding: 12px 10px;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, color 0.3s ease, font-weight 0.3s ease;
            z-index: 999;
        }

        .footer:hover {
            background-color: #004488;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
        }
    </style>

    <div class="main-title">📈 Stock Price Prediction App</div>
    <div class="footer">🚀 Made with ❤️ by Amith | Hover me!</div>
    """,
    unsafe_allow_html=True
)


# ------------------ Load Data ------------------ #
data = yf.download(stock, start=start, end=end)

if data.empty:
    st.error("❌ No data found! Please check the ticker symbol.")
    st.stop()

# ------------------ Show Data ------------------ #
st.subheader(f'📅 Historical Stock Data ({start} to {end})')
st.dataframe(data.tail(), use_container_width=True)

# ------------------ Data Preprocessing ------------------ #
data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(data_train)

past_100_days = data_train.tail(100)
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
scaled_test_data = scaler.transform(final_test_data)

# ------------------ Moving Averages ------------------ #
ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

# ------------------ Charts ------------------ #
st.subheader('📊 Visual Analysis')
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Price vs MA50**")
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Close Price', color='green')
    plt.plot(ma_50, label='MA50', color='red')
    plt.legend()
    st.pyplot(fig1)

with col2:
    st.markdown("**Price vs MA50 vs MA100**")
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Close Price', color='green')
    plt.plot(ma_50, label='MA50', color='red')
    plt.plot(ma_100, label='MA100', color='blue')
    plt.legend()
    st.pyplot(fig2)

# Full-width chart
st.markdown("**Price vs MA100 vs MA200**")
fig3 = plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Close Price', color='green')
plt.plot(ma_100, label='MA100', color='red')
plt.plot(ma_200, label='MA200', color='orange')
plt.legend()
st.pyplot(fig3)

# ------------------ Prediction ------------------ #
x_test, y_test = [], []
for i in range(100, scaled_test_data.shape[0]):
    x_test.append(scaled_test_data[i-100:i])
    y_test.append(scaled_test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Prediction
predictions = model.predict(x_test)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# ------------------ Predicted Chart ------------------ #
st.subheader('📉 Actual Price vs Predicted Price')
fig4 = plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(predictions, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'{stock} Stock Price Prediction')
plt.legend()
st.pyplot(fig4)

# ------------------ Footer ------------------ #
st.markdown(
    """
    <div class="footer">
        Stock Price Prediction App | Created by Amith
    </div>
    """,
    unsafe_allow_html=True
)
