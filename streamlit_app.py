import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Stock Analysis App", layout="wide")

# Ensure you have the VADER lexicon downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the list of stock symbols and their names
stock_symbols = {
    "BP": "BP plc",
    "EQNR": "Equinor",
    "SU": "Suncor Energy",
    "E": "Eni",
    "VLO": "Valero Energy",
    "FANG": "Diamondback Energy",
    "CTRA": "Coterra Energy",
    "DINO": "HF Sinclair"
}

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period='5y', interval='1d'):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        return yf.download(tickers=symbol, period=period, interval=interval)
    except Exception as e:
        st.write(f"Error fetching stock data for {symbol}: {str(e)}")
        return None
@st.cache_data(ttl=3600)
def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train an LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=0)
    return model

@st.cache_data(ttl=3600)
def create_ds(dataset, step):
    """Create dataset for training."""
    Xtrain, Ytrain = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

def analyze_sentiment(texts):
    """Analyze sentiment of the provided texts."""
    return [sia.polarity_scores(text)['compound'] for text in texts]

def get_stock_news_from_csv(csv_file):
    """Read stock news from a CSV file."""
    try:
        if os.path.exists(csv_file):
            return pd.read_csv(csv_file)
        else:
            st.write("CSV file not found.")
            return None
    except Exception as e:
        st.write(f"Error reading CSV file: {str(e)}")
        return None

def main():
    page = st.sidebar.radio("Navigation", ["Stock Prediction", "Live Market", "Market Sentiment"])

    if page == "Stock Prediction":
        stock_prediction_page()
    elif page == "Live Market":
        live_market_page()
    elif page == "Market Sentiment":
        market_sentiment_page()

def stock_prediction_page():
    st.subheader("Stock Prediction")
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))

    if selected_stock:
        st.write(f"Stock Selected: {stock_symbols[selected_stock]} ({selected_stock})")
        
        data = fetch_stock_data(selected_stock)
        if data is None:
            return  # Stop if data fetch failed
        
        opn = data[['Open']].values
        normalizer = MinMaxScaler(feature_range=(0, 1))
        ds_scaled = normalizer.fit_transform(opn)

        train_size = int(len(ds_scaled) * 0.70)
        ds_train, ds_test = ds_scaled[:train_size], ds_scaled[train_size:]

        time_stamp = 100
        X_train, y_train = create_ds(ds_train, time_stamp)
        X_test, y_test = create_ds(ds_test, time_stamp)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        with st.spinner("Predicting... Please wait."):
            model = train_lstm_model(X_train, y_train, X_test, y_test)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = normalizer.inverse_transform(train_predict)
        test_predict = normalizer.inverse_transform(test_predict)

        fut_inp = ds_test[-time_stamp:].reshape(1, -1)
        tmp_inp = list(fut_inp[0])

        lst_output = []
        for _ in range(30):
            if len(tmp_inp) > time_stamp:
                fut_inp = np.array(tmp_inp[1:])
                fut_inp = fut_inp.reshape((1, time_stamp, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
            else:
                fut_inp = fut_inp.reshape((1, time_stamp, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())

        lst_output = normalizer.inverse_transform(lst_output)

        tab1, tab2 = st.tabs(["Stock Prediction", "Buy/Sell Recommendation"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=opn.flatten(), mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=30), y=lst_output.flatten(), mode='lines', name='Predicted Data', line=dict(color='red')))
            fig.update_layout(title=f'{stock_symbols[selected_stock]} Stock Price Prediction', xaxis_title='Time', yaxis_title='Stock Price')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            predicted_changes = np.diff(lst_output, axis=0)
            buy_percentage = np.sum(predicted_changes > 0) / len(predicted_changes) * 100
            sell_percentage = 100 - buy_percentage

            fig_recommendation = go.Figure(data=[go.Pie(labels=['Buy', 'Sell'], values=[buy_percentage, sell_percentage], hole=0.3)])
            fig_recommendation.update_layout(title='Buy/Sell Recommendation')
            st.plotly_chart(fig_recommendation, use_container_width=True)

            recommendation = 'Buy' if buy_percentage > sell_percentage else 'Sell'
            st.write(f"Recommendation: {recommendation}")
            st.write(f"Buy: {buy_percentage:.2f}%, Sell: {sell_percentage:.2f}%")

def live_market_page():
    st.subheader("Live Stock Market Data")
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))
    
    if selected_stock:
        ticker = yf.Ticker(selected_stock)
        live_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        st.write(f"Current price of {stock_symbols[selected_stock]} ({selected_stock}): ${live_price:.2f}")

        live_data = ticker.info
        st.write("Live Data:")
        for key in ['previousClose', 'open', 'dayLow', 'dayHigh', 'volume', 'averageVolume', 'marketCap']:
            st.write(f"{key}: {live_data.get(key, 'N/A')}")

def market_sentiment_page():
    st.subheader("Market Sentiment Analysis")
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))
    
    if selected_stock:
        st.write(f"Analyzing sentiment for: {stock_symbols[selected_stock]} ({selected_stock})")
        
        csv_file = 'resources/stock_news.csv'  # Update this path if necessary
        df = get_stock_news_from_csv(csv_file)

        if df is None or df.empty:
            st.write("No news articles found in the CSV.")
            return
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            st.write("No 'date' column found in CSV. Please check the structure of the data.")
            return
        
        df = df[df['date'] >= datetime.now() - timedelta(days=30)]
        
        if df.empty:
            st.write("No relevant news articles found in the past month.")
            return
        
        if 'title' not in df.columns:
            st.write("No 'title' column found in CSV. Please check the structure of the data.")
            return
        
        # Analyze sentiment of news titles
        df['sentiment'] = analyze_sentiment(df['title'])
        
        st.write("Sentiment Data for the past month:")
        st.dataframe(df[['title', 'date', 'sentiment']])
        
        # Sentiment Distribution Pie Chart
        fig_pie = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'], 
                                           values=[(df['sentiment'] > 0).sum(), 
                                                   (df['sentiment'] < 0).sum(), 
                                                   (df['sentiment'] == 0).sum()])])
        fig_pie.update_layout(title='Sentiment Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
