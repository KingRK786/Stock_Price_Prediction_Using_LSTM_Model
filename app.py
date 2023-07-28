import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
import math
import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import io
import base64
import requests
from jinja2 import Environment

app = Flask(__name__, template_folder='template', static_url_path='/static', static_folder='static')

# Add configuration for Jinja2 template environment
app.jinja_env.globals['enumerate'] = enumerate

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/stock_names', methods=['GET'])
def stock_names():
    query = request.args.get('query')  # Get the query entered by the user
    api_key = 'HEEJPVXIN8QD1AWG'  # Replace with your Alpha Vantage API key
    api_url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={api_key}'

    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        stock_names = [result['1. symbol'] for result in data['bestMatches']]
    else:
        stock_names = []  # Empty list if API request fails

    return jsonify(stock_names=stock_names)

@app.route('/predict', methods=['POST'])
def predict():
    today = date.today()
    end = today.strftime("%Y-%m-%d")
    start = '2014-01-01'
    user_input = request.form.get('stock-input')  # Retrieve user input from the form

    df = yf.download(user_input, start, end)

    description = df.describe()

    stock = yf.Ticker(user_input)
    stock_info = stock.info

    fig1 = plt.figure(figsize=(11, 4.5))
    plt.plot(df.Close)
    plt.title('Closing Price vs Time Chart')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    fig1_base64 = plot_to_base64(fig1)

    ma100 = df.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(11, 4.5))
    plt.plot(ma100, 'r')
    plt.plot(df.Close, 'b')
    plt.title('Closing Price vs Time Chart with 100 Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    fig2_base64 = plot_to_base64(fig2)

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(11, 4.5))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    plt.title('Closing Price vs Time Chart with 100 & 200 Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    fig3_base64 = plot_to_base64(fig3)

    data_training = pd.DataFrame(df['Close'][1: int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    scaler.scale_

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    future_periods = [15, 30, 45]  # Number of days to forecast
    forecast_input = input_data[-100:]  # Take the last 100 data points as input
    x_forecast = []
    for i in range(future_periods[-1]):
        x_forecast.append(forecast_input[-100:])
        forecast = model.predict(np.array(x_forecast))
    forecast_input = np.concatenate((forecast_input, forecast[-1:]), axis=0)
    x_forecast.pop(0)
    forecasts = forecast_input[-sum(future_periods):] * scale_factor

    fig4 = plt.figure(figsize=(11, 4.5))
    plt.plot(y_test, 'b', label='Original Closing Price')
    plt.plot(y_predicted, 'r', label='Predicted Closing Price')
    plt.plot(np.arange(len(y_test), len(y_test) + len(forecasts)), forecasts, 'g', label='Forecasted Closing Price')
    plt.title('Prediction vs Original Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    fig4_base64 = plot_to_base64(fig4)

    mae = mean_absolute_error(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    accuracy = 100 - mae  # Example: using MAE as the accuracy metric

    return render_template('result.html', description=description.to_html(),
                           stock_info=stock_info, fig1=fig1_base64,
                           fig2=fig2_base64, fig3=fig3_base64,
                           fig4=fig4_base64, mae=mae, mse=mse,
                           accuracy=accuracy, forecasts=forecasts)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()
    return base64_encoded

if __name__ == '__main__':
    app.run(debug=False)
