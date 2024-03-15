STOCK PRICE PREDICTION USING LSTM
Overview:
This repository contains a Flask web application that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The app allows users to enter a stock symbol, fetches historical stock data, trains an LSTM model, and presents both past closing prices and future predictions.

Features:
~ Stock Data Fetching: Utilizes yfinance to fetch historical stock data.
~ Data Preprocessing: Scales data using MinMaxScaler for LSTM model training.
~ LSTM Model: Implements an LSTM neural network to predict future stock prices based on historical closing prices.
~ Visualization: Provides visualizations of actual vs. predicted stock prices.
~ Custom Predictions: Allows users to request predictions for custom date ranges.
