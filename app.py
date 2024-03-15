from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io
import base64
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num

from flask import Flask, render_template, request
app = Flask(__name__)

def fetch_stock_data(symbol):
    try:
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=5*365)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_dataset(dataset, time_step=1):
    X_data, y_data = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step)]
        X_data.append(a)
        y_data.append(dataset[i + time_step])
    return np.array(X_data), np.array(y_data)

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_plot_url(train_dates, y_train, train_predict, test_dates, y_test, test_predict, prediction_dates):
    plt.figure(figsize=(12, 8))
    train_dates = date2num(train_dates)
    test_dates = date2num(test_dates)
    prediction_dates = date2num(pd.to_datetime(prediction_dates))

    plt.plot(train_dates, y_train, label='Actual Train data', linewidth=2, color='blue')
    plt.plot(train_dates, train_predict, label='Predicted Train data', linewidth=2, alpha=0.7, color='orange')

    min_length = min(len(test_dates), len(test_predict))
    adjusted_test_dates = test_dates[:min_length]
    adjusted_test_predict = test_predict[:min_length]

    plt.plot(adjusted_test_dates, y_test[:min_length], label='Actual Test data', linewidth=2, color='green')
    plt.plot(adjusted_test_dates, adjusted_test_predict, label='Predicted Test data', linewidth=2, alpha=0.7, color='red')

    plt.title('Stock Price Prediction', fontsize=10)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Stock Price', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def predict_for_range(df, start_date, end_date, time_step=1):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Split data into training (70%), testing (15%), and validation (15%) sets
    train_size = int(len(scaled_data) * 0.7)
    test_size = int(len(scaled_data) * 0.15)
    train_data, test_data, val_data = scaled_data[0:train_size,:], scaled_data[train_size:train_size+test_size,:], scaled_data[train_size+test_size:,:]

    # Create datasets for training and testing
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_val, y_val = create_dataset(val_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Train the LSTM model
    model = create_lstm_model((time_step, 1))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, verbose=1)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss}')

    # Predict for the specified date range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_data = scaled_data[mask]

    # Prepare the data for prediction
    X_pred, _ = create_dataset(filtered_data.flatten(), time_step)
    X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], 1))

    # Make predictions
    predictions = model.predict(X_pred)

    # Inverse transform to get actual price predictions
    predictions = scaler.inverse_transform(predictions).flatten()

    # Fetch actual prices for the date range
    actual_prices = df.loc[mask, 'Close'].values

    # Adjust the lengths of actual prices and predictions to match
    min_length = min(len(predictions), len(actual_prices))
    predictions = predictions[:min_length]
    actual_prices = actual_prices[:min_length]

    # Prepare the results
    prediction_dates = pd.to_datetime(df.index[mask])[:min_length]
    result = pd.DataFrame({
        'Date': prediction_dates,
        'Actual Price': actual_prices,
        'Predicted Price': predictions
    })

    return result



@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    predictions_table = None
    stock_symbol = None
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol')
        df = fetch_stock_data(stock_symbol)
        if df is not None and not df.empty:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))


            time_step = 100
            test_size = 250

            train_data = scaled_data[:-test_size]
            test_data = scaled_data[-(test_size + time_step):]

            X_train, y_train = create_dataset(train_data.flatten(), time_step)
            X_test, y_test = create_dataset(test_data.flatten(), time_step)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            model = create_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=75, batch_size=64, verbose=1)

            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            train_predict = scaler.inverse_transform(train_predict).flatten()
            test_predict = scaler.inverse_transform(test_predict).flatten()

            train_dates = pd.to_datetime(df.index[:len(y_train)])
            test_dates = pd.to_datetime(df.index[len(y_train):(len(y_train) + len(y_test))])

            min_length = min(len(test_dates), len(test_predict))
            adjusted_test_dates = test_dates[:min_length]
            adjusted_test_predict = test_predict[:min_length]

            x_input = test_data[-time_step:].reshape((1, time_step, 1))
            future_predictions = []
            for _ in range(10):
                future_pred = model.predict(x_input)
                future_pred = np.squeeze(future_pred)
                future_predictions.append(future_pred)
                future_pred = future_pred.reshape(1, 1, 1)
                x_input = np.append(x_input[:, 1:, :], future_pred, axis=1)

            next_10_days = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            prediction_dates = pd.date_range(start=df.index[0], periods=10).strftime('%Y-%m-%d').tolist()
            today = datetime.date.today()
            future_dates = pd.date_range(start=today + datetime.timedelta(days=1), periods=10, freq='B')
            predictions_table = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Predicted Price': next_10_days
            })

            plot_url = create_plot_url(train_dates, y_train, train_predict, adjusted_test_dates, y_test[:min_length], adjusted_test_predict, prediction_dates)
            predictions_table = predictions_table.to_html(index=False, classes='table table-striped')

    return render_template('index.html', plot_url=plot_url, predictions_table=predictions_table, stock_symbol=stock_symbol)

@app.route('/custom_prediction', methods=['GET', 'POST'])
def custom_prediction():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        df = fetch_stock_data(stock_symbol)
        if df is None or df.empty:
            return render_template('error.html', error="Stock data not found for the given symbol.")

        prediction_results = predict_for_range(df, start_date, end_date)
        return render_template('custom_prediction_result.html',
                               predictions=prediction_results)



    return render_template('custom_prediction.html')



if __name__ == '__main__':
    app.run(debug=True)
