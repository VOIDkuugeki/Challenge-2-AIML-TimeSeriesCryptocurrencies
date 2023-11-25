from flask import Flask, render_template
import tensorflow as tf
import numpy as np
import plotly.express as px
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import math

tf.random.set_seed(10)
bnb_model = tf.keras.models.load_model('bnb.h5')
sol_model = tf.keras.models.load_model('sol.h5')

app = Flask(__name__)


#BNB
start = '2021-01-01'
end = '2023-11-25'
bnb = yf.download("BNB-USD", start, end)


bnb_data = bnb['Close'].fillna(method='ffill')
bnb_dataset = bnb_data.values.reshape(-1, 1)
bnb_training_data_len = math.ceil(len(bnb_dataset) * .8)

bnb_scaler = MinMaxScaler(feature_range=(0,1))

bnb_scaler = bnb_scaler.fit(bnb_dataset)

bnb_dataset = bnb_scaler.transform(bnb_dataset)

#now we generate
n_lookback = 120 #Input sequences
n_forecast = 60 #Prediction

bnb_lookback = bnb_dataset[-n_lookback:]

bnb_lookback = bnb_lookback.reshape(1, n_lookback, 1)
bnb_forecast = bnb_model.predict(bnb_lookback)
bnb_forecast = bnb_scaler.inverse_transform(bnb_forecast)

bnb_past = bnb[['Close']][-180:].reset_index()
bnb_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
bnb_past['Date'] = pd.to_datetime(bnb_past['Date'])
bnb_past['Forecast'] = np.nan

bnb_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
bnb_future['Date'] = pd.date_range(start=bnb_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
bnb_future['Forecast'] = bnb_forecast.flatten()
bnb_future['Actual'] = np.nan

bnb_results = pd.concat([bnb_past, bnb_future]).set_index('Date')

bnb_volume = bnb['Volume'][-1]
print(bnb_volume)

bnb_price_24h = bnb_results.Forecast[-n_forecast]
bnb_price_24h =  "{:.2f}".format(bnb_price_24h)
print(bnb_price_24h)

bnb_price_7d = bnb_results.Forecast[-n_forecast + 7]
bnb_price_7d = "{:.2f}".format(bnb_price_7d)


bnb_price_today = bnb['Close'][-1]
bnb_price_today =  "{:.2f}".format(bnb_price_today)

print(bnb_price_today)

percentage_difference = ((float(bnb_price_24h) - float(bnb_price_today)) / float(bnb_price_today)) * 100
percentage_difference = "{:.2f}".format(percentage_difference)

percentage_difference = float(percentage_difference)

bnb_percentage_difference_7d = ((float(bnb_price_7d) - float(bnb_price_today)) / float(bnb_price_today)) * 100
bnb_percentage_difference_7d = "{:.2f}".format(bnb_percentage_difference_7d)
bnb_percentage_difference_7d = float(bnb_percentage_difference_7d)


#SOL

sol = yf.download(tickers=['SOL'], period='15y')
sol_data = sol['Close'].fillna(method='ffill')
sol_dataset = sol_data.values.reshape(-1, 1)
sol_training_data_len = math.ceil(len(sol_dataset) * .8)
sol_scaler = MinMaxScaler(feature_range=(0,1))

sol_scaler = sol_scaler.fit(sol_dataset)

sol_dataset = sol_scaler.transform(sol_dataset)

n_lookback = 120 #Input sequences
n_forecast = 60 #Prediction

sol_lookback = sol_dataset[-n_lookback:]

sol_lookback = sol_lookback.reshape(1, n_lookback, 1)
sol_forecast = sol_model.predict(sol_lookback)
sol_forecast = sol_scaler.inverse_transform(sol_forecast)


sol_past = sol[['Close']][-180:].reset_index()
sol_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
sol_past['Date'] = pd.to_datetime(sol_past['Date'])
sol_past['Forecast'] = np.nan

sol_past.loc[sol_past.index[-1], 'Forecast'] = sol_past.loc[sol_past.index[-1], 'Actual']


sol_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
sol_future['Date'] = pd.date_range(start=sol_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
sol_future['Forecast'] = sol_forecast.flatten()
sol_future['Actual'] = np.nan


sol_results = pd.concat([sol_past, sol_future]).set_index('Date')
sol_volume = sol['Volume'][-1]

sol_price_24h = sol_results.Forecast[-n_forecast]
sol_price_24h =  "{:.2f}".format(sol_price_24h)
print(sol_price_24h)


sol_price_today = sol['Close'][-1]
sol_price_today =  "{:.2f}".format(sol_price_today)


sol_percentage_difference = ((float(sol_price_24h) - float(sol_price_today)) / float(sol_price_today)) * 100
sol_percentage_difference = "{:.2f}".format(sol_percentage_difference)

sol_percentage_difference = float(sol_percentage_difference)


sol_price_7d = sol_results.Forecast[-n_forecast + 7]
sol_price_7d = "{:.2f}".format(sol_price_7d)

sol_percentage_difference_7d = ((float(sol_price_7d) - float(sol_price_today)) / float(sol_price_today)) * 100
sol_percentage_difference_7d = "{:.2f}".format(sol_percentage_difference_7d)
sol_percentage_difference_7d = float(sol_percentage_difference_7d)



@app.route('/')
def index():
    fig_bnb = px.line(bnb_results, x=bnb_results.index, y=['Actual', 'Forecast'], title='BNB USD Forecasting in 2 months')
    fig_bnb.add_shape(
        go.layout.Shape(
            type="line",
            x0=bnb_results.index[-n_forecast], y0=bnb_results['Actual'].min(),
            x1=bnb_results.index[-n_forecast], y1=bnb_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    
    fig_sol = px.line(sol_results, x=sol_results.index, y=['Actual', 'Forecast'], title='Emeren Group Ltd Forecasting in 2 months')
    fig_sol.add_shape(
        go.layout.Shape(
            type="line",
            x0=sol_results.index[-n_forecast], y0=sol_results['Actual'].min(),
            x1=sol_results.index[-n_forecast], y1=sol_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    
    # Convert the Plotly figure to HTML
    div_bnb = fig_bnb.to_html(full_html=False)
    div_sol = fig_sol.to_html(full_html=False)

    # Render the HTML template with the meta graph in the appropriate section
    return render_template('/index.html', sol_percentage_difference=sol_percentage_difference, sol_volume=sol_volume,
    sol_price_24h=sol_price_24h,sol_price_today=sol_price_today,
    sol_price_7d=sol_price_7d, sol_percentage_difference_7d=sol_percentage_difference_7d,
    bnb_price_7d=bnb_price_7d, bnb_percentage_difference_7d=bnb_percentage_difference_7d,
    div_bnb=div_bnb, div_sol=div_sol,bnb_volume=bnb_volume, bnb_price_24h=bnb_price_24h, bnb_price_today = bnb_price_today, percentage_difference=percentage_difference)


@app.route('/bnb')
def meta():
    # Code to fetch data and render the Meta page
    sol_percentage_difference_7d= px.line(bnb, x=bnb.index, y='Close', title='BNB USD Data')
    div_bnb= sol_percentage_difference_7d.to_html(full_html=False)

    fig_bnb = px.line(bnb_results, x=bnb_results.index, y=['Actual', 'Forecast'], title='Meta Forecasting in 2 months')
    fig_bnb.add_shape(
        go.layout.Shape(
            type="line",
            x0=bnb_results.index[-n_forecast], y0=bnb_results['Actual'].min(),
            x1=bnb_results.index[-n_forecast], y1=bnb_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    bnb_bnb = fig_bnb.to_html(full_html=False)


    return render_template('bnb.html', 
        bnb=bnb,
        bnb_results=bnb_results,
        bnb_price_today=bnb_price_today,
        bnb_price_24h=bnb_price_24h,
        bnb_price_7d=bnb_price_7d,
        bnb_percentage_difference_7d=bnb_percentage_difference_7d,
        bnb_volume=bnb_volume,
        div_bnb=div_bnb,
        bnb_bnb=bnb_bnb, n_forecast=n_forecast
    )

@app.route('/sol')
def microsoft():
    fig_sol = px.line(sol, x=sol.index, y='Close', title='Emeren Group Ltd  Data')
    div_sol = fig_sol.to_html(full_html=False)

    fig_sol = px.line(sol_results, x=sol_results.index, y=['Actual', 'Forecast'], title='Emeren Group Ltd Forecasting in 2 months')
    fig_sol.add_shape(
        go.layout.Shape(
            type="line",
            x0=sol_results.index[-n_forecast], y0=sol_results['Actual'].min(),
            x1=sol_results.index[-n_forecast], y1=sol_results['Actual'].max(),
            line=dict(color="red", width=1, dash="dash")
        )
    )
    sol_sol = fig_sol.to_html(full_html=False)

    return render_template('sol.html',
        sol=sol,
        sol_results=sol_results,
        sol_price_today=sol_price_today,
        sol_price_24h=sol_price_24h,
        sol_price_7d=sol_price_7d,
        sol_percentage_difference_7d=sol_percentage_difference_7d,
        sol_volume=sol_volume,
        div_sol=div_sol,
        sol_sol=sol_sol, n_forecast=n_forecast
    )


if __name__ == '__main__':
    app.run(debug=True) 