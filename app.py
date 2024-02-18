from flask import Flask, render_template, send_from_directory
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import timedelta

app = Flask(__name__)

# Generate synthetic data and predict demand
np.random.seed(42)
date_range = pd.date_range('2022-01-01', '2022-02-01', freq='D')
biriyani_values = np.random.randint(10, 50, size=len(date_range))
fried_rice_values = np.random.randint(5, 30, size=len(date_range))

data_large = {
    'Date': date_range,
    'Biriyani': biriyani_values,
    'FriedRice': fried_rice_values
}

df_large = pd.DataFrame(data_large)
df_large.set_index('Date', inplace=True)

df_large.index.freq = 'D'

def predict_demand_sarima(item_data, item_name):
    model = auto_arima(item_data, seasonal=True, suppress_warnings=True)
    model_fit = model.fit(item_data)

    start_date = item_data.index[-1] + timedelta(days=1)
    end_date = start_date + timedelta(days=2)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    historical_data = item_data[-10:]
    predictions = model_fit.predict(n_periods=len(future_dates), exogenous=np.tile(historical_data, (9, 1)), dynamic=False)

    # Plot the figure
    plt.figure(figsize=(12, 6))
    plt.plot(item_data.index, item_data, label='Actual')
    plt.plot(future_dates, predictions, label='Predicted', color='red')
    plt.title(f'Demand Prediction for {item_name} (SARIMA) based on the Previous 10 Days')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()

    # Save the figure to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_str = base64.b64encode(img_stream.read()).decode('utf-8')

    return img_str

# Flask route to render the HTML page
@app.route('/a')
def index():
    # Predict demand for Biriyani and get the base64 encoded plot
    predicted_value_item = predict_demand_sarima(df_large['FriedRice'], 'Fried Rice')
    
    # Render the HTML page with the plot
    return render_template('a.html', img_str=predicted_value_item)

if __name__ == '_main_':
    app.run(debug=True)