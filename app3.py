from flask import Flask, render_template
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import timedelta

app = Flask(__name__)

# Function to generate ingredient data
def generate_ingredient_data():
    ingredient_data = {
        'Rice (grams)': [500, 800],
        'Chicken (grams)': [600, 400],
        'Vegetables (grams)': [300, 500],
        # ... (add other ingredients as needed)
    }
    return pd.DataFrame(ingredient_data, index=['Biriyani', 'Fried Rice'])

# Function to predict demand using SARIMA
def predict_demand_sarima(item_data, item_name):
    # Your existing prediction code here...

    # Example code for generating a placeholder plot
    plt.plot(item_data.index, item_data, label='Actual')
    plt.title(f'Demand Prediction for {item_name}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()

    # Save the figure to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_str = base64.b64encode(img_stream.read()).decode('utf-8')

    return img_str

# Flask route to render the new HTML page with predicted values
@app.route('/predicted_values')
def predicted_values():
    # Generate ingredient data
    df_ingredients = generate_ingredient_data()

    # Predict demand for Fried Rice and get the base64 encoded plot
    predicted_value_item = predict_demand_sarima(df_ingredients['Fried Rice'], 'Fried Rice')

    # Display tabular view for ingredients required excluding Rice Type
    df_ingredients_without_rice = df_ingredients.drop('Rice (grams)', axis=1)

    # Display tabular view for row sums
    row_sums = df_ingredients.sum(axis=1)

    return render_template('predicted_values.html', img_str=predicted_value_item, df_ingredients_without_rice=df_ingredients_without_rice.to_html(), row_sums=row_sums.to_string())

if __name__ == '__main__':
    app.run(debug=True)
