from flask import Flask, render_template
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import timedelta

app = Flask(__name__)

# Function to fit SARIMA model and make predictions based on the previous 10 days
def predict_demand_sarima(item_data):
    model = auto_arima(item_data, seasonal=True, suppress_warnings=True)
    model_fit = model.fit(item_data)

    start_date = item_data.index[-1] + timedelta(days=1)
    end_date = start_date + timedelta(days=2)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    historical_data = item_data[-10:]
    predictions = model_fit.predict(n_periods=len(future_dates), exogenous=np.tile(historical_data, (9, 1)), dynamic=False)

    return pd.Series(predictions, index=future_dates)

@app.route('/')
def index():
    # Generate synthetic demand data for Biriyani and Fried Rice
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

    # Predict demand for Biriyani and Fried Rice
    predicted_value_biriyani = predict_demand_sarima(df_large['Biriyani'])
    predicted_value_fried_rice = predict_demand_sarima(df_large['FriedRice'])

    # Create a dataset with specific values for ingredients
    ingredients_data = {
        'Item': ['Biriyani', 'Fried Rice'],
        'Rice Type': ['Basmati', 'Normal'],
        'Rice (grams)': [200, 150],
        'Chicken (grams)': [250, 150],
        'Vegetables (grams)': [150, 100],
        'Oil (grams)': [40, 25],
        'Salt (grams)': [15, 10],
        'Water (milliliters)': [200, 150],
        'Onions (grams)': [50, 30],
        'Tomatoes (grams)': [50, 30],
        'Ginger-Garlic Paste (grams)': [20, 15],
        'Chilies (grams)': [15, 10],
        'Cumin Powder (grams)': [10, 5],
        'Coriander Powder (grams)': [10, 5],
        'Turmeric Powder (grams)': [5, 3],
        'Garam Masala (grams)': [8, 5],
        'Bay Leaves (pieces)': [2, 1],
        'Cinnamon Stick (pieces)': [2, 1],
        'Cardamom Pods (pieces)': [3, 2],
        'Cloves (pieces)': [4, 3],
        'Black Peppercorns (grams)': [5, 3],
        'Star Anise (pieces)': [1, 0.5],
        'Fennel Seeds (grams)': [3, 2],
        'Mustard Seeds (grams)': [2, 1],
        'Curry Leaves (leaves)': [10, 5],
    }

    df_ingredients = pd.DataFrame(ingredients_data)
    df_ingredients.set_index('Item', inplace=True)

    # Predicted demand values for Biriyani and Fried Rice
    predicted_demand_biriyani = predicted_value_biriyani.values[0]
    predicted_demand_fried_rice = predicted_value_fried_rice.values[0]

    # Calculate the ingredients required based on predicted demand
    ingredients_required = df_ingredients.iloc[:, 1:].multiply([predicted_demand_biriyani, predicted_demand_fried_rice], axis=0)

    # Create a DataFrame for ingredients required
    df_ingredients_required = pd.concat([df_ingredients[['Rice Type']], ingredients_required], axis=1)

    # Transpose the DataFrame for a tabular view
    df_ingredients_required_tabular = df_ingredients_required.transpose()

    # Exclude the 'Rice Type' row
    df_ingredients_required_tabular_without_rice_type = df_ingredients_required_tabular.drop('Rice Type', axis=0)

    # Calculate the row sums
    row_sums = df_ingredients_required_tabular_without_rice_type.sum(axis=1)
    
    # Convert the DataFrame to a dictionary for rendering in HTML
    ingredients_dict = df_ingredients_required_tabular_without_rice_type.to_dict()

    return render_template('ind.html', ingredients=ingredients_dict, predicted_biriyani=predicted_demand_biriyani, predicted_fried_rice=predicted_demand_fried_rice, row_sums=row_sums)

if __name__ == '__main__':
    app.run(debug=True)
