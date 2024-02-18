from flask import Flask, redirect, url_for, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "hello"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
app.app_context().push()
class Users(db.Model):
    _id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

    def __init__(self, name):
        self.name = name

@app.route("/")
def ryan():
    return render_template("index2.html",r=7,names=["ryan","rohan","ruben"])


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["loginId"]
        session["user"] = user
        found_user = Users.query.filter_by(name=user).first()
        if not found_user:
            return redirect(url_for("ryan"))
            
        return redirect(url_for("user"))
    else:
        return render_template("login.html")

@app.route("/user", methods=["POST", "GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]
        if request.method == "POST":
            email = request.form["email"]
            session["email"] = email
        else:
            email = session.get("email")

        return render_template("user.html", email=email)
    else:
        return redirect(url_for("login"))
    
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


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("email", None)
    return redirect(url_for("login"))

@app.route("/user")
@app.teardown_request
def teardown_request(exception):
    db.session.close()

db.create_all()

if __name__ == "__main__":
    app.run(debug=True)


