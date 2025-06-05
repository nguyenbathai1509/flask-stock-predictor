
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    file = request.files['datafile']
    test_size = float(request.form['test_size'])
    random_state = int(request.form['random_state'])

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        df = pd.read_csv(file_path)
        df = df.drop(columns=["Date", "Dividends", "Stock Splits"], errors='ignore')

        x = df.drop("Close", axis=1).values
        y = df["Close"].values.reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LinearRegression()
        model.fit(x_train_scaled, y_train)

        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(scaler, open('scaler.pkl', 'wb'))

        train_score = model.score(x_train_scaled, y_train)
        test_score = model.score(x_test_scaled, y_test)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test_scaled)))

        return render_template('train_result.html', train_score=train_score, test_score=test_score, rmse=rmse)
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low = float(request.form['Low'])
        Volume = float(request.form['Volume'])

        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        input_data = np.array([[Open, High, Low, Volume]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        return render_template('predict.html', prediction=np.round(prediction[0][0], 2))
    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
