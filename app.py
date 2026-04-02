from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
MODEL_FILE = 'model.pkl'
DATA_FILE = 'ice-cream.csv'

model = None
current_r2 = None

def train_model():
    global model, current_r2
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    target_col = 'IceCreamsSold'
    data = data.dropna(subset=['Temperature', 'Rainfall', target_col])

    # multi-factor model: temperature + rainfall
    X = data[['Temperature', 'Rainfall']]
    y = data[target_col]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    current_r2 = r2_score(y, predictions)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return data


def load_model():
    global model, current_r2
    # Always retrain on startup to ensure feature set is consistent
    train_model()


# Ensure model is ready on startup (safest for older Flask versions)
load_model()


@app.route('/')
def home():
    return render_template('index.html', r2_score=current_r2)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        train_model()

    temp_str = request.form.get('temperature', '').strip()
    rain_str = request.form.get('rainfall', '').strip()
    if temp_str == '' or rain_str == '':
        return render_template('index.html', error='Please provide both temperature and rainfall values.', r2_score=current_r2)

    try:
        temp = float(temp_str)
        rainfall = float(rain_str)
    except ValueError:
        return render_template('index.html', error='Temperature and rainfall must be numbers.', r2_score=current_r2)

    prediction = model.predict(np.array([[temp, rainfall]]))
    predicted_sales = float(prediction[0])

    return render_template(
        'index.html',
        prediction_text=f"Predicted Sales: {predicted_sales:.2f}",
        input_temperature=temp,
        input_rainfall=rainfall,
        r2_score=current_r2,
    )


@app.route('/train', methods=['POST'])
def train():
    try:
        data = train_model()
        return render_template('index.html', message='Model trained successfully.', r2_score=current_r2)
    except Exception as e:
        return render_template('index.html', error=f"Training failed: {str(e)}", r2_score=current_r2)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return {'error': 'JSON required'}, 400

    temp = data.get('temperature')
    rainfall = data.get('rainfall')
    if temp is None or rainfall is None:
        return {'error': 'temperature and rainfall are required'}, 400

    try:
        temp = float(temp)
        rainfall = float(rainfall)
    except ValueError:
        return {'error': 'numeric values required'}, 400

    if model is None:
        train_model()

    pred = float(model.predict(np.array([[temp, rainfall]]))[0])
    return {
        'temperature': temp,
        'rainfall': rainfall,
        'predicted_sales': round(pred, 2),
        'r2_score': round(current_r2, 4) if current_r2 is not None else None,
    }


@app.route('/data')
def data_preview():
    if not os.path.exists(DATA_FILE):
        return {'error': 'Dataset not found'}, 404

    df = pd.read_csv(DATA_FILE)
    preview = df.head(10).to_dict(orient='records')
    return {'samples': preview, 'rows': len(df)}


@app.route('/plot.png')
def plot_png():
    if not os.path.exists(DATA_FILE):
        return redirect(url_for('home'))

    data = pd.read_csv(DATA_FILE).dropna(subset=['Temperature', 'Rainfall', 'IceCreamsSold'])
    x1 = data['Temperature']
    x2 = data['Rainfall']
    y = data['IceCreamsSold']

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x1, y, c=x2, cmap='coolwarm', label='Actual Sales', alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Rainfall')

    if model is not None:
        line_x = np.linspace(x1.min(), x1.max(), 100)
        rainy = data['Rainfall'].median()
        line_X = np.column_stack((line_x, np.full_like(line_x, rainy)))
        line_y = model.predict(line_X)
        ax.plot(line_x, line_y, color='red', linewidth=2, label=f'Predicted (rainfall={rainy:.2f})')

    ax.set_xlabel('Temperature')
    ax.set_ylabel('IceCreamsSold')
    ax.set_title('Ice Cream Sales vs Temperature (color=Rainfall)')
    ax.legend()
    ax.grid(True)

    img = io.BytesIO()
    plt.tight_layout()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)