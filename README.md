# Ice Cream Sales Prediction (Flask + Linear Regression)

A full-stack prototype web application to predict ice cream sales from weather features using a multivariate linear regression model.

## 🚀 Features
- Train and persist model in `model.pkl` (Scikit-learn LinearRegression)
- Backend API with Flask:
  - `/` home page
  - `/predict` form-based prediction
  - `/api/predict` JSON-based prediction (AJAX)
  - `/train` retrain model from CSV
  - `/data` sample preview of dataset
  - `/plot.png` chart image
- Dataset: `ice-cream.csv` includes `Temperature`, `Rainfall`, `IceCreamsSold`
- UI: responsive modern design (Bootstrap + custom CSS)
- Chart with scatter + regression line + rainfall color overlay

## ⚙️ Setup
1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   python app.py
   ```
4. Access in browser:
   - `http://127.0.0.1:5000`

## 🧪 Testing
- Make a prediction through UI
- Try the AJAX button (calls `/api/predict`)
- Retrain model with updated data via `/train`
- Verify sample dataset table is rendered
- Verify chart updates at `/plot.png`

## 🗄️ Data
Input file: `ice-cream.csv`
- Date
- DayOfWeek
- Month
- Temperature
- Rainfall
- IceCreamsSold

## 📦 Scoring and metrics
- R² score is computed and shown on home page.

## 🔄 Future improvements
- User CSV upload with automatic feature detection
- Add multiple regression features (day, holiday, humidity)
- Add authentication + user profiles
- Add unit tests with `pytest`
- Deploy to Heroku/Azure/GCP

## 📌 Notes
- The app performs a fresh model retrain on startup to avoid stale model schema mismatch.
- Ensure `ice-cream.csv` is present in project root.
