# SkyFare Insights — Precision Flight Price Predictor

![Flight_Wallpaper](flight_wallpaper.jpg)

**A production-ready, end-to-end machine learning project to predict flight ticket prices using feature engineering, advanced EDA, and ensemble models (Random Forest, Gradient Boosting, XGBoost).**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Preprocessing & Notes](#preprocessing--notes)
- [Modeling & Evaluation](#modeling--evaluation)
- [Usage Examples](#usage-examples)
- [Visualizations](#visualizations)
- [Insights](#insights)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
`SkyFare Insights` predicts flight ticket prices using journey metadata (airline, source, destination), temporal features (day, month, weekday, departure hour), and engineered variables (duration in minutes, number of stops). The pipeline emphasizes reproducibility, feature clarity, and robust model evaluation.

**Modeling highlights**
- Baseline: Linear Regression
- Ensembles: Random Forest, Gradient Boosting
- Best Model: XGBoost (tuned with advanced hyperparameter search)
- Pipeline-based preprocessing for consistent training & inference

---

## Features
- End-to-end pipeline: load → clean → feature engineer → train → tune → evaluate → predict
- Advanced exploratory data analysis with a dark-themed visualization style
- Custom feature engineering: `Duration_mins`, `Total_Stops`, `Travel_Day`, `Travel_Month`, `Travel_Weekday`
- Target transformation handling (`np.log1p` with inverse `np.expm1`)
- Individual model pipelines and randomized hyperparameter tuning
- Exportable predictions and saved model artifacts for deployment


---

## Preprocessing & Notes
- **Datetime parsing**: `Date_of_Journey` → `Travel_Day`, `Travel_Month`, `Travel_Weekday`.
- **Duration parsing**: convert strings like `"2h 50m"` → minutes (`Duration_mins`).
- **Stops**: `non-stop` → `0`, else integer count in `Total_Stops`.
- **Dropped columns**: `Route`, `Additional_Info`, `Arrival_Time` (optional) — drop anything that leaks future information or holds inconsistent values.
- **Null handling**: This project expects cleaned inputs for the preprocessor. If you prefer in-pipeline imputation, the ColumnTransformer can be extended to include imputers; otherwise fill or drop missing rows before transforming.
- **Scaling & Encoding**: numeric features use `RobustScaler`; categorical features are one‑hot encoded with `handle_unknown='ignore'` to avoid inference errors on unseen categories.
- **Target handling**: Model training on `np.log1p(Price)` is supported; convert predictions back with `np.expm1()` before reporting or submission.

---

## Modeling & Evaluation
- Use pipeline objects (`ColumnTransformer` + estimator) to ensure preprocessing is coupled with the model and prevents leakage.
- Evaluation metrics: **RMSE**, **MAE**, **R²**, and residual diagnostics.
- Hyperparameter tuning: `RandomizedSearchCV` over expanded XGBoost parameter space (`n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`).
- Each model is trained and tuned as an **individual pipeline** to allow independent tuning strategies and fair comparisons.

---

## Usage Examples
**Load a saved pipeline and predict (handles log-target):**
```python
import pandas as pd
import joblib
import numpy as np

pipe = joblib.load('outputs/model_XGBoost.joblib')  # pipeline: preprocess + estimator

test_df = pd.read_csv('data/Test_Data.csv')
log_preds = pipe.predict(test_df)
preds = np.expm1(log_preds)  # convert back

submission = pd.DataFrame({'ID': test_df['ID'], 'Price': preds})
submission.to_csv('outputs/flight_price_predictions.csv', index=False)
```

**Actual vs Predicted plot snippet:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([min_val, max_val],[min_val,max_val],'--',color='lime')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')
plt.show()
```

---

## Visualizations
- All notebook plots use a black/dark theme and consistent color palette for presentation quality.
- Visuals included in the notebook: distribution plots, ordered boxplots, monthly/weekday/hourly trends, correlation heatmaps, pairplots, Actual vs Predicted, and residual diagnostics.
- Example (Price Distribution Plot):
![Price_Distribution_Plot](Price-Distribution-Plot.jpg)

---

## Insights
This section highlights actionable insights discovered during the analysis and modeling process. You can copy these into a project report or present them as bullets in your portfolio.

### Key patterns
- **Temporal seasonality:** Average fares show clear monthly and weekday patterns. Weekend and holiday-adjacent dates often exhibit higher fares due to demand spikes. Business routes show weekday peaks during morning departures.
- **Departure hour effect:** Flights departing during early morning or late night tend to have different price profiles — business-hour departures (early morning, late afternoon) typically command premium pricing.
- **Stops & duration relationship:** Non-stop flights generally command higher prices on popular routes. When duration increases (longer total minutes) but stops increase too, the price effect becomes non-linear — multi-stop long itineraries can be cheaper than shorter non-stop premium flights.
- **Airline & route dominance:** Certain carriers consistently price higher across the same routes, indicating brand or service premium. High-competition routes (many carriers) show narrower price distributions.

### Feature importance (model-driven)
- `Duration_mins`, `Total_Stops`, and `Airline` frequently rank as top predictors in tree-based models (Random Forest / XGBoost). Temporal features (`Travel_Month`, `Weekday`, `Dep_Time`) provide complementary signal.

### Business recommendations
- **Dynamic pricing guardrails:** Use model insights to detect anomalous fare spikes and inform revenue management teams for manual review.
- **Customer fare guidance:** Surface predicted fare ranges to users (e.g., +/- 10–15%) rather than point estimates to account for volatility.
- **Route optimization:** Identify routes where multi-stop offerings could be marketed as cost-saving alternatives during peak months.

### Limitations & next steps
- **External factors missing:** The model does not include external demand signals (holidays calendar, competitor promotions, fuel prices), which could improve accuracy.
- **Data freshness:** Price dynamics change rapidly; retraining frequency (weekly/bi-weekly) is recommended for production use.
- **Granularity of airports:** Airport-level attributes (hub vs regional) and fare class information, if available, would enhance predictive power.

### Quick wins to improve model
1. Add holiday calendars and major event indicators (city‑level) to capture demand surges.
2. Incorporate historical price trajectories per route (time-series features) to capture momentum.
3. Try LightGBM/CatBoost and experiment with stacking ensembles for marginal gains.
4. Implement target encoding for high‑cardinality categorical variables (route, airline+route combos) with cross‑validation to minimize leakage.

---

## License
This project is released under the **MIT License** — see `LICENSE` file for details.

---
