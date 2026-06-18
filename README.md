# Solar Power Generation Prediction

A machine learning regression project for predicting solar power generation using weather and solar-position features.

This project analyzes historical solar generation data, performs preprocessing and feature engineering, trains multiple regression models, tunes hyperparameters, and compares model performance using standard regression metrics.

## Project Overview

Solar power output depends heavily on environmental conditions such as sunlight timing, temperature, wind, humidity, sky cover, visibility, and atmospheric pressure. This project uses these weather-related features to predict the amount of power generated.

The main goal is to build and compare machine learning models that can estimate `power-generated` as accurately as possible.

## Repository Contents

```text
Solar-Power-Generation-Predicition/
|
|-- Modelbuildingfi (1).ipynb
|-- solarpowergeneration.csv
`-- README.md
```

## Dataset

The dataset file is:

```text
solarpowergeneration.csv
```

It contains **2,920 rows** and **10 columns**.

### Target Variable

- `power-generated`: Solar power generated for the recorded condition.

### Feature Columns

| Column | Description |
|---|---|
| `distance-to-solar-noon` | Distance from the solar noon time point |
| `temperature` | Temperature measurement |
| `wind-direction` | Wind direction in degrees/categories |
| `wind-speed` | Wind speed measurement |
| `sky-cover` | Cloud or sky cover level |
| `visibility` | Visibility condition |
| `humidity` | Humidity percentage |
| `average-wind-speed-(period)` | Average wind speed over a period |
| `average-pressure-(period)` | Average pressure over a period |
| `power-generated` | Target output variable |

### Dataset Notes

- Original dataset size: 2,920 records
- One missing value was found in `average-wind-speed-(period)`
- After handling the missing value, the modeling dataset contains 2,919 records
- Target range: `0` to `36,580`
- Average generated power: approximately `6,980`

## Notebook Workflow

File: `Modelbuildingfi (1).ipynb`

The notebook includes the full machine learning workflow:

1. Importing required libraries
2. Loading the solar power generation dataset
3. Understanding dataset shape, data types, and summary statistics
4. Checking and handling missing values
5. Exploratory data analysis with visualizations
6. Outlier analysis using boxplots
7. Correlation analysis using a heatmap
8. Feature engineering for wind direction
9. Train-test splitting
10. Scaling and transformation of numerical features
11. Hyperparameter tuning
12. Model training and evaluation
13. Model comparison

## Feature Engineering and Preprocessing

The notebook applies several important preprocessing steps:

- Missing value handling for `average-wind-speed-(period)`
- Wind direction transformation using sine and cosine features
- Scaling numerical features with `StandardScaler`
- Transformation of skewed features such as `visibility` and `humidity` using `PowerTransformer`
- Encoding of categorical-style features such as `sky-cover`
- Train-test split using an 80/20 split

The final training data used approximately:

- Training samples: 2,335
- Testing samples: 584
- Final modeled features: 13

## Machine Learning Models Used

The following regression models were trained and compared:

- LightGBM Regressor
- CatBoost Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Bagging Regressor with Decision Tree base estimator

## Hyperparameter Tuning

The notebook uses hyperparameter search methods such as:

- `RandomizedSearchCV` for LightGBM
- `GridSearchCV` for CatBoost
- `GridSearchCV` for Random Forest
- `GridSearchCV` for Gradient Boosting

## Model Performance

| Model | MSE | RMSE | MAE | R2 Score |
|---|---:|---:|---:|---:|
| CatBoost | 7,586,650.72 | 2,754.39 | 1,479.92 | 0.9369 |
| LightGBM | 7,990,401.63 | 2,826.73 | 1,520.72 | 0.9335 |
| Gradient Boosting | 8,382,533.55 | 2,895.26 | 1,654.81 | 0.9302 |
| Random Forest | 8,438,618.02 | 2,904.93 | 1,479.60 | 0.9298 |
| Bagging Regressor | 8,575,737.23 | 2,928.44 | 1,500.72 | 0.9286 |

The best-performing model in this project was **CatBoost Regressor**, with an R2 score of approximately **0.9369** and RMSE of approximately **2,754.39**.

## Key Insights

- `distance-to-solar-noon` has a strong relationship with `power-generated`.
- Weather-related features such as humidity, visibility, wind speed, and pressure influence solar output.
- Transforming wind direction into sine and cosine features helps represent circular direction data better.
- Ensemble models perform strongly on this regression problem.
- CatBoost produced the best overall score among the tested models.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- LightGBM
- CatBoost
- Jupyter Notebook

## How to Run This Project

1. Clone the repository:

```bash
git clone https://github.com/roshaankandimalla/Solar-Power-Generation-Predicition.git
cd Solar-Power-Generation-Predicition
```

2. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost notebook
```

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open and run:

```text
Modelbuildingfi (1).ipynb
```

## Future Improvements

- Rename the repository from `Predicition` to `Prediction` for clarity
- Add a `requirements.txt` file
- Save the best model using `joblib` or `pickle`
- Build a Streamlit web app for solar power prediction
- Add SHAP-based model explainability
- Add cross-validation results for all final models
- Add feature importance plots to the README
- Deploy the best model as an API

## Author

**Roshaankandimalla**

GitHub: [roshaankandimalla](https://github.com/roshaankandimalla)

## Disclaimer

This project is intended for educational and portfolio purposes. Predictions may vary based on data quality, weather conditions, feature engineering, and model tuning.
