# ChronoForest

## Sales Forecasting using ARIMA and Random Forest

ChronoForest is a sales forecasting project developed using Python. The project uses historical Superstore sales data to forecast sales by combining an ARIMA time-series model with a Random Forest regression model.

The idea is to first use ARIMA to capture the time-based pattern in sales. The errors made by ARIMA are then modeled using Random Forest with features such as month and day of the week. The predictions from both models are combined to obtain the final ChronoForest prediction.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Daily Sales Aggregation](#daily-sales-aggregation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [ARIMA Model](#arima-model)
- [Residual Calculation](#residual-calculation)
- [Random Forest Model](#random-forest-model)
- [Final ChronoForest Prediction](#final-chronoforest-prediction)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)


---

## About the Project

Sales forecasting can help businesses understand sales patterns and make better decisions related to planning and inventory.

In this project, transaction-level Superstore sales data is converted into a daily time series and used for forecasting.

ChronoForest uses two models:

1. ARIMA is used to model the main time-series pattern in the sales data.
2. Random Forest is used to model the residuals produced by ARIMA.

The predictions from both models are then combined to generate the final prediction.

The complete implementation is available in the Jupyter Notebook.

---

## Objectives

The main objectives of the project are:

- Analyze historical sales data.
- Clean and preprocess the dataset.
- Convert transaction-level data into daily sales data.
- Study sales trends over time.
- Analyze sales based on the day of the week.
- Build an ARIMA model for sales forecasting.
- Calculate the residuals from the ARIMA model.
- Use Random Forest to model the residuals.
- Combine the predictions from both models.
- Evaluate the final model using Mean Absolute Error (MAE).

---

## Dataset

The project uses the Superstore Sales dataset.

The original dataset contains:

- 51,290 records
- 21 columns
- Sales data from 2011 to 2014

Some of the important columns in the dataset are:

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `order_date` | Date when the order was placed |
| `ship_date` | Date when the order was shipped |
| `ship_mode` | Shipping method |
| `customer_name` | Name of the customer |
| `segment` | Customer segment |
| `country` | Country |
| `region` | Geographical region |
| `category` | Product category |
| `sub_category` | Product sub-category |
| `product_name` | Name of the product |
| `sales` | Sales amount |
| `quantity` | Quantity ordered |
| `discount` | Discount applied |
| `profit` | Profit generated |
| `shipping_cost` | Shipping cost |
| `order_priority` | Order priority |

The `order_date` and `sales` columns are mainly used to create the time series for forecasting.

---

## Project Workflow

The project follows the workflow below:

```text
Superstore Sales Dataset
          |
          v
   Data Preprocessing
          |
          v
   Daily Sales Aggregation
          |
          v
 Exploratory Data Analysis
          |
          v
       ARIMA Model
          |
          v
   ARIMA Prediction
          |
          v
  Residual Calculation
          |
          v
 Month + Day of Week Features
          |
          v
   Random Forest Model
          |
          v
 Random Forest Prediction
          |
          v
 Final ChronoForest Prediction
          |
          v
     Model Evaluation
```

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Duplicate records are removed.
2. The `order_date` column is converted into datetime format.
3. Rows with missing `order_date` or `sales` values are removed.
4. Remaining missing values are handled.
5. The dataset is sorted according to `order_date`.

After preprocessing, the data is used for further analysis and time-series forecasting.

---

## Daily Sales Aggregation

The original dataset contains individual order records.

For forecasting, the sales are grouped by date and the total sales for each day are calculated.

```python
daily_sales = df.groupby('order_date')['sales'].sum().reset_index()
```

This produces a daily sales dataset with 1,430 observations.

The daily sales series is then extracted:

```python
sales_series = daily_sales['sales']
```

This series is used as the input for the ARIMA model.

---

## Exploratory Data Analysis

The project performs exploratory analysis to understand the sales data before building the forecasting model.

### Daily Sales Trend

A line plot is used to visualize changes in sales over time.

This helps in understanding the overall sales pattern and variations across different dates.

### Day of the Week Analysis

The day of the week is extracted from the date:

```python
daily_sales['day_of_week'] = daily_sales.index.dayofweek
```

The average sales for each day of the week are then calculated.

This is used to check whether sales show any weekly pattern.

---

# ARIMA Model

ARIMA is used as the first part of the ChronoForest model.

ARIMA is a time-series model that uses historical values of a series to model its temporal behavior.

The model used in this project is:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(sales_series, order=(2, 1, 2))
model_fit = model.fit()
```

The ARIMA parameters used are:

- `p = 2` - autoregressive component
- `d = 1` - differencing
- `q = 2` - moving average component

The trained model is then used to generate predictions:

```python
arima_pred = model_fit.predict(
    start=0,
    end=len(sales_series)-1
)
```

The predictions are converted into a Pandas Series for further processing.

---

## ARIMA Evaluation

The ARIMA model is evaluated using Mean Absolute Error (MAE).

```python
from sklearn.metrics import mean_absolute_error

mae_arima = mean_absolute_error(
    sales_series,
    arima_pred
)

print("ARIMA MAE:", mae_arima)
```

The MAE obtained in the project is:

```text
ARIMA MAE: 4304.2145
```

---

# Residual Calculation

After obtaining the ARIMA predictions, the residuals are calculated.

A residual is the difference between the actual sales and the predicted sales.

```python
residuals = sales_series - arima_pred
```

In simple terms:

```text
Residual = Actual Sales - ARIMA Prediction
```

These residuals represent the part of the sales pattern that was not captured by the ARIMA model.

The residuals are then used as the target variable for the Random Forest model.

---

# Random Forest Model

Random Forest is used to model the residuals produced by ARIMA.

Two features are created from the date:

```python
daily_sales['month'] = daily_sales.index.month
daily_sales['day_of_week'] = daily_sales.index.dayofweek
```

The feature matrix is:

```python
X = daily_sales[['month', 'day_of_week']]
```

The target variable is:

```python
y = residuals
```

The Random Forest model is initialized with 100 trees:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The model is trained using:

```python
rf.fit(X, y)
```

After training, Random Forest predicts the residual component:

```python
rf_pred = rf.predict(X)
```

---

# Final ChronoForest Prediction

The final ChronoForest prediction is calculated by combining the ARIMA prediction and the Random Forest residual prediction.

```python
final_pred = arima_pred + rf_pred
```

Since sales cannot be negative, negative predictions are replaced with zero:

```python
final_pred = final_pred.clip(lower=0)
```

The final prediction can therefore be represented as:

```text
Final Prediction
      =
ARIMA Prediction
      +
Random Forest Residual Prediction
```

---

# Model Evaluation

The final ChronoForest model is evaluated using Mean Absolute Error.

```python
from sklearn.metrics import mean_absolute_error

mae_final = mean_absolute_error(
    sales_series,
    final_pred
)

print("Final Model MAE:", mae_final)
```

The final MAE obtained in the project is:

```text
Final Model MAE: 3227.1081
```

---

# Results

The results obtained from the project are:

| Model | MAE |
|---|---:|
| ARIMA | 4304.21 |
| ChronoForest | 3227.11 |

The ChronoForest model gives a lower MAE than the ARIMA model used in this project.

The reduction in MAE is approximately 25% compared with the ARIMA result.

The notebook also contains a plot comparing the actual sales values with the final ChronoForest predictions.

---

## Technologies Used

The project was developed using:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Jupyter Notebook
- Google Colab
- Microsoft Excel

---

# Libraries

The main Python libraries used in the project are:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ChronoForest.git
```

Move into the project directory:

```bash
cd ChronoForest
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl
```

---

# How to Run

## Using Google Colab

1. Open `ChronoForest.ipynb` in Google Colab.
2. Upload the Superstore Sales Excel dataset.
3. Run the notebook cells in order.
4. The notebook will perform:
   - Data preprocessing
   - Daily sales aggregation
   - Exploratory data analysis
   - ARIMA training
   - ARIMA prediction
   - Residual calculation
   - Random Forest training
   - Final prediction
   - Model evaluation

## Using Jupyter Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ChronoForest.ipynb
```

Run the notebook cells from beginning to end.

---

# Project Structure

```text
ChronoForest/
│
├── ChronoForest.ipynb
├── superstore_sales_final.xlsx
└── README.md
```

### `ChronoForest.ipynb`

Contains the complete implementation of the project, including:

- Data preprocessing
- Data analysis
- Visualization
- ARIMA model
- Residual calculation
- Random Forest model
- Final prediction
- Model evaluation

### `superstore_sales_final.xlsx`

Contains the Superstore sales data used for the project.

### `README.md`

Contains information about the project, setup instructions, methodology and results.

---

# Future Improvements

The current implementation can be improved in several ways.

### Model Improvements

- Tune the ARIMA parameters instead of using a fixed `(2,1,2)` configuration.
- Try SARIMA for handling seasonality.
- Tune the Random Forest hyperparameters.
- Add lag features such as previous-day and previous-week sales.
- Include additional features that may affect sales.

### Evaluation Improvements

- Use a proper time-based train/test split.
- Evaluate the model on unseen future data.
- Add RMSE and MAPE along with MAE.
- Compare the model with other forecasting methods.

### Other Models

The project can also be extended by comparing the results with:

- SARIMA
- XGBoost
- Gradient Boosting
- Prophet
- LSTM

---

# Conclusion

ChronoForest combines ARIMA and Random Forest to forecast sales using historical sales data.

ARIMA is used to capture the main time-series pattern, while Random Forest is used to model the residuals using month and day-of-week features.

In the current implementation:

```text
ARIMA MAE        : 4304.21
ChronoForest MAE : 3227.11
```

The ChronoForest model achieves a lower MAE than the ARIMA model used in the project.

The project demonstrates a simple hybrid approach to sales forecasting by combining a traditional time-series model with a machine learning regression model.

---




