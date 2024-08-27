 # Fraud Detection with XGBoost and Dask

This project demonstrates how to use XGBoost and Dask to build a fraud detection model. The code is written in Python and uses the Dask library for distributed computing.

## Step 1: Import Libraries

The first step is to import the necessary libraries.

```python
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
```

## Step 2: Load the Data

The next step is to load the transaction data into a Dask DataFrame.

```python
df = dd.read_csv('/Users/srimedha/Desktop/project/transactions.csv')
```

## Step 3: Preprocess the Data

The data needs to be preprocessed before it can be used for training the model. This includes converting non-categorical columns to categorical, one-hot encoding categorical columns, dropping unnecessary columns, handling missing values, and treating outliers.

```python
# Convert non-categorical columns to categorical and ensure known categories
df['type'] = df['type'].astype('category').cat.as_known()

# One-hot encode the 'type' column
df = dd.get_dummies(df, columns=['type'], drop_first=True)

# Drop unnecessary columns (if any)
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Handling missing values (if any)
df = df.dropna()

# Outlier detection and treatment (assuming 'amount' is a numerical feature)
df['amount'] = df['amount'].clip(upper=df['amount'].quantile(0.99))

# Feature scaling (optional, depending on your algorithm)
# df['amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

# Handling imbalanced classes (assuming 'isFraud' is your target variable)
df_majority = df[df['isFraud'] == 0]
df_minority = df[df['isFraud'] == 1]
df_
