import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    client = Client()  # Use default settings (local cluster)

    df = dd.read_csv('/Users/srimedha/Desktop/project/transactions.csv')
    df['type'] = df['type'].astype('category').cat.as_known()
    df = dd.get_dummies(df, columns=['type'], drop_first=True)
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    df = df.dropna()
    df['amount'] = df['amount'].clip(upper=df['amount'].quantile(0.99))
    df_majority = df[df['isFraud'] == 0]
    df_minority = df[df['isFraud'] == 1]
    df_majority_downsampled = df_majority.sample(frac=0.2, random_state=42)
    df_balanced = dd.concat([df_majority_downsampled, df_minority])
    df['totalBalanceOrig'] = df['oldbalanceOrg'] + df['newbalanceOrig']

    df_train, df_test = df_balanced.random_split([0.8, 0.2], random_state=42)

    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    target = 'isFraud'
    X_train = df_train[features + df_train.columns[df_train.columns.str.startswith('type_')].tolist()]
    y_train = df_train[target]
    X_test = df_test[features + df_test.columns[df_test.columns.str.startswith('type_')].tolist()]
    y_test = df_test[target]

    # Define the parameter grid
    param_grid = {'max_depth': [3, 5, 7, 9]}

    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train.compute(), y_train.compute())

    # Print the best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Extract the best model with the optimal max_depth
    best_model = grid_search.best_estimator_

    # Train the best model on the entire training set
    best_model.fit(X_train.compute(), y_train.compute())

    # Save the best model
    #best_model.save_model('/Users/srimedha/Desktop/project/best_model.xgb')

    # Make predictions on the test set
    y_pred = best_model.predict(X_test.compute())

    # Evaluate the performance of the best model
    accuracy = accuracy_score(y_test.compute(), y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    print(classification_report(y_test.compute(), y_pred))
