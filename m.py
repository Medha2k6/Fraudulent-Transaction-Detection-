import xgboost as xgb

# Load your existing XGBoost model
bst = xgb.Booster({'nthread': 4})
bst.load_model('/Users/srimedha/Desktop/project/best_model.xgb')  # Replace with the actual path

# Save the model in JSON format
bst.save_model('/Users/srimedha/Desktop/project/model.json')
