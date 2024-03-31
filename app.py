from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load the XGBoost model
bst = xgb.Booster({'nthread': 4})
bst.load_model('/Users/srimedha/Desktop/project/best_model.xgb')  # Replace with the path to your model file

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Extract form data
            step = int(request.form['step'])
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])
            isFlaggedFraud = int(request.form['isFlaggedFraud'])

            # Create a DataFrame for prediction
            data = {
                'step': [step],
                'amount': [amount],
                'oldbalanceOrg': [oldbalanceOrg],
                'newbalanceOrig': [newbalanceOrig],
                'oldbalanceDest': [oldbalanceDest],
                'newbalanceDest': [newbalanceDest],
                'isFlaggedFraud': [isFlaggedFraud],
            }
            df = pd.DataFrame(data)

            # Make prediction
            prediction = int(bst.predict(xgb.DMatrix(df))[0] > 0.5)

        except Exception as e:
            error = str(e)

    return render_template('app.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
