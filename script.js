// Include the xgboost-predictor library
const xgboostPredictor = require('xgboost-predictor');

// Load the XGBoost model
const modelUrl = 'path/to/your/model.json';  // Replace with the path to your model file
const xgboostModel = xgboostPredictor.loadModel(modelUrl);

function predictFraud() {
    // Get values from the form
    const transactionType = document.getElementById('transactionType').value;
    const step = parseInt(document.getElementById('step').value);
    const amount = parseFloat(document.getElementById('amount').value);
    const oldbalanceOrg = parseFloat(document.getElementById('oldbalanceOrg').value);
    const newbalanceOrig = parseFloat(document.getElementById('newbalanceOrig').value);
    const oldbalanceDest = parseFloat(document.getElementById('oldbalanceDest').value);
    const newbalanceDest = parseFloat(document.getElementById('newbalanceDest').value);
    const isFlaggedFraud = parseInt(document.getElementById('isFlaggedFraud').value);

    // Prepare the input data for prediction
    const inputData = {
        step,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        isFlaggedFraud,
        [`type_${transactionType}`]: 1,  // Assuming one-hot encoding for the transaction type
    };

    // Make the prediction
    const prediction = xgboostModel.predict(inputData);

    // Display the prediction result
    const predictionResultElement = document.getElementById('predictionResult');
    predictionResultElement.innerHTML = `
        <p>Is Fraudulent: ${prediction.isFraud}</p>
        <p>Probability: ${prediction.probability.toFixed(4)}</p>
    `;
}
