import pandas as pd

def predict_transaction(model, input_data):
    """
    Predict whether a transaction is fraudulent or not.

    Parameters:
    - model: The trained machine learning model.
    - input_data: A dictionary containing the transaction features.

    Returns:
    - A dictionary with the prediction result.
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    # Probability of being fraud
    probability = model.predict_proba(input_df)[0][1]

    # Return the result as a dictionary
    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
    }