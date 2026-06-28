import joblib
import pandas as pd


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_transaction(model, transaction_data: dict):
    df = pd.DataFrame([transaction_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }