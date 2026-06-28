from src.predict import load_model, predict_transaction
from src.data_preprocessing import load_data

def test_model_prediction():
    model = load_model("models/fraud_model.pkl")

    df = load_data("data/raw/creditcard.csv")

    sample = df.drop("Class", axis=1).iloc[0].to_dict()

    result = predict_transaction(model, sample)

    assert "prediction" in result
    assert "fraud_probability" in result

    assert result["prediction"] in [0, 1]

    assert 0 <= result["fraud_probability"] <= 1