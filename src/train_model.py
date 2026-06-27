import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from evaluate_model import evaluate_model, print_metrics
from data_preprocessing import load_data, split_features_target, split_data


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

    joblib.dump(model, "models/fraud_model.pkl")
    print("Model saved to models/fraud_model.pkl")