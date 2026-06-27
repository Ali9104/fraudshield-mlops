import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    print("Fraud ratio in train:")
    print(y_train.value_counts(normalize=True) * 100)