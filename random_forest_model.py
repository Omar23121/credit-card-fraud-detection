import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_threshold(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n{'=' * 50}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 50}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Fraud class summary:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


def main():
    df = pd.read_csv("creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        evaluate_threshold(y_test, y_probs, threshold)


if __name__ == "__main__":
    main()