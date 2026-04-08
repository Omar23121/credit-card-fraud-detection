import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Probability of class 1 (fraud)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

    for threshold in thresholds:
        evaluate_threshold(y_test, y_probs, threshold)

if __name__ == "__main__":
    main()