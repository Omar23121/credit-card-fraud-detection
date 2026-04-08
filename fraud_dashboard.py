import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

@st.cache_resource
def train_model(df):
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

    return model, X, X_test, y_test, y_probs

def evaluate_threshold(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return cm, precision, recall, f1, report_dict

# Load data and model
df = load_data()
model, X, X_test, y_test, y_probs = train_model(df)

# Title
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    "This dashboard analyzes fraudulent vs normal transactions using a Random Forest model."
)

# Dataset overview
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
col3.metric("Fraud Rate", f"{df['Class'].mean() * 100:.2f}%")

# Charts side by side
st.subheader("📈 Data Visualizations")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📊 Class Distribution")
    class_counts = df["Class"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(3.5, 2))
    ax1.bar(class_counts.index.astype(str), class_counts.values)
    ax1.set_title("Fraud vs Non-Fraud", fontsize=10)
    ax1.set_xlabel("Class", fontsize=8)
    ax1.set_ylabel("Count", fontsize=8)
    ax1.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig1)

with chart_col2:
    st.markdown("### 💰 Transaction Amount Distribution")
    fig2, ax2 = plt.subplots(figsize=(4, 2.2))
    ax2.hist(df["Amount"], bins=50)
    ax2.set_title("Transaction Amount", fontsize=10)
    ax2.set_xlabel("Amount", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)
    ax2.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

# Feature importance
st.subheader("🔥 Feature Importance")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

top_features = feature_importance_df.head(10)

fig3, ax3 = plt.subplots(figsize=(4.5, 2.4))
ax3.bar(top_features["Feature"], top_features["Importance"])
ax3.set_title("Top 10 Features", fontsize=10)
ax3.set_xlabel("Feature", fontsize=8)
ax3.set_ylabel("Importance", fontsize=8)
ax3.tick_params(axis="x", labelrotation=45, labelsize=8)
ax3.tick_params(axis="y", labelsize=8)
plt.tight_layout()
st.pyplot(fig3)

# Threshold tuning
st.subheader("⚙️ Model Performance & Threshold Tuning")
threshold = st.slider("Choose threshold", 0.1, 0.99, 0.5, 0.01)

cm, precision, recall, f1, report_dict = evaluate_threshold(y_test, y_probs, threshold)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Precision", f"{precision:.4f}")
metric_col2.metric("Recall", f"{recall:.4f}")
metric_col3.metric("F1-score", f"{f1:.4f}")

st.markdown(
    "⚠️ The model balances fraud detection and false positives, achieving strong performance on highly imbalanced data."
)

# Confusion matrix
st.subheader("🧩 Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Fraud"],
    columns=["Predicted Normal", "Predicted Fraud"]
)
st.dataframe(cm_df, use_container_width=True)

# Classification report
st.subheader("📋 Classification Report")
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# Top features table
st.subheader("📌 Top 5 Important Features")
st.dataframe(feature_importance_df.head(5), use_container_width=True)