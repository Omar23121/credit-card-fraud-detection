# Credit Card Fraud Detection System

## Overview
This project builds a machine learning system to detect fraudulent credit card transactions using real-world data. It addresses the challenge of highly imbalanced datasets and demonstrates practical techniques for improving fraud detection performance.

## Key Features
- Data analysis and visualization (EDA)
- Handling imbalanced data
- Logistic Regression baseline model
- Random Forest model (final model)
- Threshold tuning for precision-recall balance
- Feature importance analysis
- Interactive Streamlit dashboard

## Results
- Precision: 0.81  
- Recall: 0.82  
- F1-score: 0.81  

The Random Forest model achieved a strong balance between detecting fraud and minimizing false positives.

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

## Project Structure
```
credit-card-fraud-detection/
│
├── load_data.py
├── eda.py
├── train_model.py
├── random_forest_model.py
├── feature_importance.py
├── fraud_dashboard.py
├── README.md
├── .gitignore
```

## How to Run

### 1. Install dependencies
```
pip install pandas numpy matplotlib scikit-learn streamlit
```

### 2. Download dataset
Download from:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Place `creditcard.csv` in the project folder.

### 3. Run dashboard
```
streamlit run fraud_dashboard.py
```

## Insights
- The dataset is highly imbalanced (~0.17% fraud)
- Fraud detection requires balancing precision and recall
- Threshold tuning significantly improves performance
- Random Forest captures complex fraud patterns effectively

## Future Improvements
- Deploy dashboard online  
- Add real-time prediction input  
- Try advanced models (XGBoost)  
