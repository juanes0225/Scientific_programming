#src\analysis\model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def train_and_evaluate_logistic_regression(data: pd.DataFrame, features: list, target: str):
   
    X = data[features]
    y = data[target]

    
# Split data into training and test (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar y entrenar el modelo
    model = LogisticRegression(max_iter=1000)              # Complexity O((f+1)csE), f = number of characteristics
    model.fit(X_train, y_train)

    
# Initialize and train the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    print("Logistic Regression Performance Metrics:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")

    return metrics