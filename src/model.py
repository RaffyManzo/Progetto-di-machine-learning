import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(input_file, output_file):
    """Preprocess the data by handling missing values and encoding categories."""
    # Load dataset
    data = pd.read_csv(input_file)

    # Drop 'City' column
    data_cleaned = data.drop(columns=["City"], errors="ignore")

    # Handle missing values
    critical_columns = ["JoiningYear", "PaymentTier", "LeaveOrNot", "ExperienceInCurrentDomain"]
    data_cleaned = data_cleaned.dropna(subset=critical_columns)
    data_cleaned['Gender'] = data_cleaned['Gender'].fillna(data_cleaned['Gender'].mode()[0])
    data_cleaned['Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].mean())
    # Creazione della nuova categoria "Non Definito" per i valori mancanti nella colonna EverBenched
    data_cleaned['EverBenched'] = data_cleaned['EverBenched'].fillna('NotDefined')

    # One-hot encode categorical variables
    encoded_data = pd.get_dummies(data_cleaned, drop_first=False)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    encoded_data.to_csv(output_file, index=False)

    return encoded_data

def split_data(encoded_data):
    """Split the data into training and testing sets."""
    X = encoded_data.drop('LeaveOrNot', axis=1)
    y = encoded_data['LeaveOrNot']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, balance_data=False):
    """Train and evaluate Logistic Regression and Random Forest models."""
    if balance_data:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardizzare i dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(random_state=42, max_iter=5000, solver="saga", class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)
    logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])

    print("\nLogistic Regression Metrics:")
    print(classification_report(y_test, y_pred_logreg))
    print("ROC-AUC:", logreg_auc)

    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\nRandom Forest Metrics:")
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", rf_auc)

    # Confusion Matrices
    print("\nConfusion Matrix for Logistic Regression:")
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=[0, 1])
    disp_logreg.plot()
    plt.title("Confusion Matrix: Logistic Regression")
    plt.savefig("confusion_matrix_logreg.png")
    plt.close()

    print("\nConfusion Matrix for Random Forest:")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
    disp_rf.plot()
    plt.title("Confusion Matrix: Random Forest")
    plt.savefig("confusion_matrix_rf.png")
    plt.close()

    # ROC Curves
    print("\nROC Curve for Logistic Regression:")
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(logreg, X_test_scaled, y_test, ax=ax, name="Logistic Regression")
    plt.title("ROC Curve: Logistic Regression")
    plt.savefig("roc_curve_logreg.png")
    plt.close()

    print("\nROC Curve for Random Forest:")
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    plt.title("ROC Curve: Random Forest")
    plt.savefig("roc_curve_rf.png")
    plt.close()

def main():
    # Define file paths
    input_file = "data/raw/Employee_with_missing.csv"
    processed_file = "data/processed/Employee_processed.csv"

    # Preprocess data
    print("Preprocessing data...")
    encoded_data = preprocess_data(input_file, processed_file)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(encoded_data)

    # Evaluate without balancing
    print("Evaluating models without balancing...")
    train_and_evaluate_models(X_train, X_test, y_train, y_test, balance_data=False)

    # Evaluate with balancing
    print("Evaluating models with balancing...")
    train_and_evaluate_models(X_train, X_test, y_train, y_test, balance_data=True)

if __name__ == "__main__":
    main()
