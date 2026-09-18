import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_loader import load_data
from src.preprocess import create_preprocessing_pipeline

def train_model(data_path):
    df = load_data(data_path)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print("[INFO] Preprocessing data and splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("[INFO] Training Random Forest Classifier...")
    model_pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=" * 50)
    print("Model Evaluation Metrics:")
    print("-" * 50)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc:.3f}")
    print("=" * 50)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/churn_model.pkl")
    print("[INFO] Model saved successfully to models/churn_model.pkl")
