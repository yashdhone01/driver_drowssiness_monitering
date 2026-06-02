import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

def train_and_evaluate():
    print("[*] Loading driver_safety_resilient.csv...")
    try:
        df = pd.read_csv("driver_safety_resilient.csv")
    except FileNotFoundError:
        print("[!] Error: 'driver_safety_resilient.csv' not found. Run 'generate_dataset.py' first.")
        return

    # Task 1: Mapping
    X = df[['EAR', 'mouth_ratio', 'head_tilt', 'eye_closure_time', 'no_movement_time']]
    y = df['label']

    # Task 2: Standard Train-Test Separation Sequence Mapping Test Bounds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Task 5: Model Regularization enforcing generalized decisions against native tracking loops
    print("[*] Initializing regularized random forest constraints (max_depth=5, n_estimators=75)...")
    clf = RandomForestClassifier(n_estimators=75, max_depth=5, random_state=42)

    print("[*] Executing 5-Fold Cross Validation targeting explicit dataset chunks natively...")
    # Task 3: Evaluating logic stability independently 
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    # Run fit to native sequences securely
    clf.fit(X_train, y_train)
    
    # Task 7: Persisting 
    joblib.dump(clf, "model.pkl")

    # Task 4 & 6: Comprehensive Metric Validation Overrides
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred, labels=['alert', 'drowsy', 'unconscious'])

    print("\n" + "="*50)
    print("       *** MODEL REGULARIZATION METRICS SUMMARY ***")
    print("="*50)
    print(" [1] Cross Validation Performance (5-Fold)")
    print(f"     > Avg Accuracy:  {cv_scores.mean()*100:.2f}%")
    print(f"     > Variance +/-:  {cv_scores.std()*100:.2f}%")
    print("\n [2] Generalization (20% Hold-Out Layer)")
    print(f"     > Test Accuracy: {test_acc*100:.2f}%")
    
    print("\n [3] Confusion Matrix (Rows: Actual, Cols: Predicted)")
    print("     [Alert / Drowsy / Unconscious]")
    for i, label in enumerate(['alert', 'drowsy', 'uncon']):
        print(f" {label[:5]:>5} {conf_mat[i]}")

    print("\n [4] Classification Structure")
    print(classification_report(y_test, y_pred))
    print("="*50)
    print("[+] Generalized, non-overfitting model tightly exported to 'model.pkl'.")

if __name__ == "__main__":
    train_and_evaluate()