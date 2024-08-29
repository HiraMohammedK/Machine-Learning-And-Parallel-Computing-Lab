import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}
for clf_name, clf in classifiers.items():
    pipeline = make_pipeline(StandardScaler(), clf)
    with warnings.catch_warnings():  # Suppress all warnings
        warnings.filterwarnings("ignore")
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[clf_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cross Validation Mean Accuracy': np.mean(scores),
        'Cross Validation Std': np.std(scores)
    }

print("Evaluation Metrics:")
for clf_name, metrics in results.items():
    print(f"\nClassifier: {clf_name}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print(f"  Cross Validation Mean Accuracy: {metrics['Cross 
    Validation Mean Accuracy']:.4f} ± {metrics['Cross Validation Std']:.4f}")

# Choosing top 4 classifiers based on mean accuracy
top_classifiers = sorted(results.items(), key=lambda x: x[1]['Cross 
Validation Mean Accuracy'], reverse=True)[:4]

# ensemble methods on top classifiers
print("\nApplying Ensemble Methods:")

# Bagging
print("\nBagging:")
bagging_clf = BaggingClassifier(base_estimator=None, n_estimators=10, random_state=42)
for clf_name, _ in top_classifiers:
    pipeline = make_pipeline(StandardScaler(), classifiers[clf_name])
    with warnings.catch_warnings():  # Suppress all warnings
        warnings.filterwarnings("ignore")
        bagging_pipeline = make_pipeline(StandardScaler(), bagging_clf)
        bagging_scores = cross_val_score(bagging_pipeline, X_train, 
        y_train, cv=5, scoring='accuracy')
    print(f"{clf_name}: Mean Bagging Accuracy: 
    {np.mean(bagging_scores):.4f} ± {np.std(bagging_scores):.4f}")

# Boosting with AdaBoost (using Decision Tree as base estimator)
print("\nBoosting:")
boosting_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
n_estimators=50, algorithm='SAMME', random_state=42)
for clf_name, _ in top_classifiers:
    pipeline = make_pipeline(StandardScaler(), classifiers[clf_name])
    with warnings.catch_warnings():  # Suppress all warnings
        warnings.filterwarnings("ignore")
        boosting_pipeline = make_pipeline(StandardScaler(), boosting_clf)
        boosting_scores = cross_val_score(boosting_pipeline, X_train, 
        y_train, cv=5, scoring='accuracy')
    print(f"{clf_name}: Mean Boosting Accuracy: {np.mean(boosting_scores):.4f} ± {np.std(boosting_scores):.4f}")

# Stacking
print("\nStacking:")
stacking_estimators = [(clf_name, make_pipeline(StandardScaler(), clf)) 
for clf_name, _ in top_classifiers]
stacking_clf = StackingClassifier(estimators=stacking_estimators, 
final_estimator=LogisticRegression(max_iter=1000, random_state=42))
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore")
    stacking_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Stacking Mean Accuracy: {np.mean(stacking_scores):.4f} ± {np.std(stacking_scores):.4f}")
