from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle
import os
from src.preprocess import load_data, preprocess_data, save_preprocessed_data, load_preprocessed_data
from sklearn.preprocessing import StandardScaler

# Check if preprocessed data exists
if os.path.exists('preprocessed/preprocessed_X.pkl') and os.path.exists('preprocessed/preprocessed_y.pkl') and os.path.exists('preprocessed/preprocessed_vectorizer.pkl'):
    print("Loading preprocessed data...")
    X, y, vectorizer = load_preprocessed_data()
else:
    print("Loading and preprocessing data...")
    data = load_data()
    X, y, vectorizer = preprocess_data(data)
    save_preprocessed_data(X, y, vectorizer)
    print("Data loaded and preprocessed successfully.")

# Scaling the data
scaler = StandardScaler(with_mean=False)  # with_mean=False to avoid issues with sparse matrices
X = scaler.fit_transform(X)

# Examine the distribution of classes
class_distribution = pd.Series(y).value_counts()
print("Class distribution before balancing:")
print(class_distribution)

# Remove rare classes
min_samples = 5  # Minimum number of samples required for each class
filtered_classes = class_distribution[class_distribution >= min_samples].index
X = X[y.isin(filtered_classes)]
y = y[y.isin(filtered_classes)]

# Handle class imbalance using SMOTE
print("Handling class imbalance using SMOTE...")
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after balancing:")
print(pd.Series(y_resampled).value_counts())

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Data split successfully.")

# Helper function to print classification report
def print_classification_report(report):
    print("accuracy: ", report['accuracy'])
    print("macro avg: ", report['macro avg'])
    print("weighted avg: ", report['weighted avg'])

# Logistic Regression
print("Training Logistic Regression...")
logreg = LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear', tol=1e-3)
param_grid_logreg = {'C': [0.01, 0.1, 1, 10, 100]}
grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5)
grid_logreg.fit(X_train, y_train)
y_pred_logreg = grid_logreg.predict(X_test)
report_logreg = classification_report(y_test, y_pred_logreg, output_dict=True, zero_division=0)
print_classification_report(report_logreg)
print("Best Score for Logistic Regression: ", grid_logreg.best_score_)

# Support Vector Machine
print("Training Support Vector Machine...")
svm = SVC(class_weight='balanced')
param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
y_pred_svm = grid_svm.predict(X_test)
report_svm = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)
print_classification_report(report_svm)
print("Best Score for Support Vector Machine: ", grid_svm.best_score_)

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(class_weight='balanced')
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
print_classification_report(report_rf)
print("Best Score for Random Forest: ", grid_rf.best_score_)

# Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier(class_weight='balanced')
param_grid_dt = {'max_depth': [5, 10, 20, 30]}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)
y_pred_dt = grid_dt.predict(X_test)
report_dt = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)
print_classification_report(report_dt)
print("Best Score for Decision Tree: ", grid_dt.best_score_)

# Naive Bayes
print("Training Naive Bayes...")
nb = MultinomialNB()
param_grid_nb = {'alpha': [0.01, 0.1, 1, 10]}
grid_nb = GridSearchCV(nb, param_grid_nb, cv=5)
grid_nb.fit(X_train, y_train)
y_pred_nb = grid_nb.predict(X_test)
report_nb = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
print_classification_report(report_nb)
print("Best Score for Naive Bayes: ", grid_nb.best_score_)

# K-Nearest Neighbors
print("Training K-Nearest Neighbors...")
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
y_pred_knn = grid_knn.predict(X_test)
report_knn = classification_report(y_test, y_pred_knn, output_dict=True, zero_division=0)
print_classification_report(report_knn)
print("Best Score for K-Nearest Neighbors: ", grid_knn.best_score_)

# Saving the Best Model and Vectorizer
best_model = None
best_score = 0
for grid in [grid_logreg, grid_svm, grid_rf, grid_dt, grid_nb, grid_knn]:
    if grid.best_score_ > best_score:
        best_model = grid.best_estimator_
        best_score = grid.best_score_

print("Saving the best model and vectorizer...")
pickle.dump(best_model, open('model/best_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
print("Model and vectorizer saved successfully.")

print("Training completed and model saved successfully.")
