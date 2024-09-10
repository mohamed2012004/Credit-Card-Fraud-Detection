from collections import Counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler

# Load data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Split data into features and target
x_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]

# Class distribution
counter=Counter(y_train)
print('Class distribution:', dict(counter))  # Print class distribution in the training set

# Define a function for MinMax Scaling
def fit_transform(data):
    """
    Fit and transform the data using MinMaxScaler.
"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler


# Feature scaling
x_train_scaled, scaler = fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)  # Scale the test set using the same scaler

# Initialize Logistic Regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)  # Added class_weight

# Train the Logistic Regression model
model.fit(x_train_scaled, y_train)

# Predict and evaluate on the test set
y_pred = model.predict(x_test_scaled)
y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]  # Get probabilities for the positive class

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Calculate Average Precision (AP) Score
ap_score = average_precision_score(y_test, y_pred_prob)
print(f"Average Precision Score: {ap_score:.4f}")
