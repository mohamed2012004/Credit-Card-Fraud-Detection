import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler
from Optimal_Approaches_for_Imbalance_Data import ImbalanceDataHandler

# Load the data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('val.csv')



# Split data into features and target
x_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
x_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]

# Define a function for MinMax Scaling
def fit_transform(data):
    """
    Fit and transform the data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler

#
# # Initialize Logistic Regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Initialize ImbalanceDataHandler with the parameter grid and model
imb_handler = ImbalanceDataHandler("config.json","f1")

# Resample and get the best training data
best_X_train, best_y_train = imb_handler.fit(x_train, y_train, x_test, y_test)

# Save the resampled data
pd.DataFrame(best_X_train).to_csv('best_x_train_resampled.csv', index=False)
pd.DataFrame(best_y_train).to_csv('best_y_train_resampled.csv', index=False)


# Feature scaling
x_train_scaled, scaler = fit_transform(best_X_train)
x_test_scaled = scaler.transform(x_test)  # Scale the test set using the same scaler



# Train the Logistic Regression model
model.fit(x_train_scaled, best_y_train)



# Predict and evaluate on the test set
y_pred = model.predict(x_test_scaled)
y_pred_prob=model.predict_proba(x_test_scaled)[:,1]

# Evaluation metrics
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Calculate Average Precision (AP) Score
ap_score = average_precision_score(y_test, y_pred_prob)
print(f"Average Precision Score: {ap_score:.4f}")
print(f"F1 Score:{f1_score(y_test, y_pred):.3f}")
