import json
import warnings
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def main():
    # Load the original and resampled data
    data_test = pd.read_csv('val.csv')


    # Split data into features and target
    x_test = data_test.iloc[:, :-1]
    y_test = data_test.iloc[:, -1]

    # Load the resampled training data
    x_train_res = pd.read_csv('best_x_train_resampled.csv')
    y_train_res = pd.read_csv('best_y_train_resampled.csv')

    # Feature scaling function
    def fit_transform(data):
        """
        Fit and transform the data using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler

    # Feature scaling
    x_train_scaled, scaler = fit_transform(x_train_res)
    x_test_scaled = scaler.transform(x_test)

    # Load the configuration file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Extract parameter grid for logistic regression
    lr_params = config.get('logistic_regression', {})
    lr_model = LogisticRegression(**{k: v for k, v in lr_params.items() if k not in ['C', 'penalty']})


    # Extract parameters for RandomForestClassifier
    rf_params = config.get('random_forest', {})
    rf_model = RandomForestClassifier(**rf_params)

    # Extract parameters for VotingClassifier
    voting_params = config.get('voting_classifier', {})
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('lr', lr_model)],
        **voting_params
    )

    # Train the VotingClassifier
    voting_clf.fit(x_train_scaled, y_train_res)

    # Predict probabilities and labels on the original test set
    y_pred_prob = voting_clf.predict_proba(x_test_scaled)[:, 1]
    y_pred = voting_clf.predict(x_test_scaled)

    # Evaluation metrics
    print("Voting Classifier Performance:")
    print(f"Average Precision Score: {average_precision_score(y_test, y_pred_prob):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print(f"F1 Score:{f1_score(y_test, y_pred):.3f}")
if __name__ == '__main__':
    main()
