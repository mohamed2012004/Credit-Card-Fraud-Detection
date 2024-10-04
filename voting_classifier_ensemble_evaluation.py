import json
import warnings
from collections import Counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def load_data(train_path, test_path):
    """Load train and test datasets."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def preprocess_data(X_train, Y_train, X_test, smote_config):
    """Handle imbalanced data and scale features."""
    counter = Counter(Y_train)
    max_count = max(counter.values())

    # Handle imbalanced data with SMOTE
    smote = SMOTE(
        random_state=smote_config.get("random_state", 1),
        k_neighbors=smote_config.get("k_neighbors", 5),
        sampling_strategy={1: max_count // 50}
    )
    X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

    # Scale features
    scaler = MinMaxScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_resampled_scaled, Y_resampled, X_test_scaled


def initialize_models(config):
    """Initialize Logistic Regression and Random Forest models."""
    lr_params = config.get('logistic_regression', {})
    lr_model = LogisticRegression(**{k: v for k, v in lr_params.items() if k not in ['C', 'penalty']})

    rf_params = config.get('random_forest', {})
    rf_model = RandomForestClassifier(**rf_params)

    return lr_model, rf_model


def create_voting_classifier(lr_model, rf_model, voting_params):
    """Create a Voting Classifier."""
    return VotingClassifier(estimators=[('rf', rf_model), ('lr', lr_model)], **voting_params)


def perform_grid_search(X_train, y_train, voting_clf, param_grid, skf, scoring_metric):
    """Perform grid search to find the best model parameters."""
    grid_search = GridSearchCV(voting_clf, param_grid, cv=skf, scoring=scoring_metric)
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(grid_search, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = grid_search.best_estimator_.predict(X_test)

    print("Voting Classifier Performance:")
    print(f"Average Precision Score: {average_precision_score(y_test, y_pred_prob):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print(f"F1 Score (Test): {f1_score(y_test, y_pred):.3f}")
    print(grid_search.best_estimator_)


def main():
    # Suppress specific warnings
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Load the configuration file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Load the original and resampled data
    data_train, data_test = load_data('train.csv', 'val.csv')

    # Split data into features (X) and target (y)
    X_train, Y_train = data_train.iloc[:, :-1], data_train.iloc[:, -1]
    X_test, y_test = data_test.iloc[:, :-1], data_test.iloc[:, -1]

    # Preprocess data
    smote_config = config.get("smote", {})
    X_train_scaled, Y_train_resampled, X_test_scaled = preprocess_data(X_train, Y_train, X_test, smote_config)

    # Initialize models
    lr_model, rf_model = initialize_models(config)

    # Create Voting Classifier
    voting_clf = create_voting_classifier(lr_model, rf_model, config.get('voting_classifier', {}))

    # Cross-validation setup
    cv_config = config["cross_validation"]
    skf = StratifiedKFold(n_splits=cv_config["n_splits"], random_state=cv_config["random_state"],
                          shuffle=cv_config["shuffle"])

    # Extract scoring metric from config
    scoring_metric = "f1"
    param_grid = {
        'lr__C': config["logistic_regression"]["C"],
        'lr__penalty': config["logistic_regression"]["penalty"],
        'rf__n_estimators': config["random_forest"]["n_estimators"],
        'rf__max_depth': config["random_forest"]["max_depth"]
    }
    param_grid = {key: (value if isinstance(value, list) else [value]) for key, value in param_grid.items()}

    # Perform Grid Search
    grid_search = perform_grid_search(X_train_scaled, Y_train_resampled, voting_clf, param_grid, skf, scoring_metric)

    # Evaluate the model
    evaluate_model(grid_search, X_test_scaled, y_test)


if __name__ == '__main__':
    main()
