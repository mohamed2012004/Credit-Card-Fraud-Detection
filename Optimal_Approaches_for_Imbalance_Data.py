from collections import Counter
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
import json

from sklearn.preprocessing import MinMaxScaler


class ImbalanceDataHandler:
    def __init__(self, config_file):
        """
        Initialize the ImbalanceDataHandler with a configuration file.

        config_file: Path to the JSON configuration file.
        """
        # Load configuration
        with open(config_file, 'r') as file:
            self.config = json.load(file)

        # Initialize the model and parameter grid from config
        self.model = LogisticRegression(**self.config["logistic_regression"])
        self.param_grid = {'C': self.config["logistic_regression"]["C"],
                           'penalty': self.config["logistic_regression"]["penalty"]}

        self.best_method = None
        self.best_model = None
        self.best_X_train = None
        self.best_y_train = None
        self.scaler = MinMaxScaler()

    def resample_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Perform resampling, fit models, and evaluate their performance.
        """
        counter = Counter(y_train)
        max_count = max(counter.values())
        min_count = min(counter.values())

        # Load resampling configurations
        smote_config = self.config.get("smote", {})
        random_under_sampler_config = self.config.get("random_under_sampler", {})

        methods = {
            "Original": (X_train, y_train),  # No resampling
            "SMOTE": SMOTE(random_state=smote_config.get("random_state", 1),
                           k_neighbors=smote_config.get("k_neighbors", 3),
                           sampling_strategy={1: max_count}).fit_resample(X_train, y_train),  # OverSampling
            "RandomUnderSampler": RandomUnderSampler(random_state=random_under_sampler_config.get("random_state", 1),
                                                     sampling_strategy={0: min_count*100}).fit_resample(
                X_train, y_train)  # UnderSampling
        }

        best_roc_auc = -np.inf
        best_ap = -np.inf

        # Load cross-validation configuration
        cv_config = self.config["cross_validation"]
        skf = StratifiedKFold(n_splits=cv_config["n_splits"], random_state=cv_config["random_state"],
                              shuffle=cv_config["shuffle"])

        for method_name, (X_res, y_res) in methods.items():
            # Create pipeline with resampling and logistic regression model
            pipeline = ImbPipeline(steps=[('model', self.model)])
            X_res_scaled = self.scaler.fit_transform(X_res)
            X_test_scaled = self.scaler.transform(X_test)
            # Perform Grid Search
            grid_search = GridSearchCV(
                pipeline,
                {'model__' + key: value for key, value in self.param_grid.items()},
                cv=skf,
                scoring='average_precision'
            )

            grid_search.fit(X_res_scaled, y_res)

            y_pred_prob = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            ap = average_precision_score(y_test, y_pred_prob)

            print(f"Method: {method_name}, ROC AUC: {roc_auc:.3f}, AP: {ap:.3f}")

            if (roc_auc > best_roc_auc or
                    (roc_auc == best_roc_auc and ap > best_ap)):
                best_roc_auc = roc_auc
                best_ap = ap
                self.best_method = method_name
                self.best_model = grid_search.best_estimator_
                self.best_X_train = X_res
                self.best_y_train = y_res

        print(f"\nBest Resampling Method: {self.best_method} with ROC AUC: {best_roc_auc:.3f} and AP: {best_ap:.3f}")

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Perform resampling and model fitting with the best method.
        """
        self.resample_and_evaluate(X_train, y_train, X_test, y_test)
        return self.best_X_train, self.best_y_train



