from collections import Counter
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
import json
from sklearn.preprocessing import MinMaxScaler


class ImbalanceDataHandler:
    def __init__(self, config_file, metric):
        """
        Initialize the ImbalanceDataHandler with a configuration file.

        config_file: Path to the JSON configuration file.
        metric: The evaluation metric to optimize ('f1', 'AP', 'ROC').
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
        self.metric = metric

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
            "SMOTE": SMOTE(random_state=smote_config.get("random_state", 1),
                           k_neighbors=smote_config.get("k_neighbors", 3),
                           sampling_strategy={1: max_count // 20}).fit_resample(X_train, y_train),  # OverSampling
            "RandomUnderSampler": RandomUnderSampler(random_state=random_under_sampler_config.get("random_state", 1),
                                                     sampling_strategy={0: min_count * 20}).fit_resample(
                X_train, y_train)  # UnderSampling
        }

        best_score = -np.inf

        # Load cross-validation configuration
        cv_config = self.config["cross_validation"]
        skf = StratifiedKFold(n_splits=cv_config["n_splits"], random_state=cv_config["random_state"],
                              shuffle=cv_config["shuffle"])

        metric_scoring = {'f1': 'f1', 'AP': 'average_precision', 'ROC': 'roc_auc'}

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
                scoring=metric_scoring[self.metric]  # Dynamic metric selection
            )

            grid_search.fit(X_res_scaled, y_res)

            if self.metric == "f1":
                y_pred = grid_search.best_estimator_.predict(X_test_scaled)
                score = f1_score(y_test, y_pred)
            else:
                y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
                if self.metric == "AP":
                    score = average_precision_score(y_test, y_pred_proba)
                elif self.metric == "ROC":
                    score = roc_auc_score(y_test, y_pred_proba)

            print(f"Method: {method_name}, {self.metric} score: {score:.3f}")
            if score > best_score:
                best_score = score
                self.best_method = method_name
                self.best_model = grid_search.best_estimator_
                self.best_X_train = X_res
                self.best_y_train = y_res

        print(f"\nBest Resampling Method: {self.best_method} with {self.metric}: {best_score:.3f}")
        print(Counter(self.best_y_train))
        print(50 * "-")

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Perform resampling and model fitting with the best method.
        """
        self.resample_and_evaluate(X_train, y_train, X_test, y_test)
        return self.best_X_train, self.best_y_train

