# Credit Card Fraud Detection

## Project Title
Credit Card Fraud Detection

## Project Description
This project aims to build and evaluate machine learning models to detect fraudulent credit card transactions. It involves data preprocessing, handling class imbalance, model training, and evaluation using various techniques and configurations.

## File Structure
- 📂 `data_train.csv`: Training dataset with features and target variable.
- 📂 `data_val.csv`: Validation dataset with features and target variable.
- ⚙️ `config.json`: Configuration file containing hyperparameters and resampling settings.
- 🛠️ `Optimal_Approaches_for_Imbalance_Data.py`: Defines `ImbalanceDataHandler` class for handling imbalanced data with flexibility to choose the evaluation metric (ROC AUC, AP, F-score).
- 🧹 `preprocess_and_train.py`: Script for preprocessing data, scaling features, and training the initial Logistic Regression model.
- 📊 `evaluate_models.py`: Script for evaluating models using a Voting Classifier.
- 📁 `best_x_train_resampled.csv`: Resampled training features.
- 📁 `best_y_train_resampled.csv`: Resampled training target variable.

## Features
- **Data Preprocessing**: Handles missing values and scales features using MinMaxScaler.
- **Imbalanced Data Handling**: Implements SMOTE and RandomUnderSampler for class balancing.
- **Model Training**: Utilizes Logistic Regression and Voting Classifier combining Logistic Regression and Random Forest.
- **Model Evaluation**: Provides comprehensive performance metrics including ROC AUC, average precision score, confusion matrix, and classification report.

## Project Workflow

### 1. Initial Logistic Regression on Imbalanced Data
- Applied Logistic Regression to the imbalanced dataset.
- Observed significant differences in the counts of class 0 and class 1, highlighting the imbalance issue.

### 2. Comparison of Resampling Techniques
- **Data Preparation**: Prepared the original dataset and applied different resampling techniques:
  - **Oversampling**: Increased the number of samples in the minority class.
  - **Undersampling**: Reduced the number of samples in the majority class.
- **Model Evaluation**: Used GridSearchCV to find the best hyperparameters for each technique.
- **Metrics**: Compared performance using ROC AUC, Average Precision (AP), and F-score metrics.

### 3. Saving Best Resampled Data
- Saved the data resulting from the best resampling method into two separate files:
  - `best_x_train_resampled.csv`
  - `best_y_train_resampled.csv`

### 4. Model Training on Resampled Data
- **Logistic Regression**: Trained Logistic Regression on the resampled data and obtained promising results.

### 5. Validation with Random Forest and Voting Classifier
- **Random Forest**: Added Random Forest to the analysis to improve performance and validate results.
- **Voting Classifier**: Combined Random Forest and Logistic Regression using a Voting Classifier.
  - **Importance**: Highlighted the significance of Random Forest and its combination with Logistic Regression.
  - **Impact**: The use of a Voting Classifier incorporating both Random Forest and Logistic Regression provides a powerful approach to credit card fraud detection. It combines the ensemble strength of Random Forest with the probabilistic clarity of Logistic Regression, resulting in improved accuracy, robustness, and practical effectiveness in identifying fraudulent transactions.

## Dashboard and Comparative Analysis
After completing the project, a comprehensive dashboard was created to visualize and compare the performance of different techniques. The analysis focused on several key metrics: False Negatives (FN), Average Precision (AP), ROC AUC, False Positives (FP), True Negatives (TN), and True Positives (TP). Here’s a summary of the findings:

### 1. True Negative Rate (TNR)
- **Highest Rate**: Imbalanced Data
  - **Lower Sensitivity (True Positive Rate)**: A high TNR might come at the expense of a lower True Positive Rate (TPR), meaning the model may miss a significant number of fraudulent transactions. This is critical in fraud detection where missing actual fraud cases can lead to financial losses.
  - **Increased False Negatives (FN)**: With a focus on high TNR, the model may become too conservative, resulting in a higher number of false negatives. In the context of credit card fraud, this means that some fraudulent transactions might go undetected.

### 2. False Positive Rate (FPR)
- **Highest Rate**: Voting Classifier
  - **Observation**: The Voting Classifier showed the highest False Positive Rate (FPR), meaning it incorrectly flagged more legitimate transactions as fraudulent.
  - **Significance**: While the Voting Classifier improves overall performance, its higher FPR suggests that it may lead to more false alarms. Balancing this with other metrics is crucial to minimize disruption for legitimate users.

### 3. Performance by Metric (F-score)
- **Voting Classifier After Each Metric**: Applied the Voting Classifier after each metric evaluation (ROC AUC, AP, and F-score).
- **Dashboard Based on F-score**: The final dashboard was based on the F-score metric, providing a balanced view of precision and recall.
- **Maximum FPR**: SMOTE showed the highest False Positive Rate.
- **Maximum TNR**: Imbalanced data produced the highest True Negative Rate.

## Importance of Findings
These findings are critical for improving the effectiveness of credit card fraud detection. The careful comparison of resampling techniques and evaluation metrics helps identify the optimal approach for reducing both false positives and false negatives, thus minimizing financial risk while maintaining a smooth user experience. The steps taken, including the use of the Voting Classifier and balancing the metrics, demonstrate the importance of selecting the right combination of models and metrics in fraud detection.
