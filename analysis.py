import pandas as pd
import numpy as np


bank_dataset = pd.read_csv('inputs/bank.csv', sep=";")

print(bank_dataset.head())

# Map the data to 0,1
bank_dataset['y'] = bank_dataset['y'].map({"yes": 1, "no": 0})
print(bank_dataset.head())

bank_dataset.to_csv("inputs/cleaned_bank_dataset.py", sep="|")

important_features = ['job', 'age', 'education', 'default', 'loan', 'balance', 'housing']
important_features_dataset = bank_dataset[important_features]

# Get Dummies
important_features_dataset = pd.get_dummies(important_features_dataset, columns=important_features,drop_first=True)
print(f"important_features_dataset={important_features_dataset}")
# Split into train and test.
from sklearn.model_selection import train_test_split
X = important_features_dataset
Y = bank_dataset['y']

print(f"X = {X}\n Y = {Y}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now we build a simple model

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, Y_train)


y_pred_with_logistic_regression = logistic_model.predict(X_test)

acc = accuracy_score(Y_test,y_pred_with_logistic_regression)
conf_matrix = confusion_matrix(Y_test,y_pred_with_logistic_regression)
classification_report = classification_report(Y_test,y_pred_with_logistic_regression)

print(f"accuracy = {acc}\n conf_matrix = {conf_matrix}\n classification_report={classification_report}")