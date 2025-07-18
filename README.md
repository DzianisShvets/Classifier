Machine learning classifier that predicts the species of an iris flower based on measurements of its petals and sepals.
Used Logistic Regression to classify into 3 types: Setosa, Versicolor, and Virginica.

Step 1: Import libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
Load all necessary libraries for data handling, machine learning, and visualization.

Step 2: Load the Iris dataset

iris = load_iris()
X = iris.data
y = iris.target
Load feature data X and target labels y from the Iris dataset.

Step 3: Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Split the data into 80% training and 20% testing.

Step 4: Scale the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Normalize feature values to make them suitable for the model.

Step 5: Train the Logistic Regression model

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
Train the model on the training data to learn the patterns.

Step 6: Predict on test data

y_pred = model.predict(X_test_scaled)
Use the trained model to predict flower species on test data.

Step 7: Evaluate the model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
Print accuracy, detailed class-wise metrics, and a confusion matrix.

Step 8: Visualize the confusion matrix

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

Model Performance:

High accuracy on test data (typically >90%)
Confusion matrix shows very few misclassifications
Classification report gives strong precision, recall, and F1-scores for all classes

