# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools
import pandas as pd
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.svm import SVC  # Import the SVC class
from utils import tune_hparams,train_dev_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm,metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


if __name__=='__main__':

    # Load the MNIST dataset
    mnist = datasets.fetch_openml("mnist_784", version=1)
    X, y = mnist.data, mnist.target
    X = X / 255.0  # Scale pixel values to [0, 1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for the production model (SVM) using GridSearchCV
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5)
    svm_grid_search.fit(X_train, y_train)
    production_model = svm_grid_search.best_estimator_

    # Hyperparameter tuning for the candidate model (Decision Tree) using GridSearchCV
    dt_param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5)
    dt_grid_search.fit(X_train, y_train)
    candidate_model = dt_grid_search.best_estimator_

    # Make predictions
    production_predictions = production_model.predict(X_test)
    candidate_predictions = candidate_model.predict(X_test)

    # Calculate accuracies
    production_accuracy = accuracy_score(y_test, production_predictions)
    candidate_accuracy = accuracy_score(y_test, candidate_predictions)

    # Calculate confusion matrices
    production_confusion_matrix = confusion_matrix(y_test, production_predictions, labels=np.unique(y_test))
    candidate_confusion_matrix = confusion_matrix(y_test, candidate_predictions, labels=np.unique(y_test))

    # Calculate the number of samples predicted correctly in production but not in candidate
    correct_in_production_not_in_candidate = np.sum((production_predictions == y_test) & (candidate_predictions != y_test))

    # Calculate macro-average F1 scores
    production_f1 = f1_score(y_test, production_predictions, average='macro')
    candidate_f1 = f1_score(y_test, candidate_predictions, average='macro')

    # Print the results
    print(f"Production Model Accuracy: {production_accuracy:.2f}")
    print(f"Candidate Model Accuracy: {candidate_accuracy:.2f}")
    print("Production Model Confusion Matrix:")
    print(production_confusion_matrix)
    print("Candidate Model Confusion Matrix:")
    print(candidate_confusion_matrix)
    print(f"Samples predicted correctly in production but not in candidate: {correct_in_production_not_in_candidate}")
    print(f"Production Model Macro-average F1 Score: {production_f1:.2f}")
    print(f"Candidate Model Macro-average F1 Score: {candidate_f1:.2f}")