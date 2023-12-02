from flask import Flask, request,jsonify
from joblib import dump, load
import numpy as np
from utils import read_digits,preprocess_data

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
# 1. Get the dataset
X, y = read_digits()
X = preprocess_data(X)

def load_model():
    # Load SVM model
    svm_model = load('svm_M22AIE242_gamma:0.01_C:10.joblib')
    # Load Logistic Regression model
    lr_model = load('logit_M22AIE242_solver:lbfgs.joblib')
    # Load Decision Tree model
    tree_model = load('tree_M22AIE242_max_depth:15_min_samples_split:2_criterion:entropy.joblib')

    return svm_model, lr_model, tree_model

@app.route("/predict/<model_type>",methods=['POST'])
def predict(model_type):
    svm_model, lr_model, tree_model=load_model()

    if model_type == 'svm':
        prediction=svm_model.predict(X)
    elif model_type == 'lr':
        prediction=lr_model.predict(X)
    elif model_type == 'tree':
        prediction=tree_model.predict(X)
    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)