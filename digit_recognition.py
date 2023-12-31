"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str, help = "Model choices = {svm, tree}", default = "svm",)
parser.add_argument("--test_sizes", type=float,help="test size", default=0.2)
parser.add_argument("--dev_sizes", type=float,help="dev size", default=0.2)
parser.add_argument("--max_run", type=int,help="test size", default=5)
args = parser.parse_args()


results = []
test_sizes =  [args.test_sizes]
dev_sizes  =  [args.dev_sizes]
model_ = args.model.split(",")
model_types= [i for i in model_ if i!= ","]
max_run = args.max_run

# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
classifier_param_dict = {}
# 2.1. SVM
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations

# 2.2 Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
min_samples_split = [2,5,10]
criterion = ["gini","entropy","log_loss"]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_params_tree['min_samples_split'] = min_samples_split
h_params_tree['criterion'] = criterion
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations


for cur_run_i in range(max_run): 
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)


            for model_type in model_types:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                y_dev, current_hparams, model_type)        
            
                # loading of model         
                best_model = load(best_model_path)

                test_acc = predict_and_eval(best_model, X_test, y_test)
                train_acc = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)
# import pdb
# pdb.set_trace()
print(pd.DataFrame(results).groupby('model_type').describe().T)
                