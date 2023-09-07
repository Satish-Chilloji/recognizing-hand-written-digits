# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from utils import tune_hparams,train_dev_test_split
from sklearn import svm,metrics


if __name__=='__main__':
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X=data
    y=digits.target

    gamma_ranges = [0.001, 0.01, 0.1, 1, 100]
    C_ranges = [0.1, 1, 2, 5, 10]

    list_of_all_param_combination = list(itertools.product(gamma_ranges, C_ranges))


    test_size=[0.1,0.2,0.3]
    dev_size=[0.1,0.2,0.3]
    list_of_size=list(itertools.product(test_size, dev_size))
    for test_frac, dev_frac in list_of_size:
        X_train,Y_train, X_dev, Y_dev, X_test, Y_test=train_dev_test_split(X,y,test_frac,dev_frac)
        model,gamma,C,cur_metric,train_metric=tune_hparams(X_train,Y_train,X_dev,Y_dev,list_of_all_param_combination)
        predicted_test=model.predict(X_test)
        test_metric = metrics.accuracy_score(y_pred=predicted_test, y_true=Y_test)
        print(f'Train Size:{1-(test_frac+dev_frac)} Test Size:{test_frac} Dev Size:{dev_frac} Train Acc:{train_metric} Test Acc:{test_metric} Val Acc:{cur_metric}')
        print(f'Found best metric for the SVM model with gamma:{gamma} and C:{C}')
        print('\n')