# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from utils import tune_hparams,train_dev_test_split
from sklearn import svm,metrics
from skimage.transform import resize

if __name__=='__main__':
    # Load the digits dataset from scikit-learn
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X=digits.images
    y=digits.target

    gamma_ranges = [0.001, 0.01, 0.1, 1, 100]
    C_ranges = [0.1, 1, 2, 5, 10]

    list_of_all_param_combination = list(itertools.product(gamma_ranges, C_ranges))

    test_size=[0.2]
    dev_size=[0.1]
    # Define a list of image sizes to evaluate
    image_sizes = [(4, 4), (6, 6), (8, 8)]
    list_of_size=list(itertools.product(test_size, dev_size))


# Define a function to resize images and evaluate performance
def image_resize(image_size,X_train,X_dev,X_test):
    # Resize the training and testing images
    X_train_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_train])
    X_test_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_test])
    X_dev_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_dev])
    # Flatten the resized images
    X_train_flat = X_train_resized.reshape(len(X_train_resized), -1)
    X_test_flat = X_test_resized.reshape(len(X_test_resized), -1)
    X_dev_flat = X_dev_resized.reshape(len(X_test_resized), -1)
    return X_train_resized,X_test_resized,X_dev_resized

for size in list_of_size:
    X_train,Y_train, X_dev, Y_dev, X_test, Y_test=train_dev_test_split(X,y,0.2,0.1)
    print(X_train.shape)
    X_train_flat,X_dev_flat,X_test_flat=image_resize(size,X_train,X_dev,X_test)
    print(X_train_flat.shape)
    model,gamma,C,cur_metric,train_metric=tune_hparams(X_train_flat,Y_train,X_dev_flat,Y_dev,list_of_all_param_combination)
    predicted_test=model.predict(X_test_flat)
    test_metric = metrics.accuracy_score(y_pred=predicted_test, y_true=Y_test)
    print(f'Image Size:{size} Train Size:{1-(0.2+0.1)} Test Size:{0.2} Dev Size:{0.1} Train Acc:{train_metric} Test Acc:{test_metric} Val Acc:{cur_metric}')
    print(f'Found best metric for the SVM model with gamma:{gamma} and C:{C}')
    print('\n')
