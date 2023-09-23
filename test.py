import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from utils import tune_hparams,train_dev_test_split

# Load the digits dataset from scikit-learn
digits = datasets.load_digits()

# Split the dataset into training and testing sets
X_train,y_train, X_dev, Y_dev, X_test, y_test = train_dev_test_split(digits.images, digits.target, 0.2,0.1)

# Define a function to resize images and evaluate performance
def evaluate_performance(image_size):
    # Resize the training and testing images
    X_train_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_train])
    X_test_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_test])
    X_dev_resized = np.array([resize(image, image_size, anti_aliasing=True) for image in X_dev])
    
    # Flatten the resized images
    X_train_flat = X_train_resized.reshape(len(X_train_resized), -1)
    X_test_flat = X_test_resized.reshape(len(X_test_resized), -1)
    X_dev_flat = X_dev_resized.reshape(len(X_dev_resized), -1)
    
    # Train a Support Vector Machine (SVM) classifier
    svm_classifier = SVC(kernel='linear', C=1)
    svm_classifier.fit(X_train_flat, y_train)
    
    # Predict on the test set
    y_train_pred = svm_classifier.predict(X_train_flat)
    y_test_pred = svm_classifier.predict(X_test_flat)
    y_dev_pred = svm_classifier.predict(X_dev_flat)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    dev_accuracy = accuracy_score(Y_dev, y_dev_pred)
    
    return train_accuracy,test_accuracy,dev_accuracy

# Define a list of image sizes to evaluate
image_sizes = [(4, 4), (6, 6), (8, 8)]

# Evaluate performance for each image size
for size in image_sizes:
    train_accuracy,test_accuracy,dev_accuracy = evaluate_performance(size)
    print(f"Image Size {size}: Train Accuracy with Size {0.7} = {train_accuracy:.2f} Test Accuracy  with Size {0.2} = {test_accuracy:.2f} Dev Accuracy with Size {0.1} = {dev_accuracy:.2f}")
