# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False)
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=dev_size, shuffle=False)
    return X_train,X_test,X_dev,y_train,y_test,y_dev

def predict_and_eval(model, X_test, y_test):
    predicted=model.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

if __name__=='__main__':
    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    X=data
    y=digits.target

    X_train,X_test,X_dev,y_train,y_test,y_dev=split_train_dev_test(X,y,0.2,0.2)

    # Create a classifier: a support vector classifier
    model = svm.SVC(gamma=0.001)

    # Learn the digits on the train subset
    model.fit(X_train, y_train)

    predict_and_eval(model,X_test,y_test)