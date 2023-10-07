import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, 

def predict_and_eval(model, X_test, y_test):
    predicted=model.predict(X_test)

def train_dev_test_split(data, label, test_frac, dev_frac):

    X_train_dev, X_test, Y_train_dev, Y_test = train_test_split(
        data, label, test_size=test_frac, shuffle=True
    )
    #train_size=1-test_frac-dev_frac
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X_train_dev, Y_train_dev, test_size=dev_frac, shuffle=True
    )

    return X_train,Y_train, X_dev, Y_dev, X_test, Y_test

def get_combinations(param_name,param_values,base_combinations):
    new_combinations=[]
    for value in param_values:
        for combinations in base_combinations:
            combinations[param_name]=value
            new_combinations.append(combinations.copy())
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_list):
    base_combinations=[{}]
    for param_name, param_values in dict_of_param_list.items():
        base_combinations=get_combinations(param_name,param_values,base_combinations)
    return base_combinations

def tune_hparams(X_train, Y_train, X_dev, Y_dev, list_of_all_param_combination):
    best_acc_so_far=-1
    best_train_metric=-1
    for gamma, C in list_of_all_param_combination:
        model=svm.SVC(C=C,gamma=gamma)
        model.fit(X_train,Y_train)
        predicted_dev=model.predict(X_dev)
        cur_metric = metrics.accuracy_score(y_pred=predicted_dev, y_true=Y_dev)
        predicted_train=model.predict(X_train)
        train_metric = metrics.accuracy_score(y_pred=predicted_train, y_true=Y_train)

        if cur_metric > best_acc_so_far:
            best_acc_so_far = cur_metric
            best_train_metric = train_metric
            best_model=model
            b_gamma=gamma
            b_C=C
    return best_model,b_gamma,b_C,best_acc_so_far,best_train_metric

    
