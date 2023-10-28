from utils import get_hyperparameter_combinations
from utils import train_dev_test_split,read_digits

def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4

def test_wrong_answer():
    assert not inc(3) == 5

def test_hyparam_combinations():
    # Test case to check that all the possible combination of paramas are indeed generated.
    gamma_list=[0.001,0.01,0.1,1]
    C_list=[1,10,100,1000]
    h_params={}
    h_params['gamma']=gamma_list
    h_params['C']=C_list
    h_params_combinations=get_hyperparameter_combinations(h_params)
    assert len(h_params_combinations)==len(gamma_list)*len(C_list)
    expected_param_combo_1={'gamma':0.001,'C':1}
    expected_param_combo_2={'gamma':0.01,'C':1}
    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

def test_data_split():
    X, y =read_digits()
    #import pdb; pdb.set_trace()
    X = X[:100,:]
    y = y[:100]
    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - test_size - dev_size

    X_train,Y_train, X_dev, Y_dev, X_test, Y_test = train_dev_test_split(X, y, test_frac=test_size, dev_frac=dev_size)
    print(len(X))
    assert (len(X_train) == int(len(X)*train_size)) and (len(X_test) == len(X)*test_size) and (len(X_dev) == len(X)*dev_size)