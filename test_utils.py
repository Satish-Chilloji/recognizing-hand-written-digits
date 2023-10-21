from utils import get_hyperparameter_combinations

def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 4

def test_wrong_answer():
    assert not inc(3) == 5

# def test_wrong_answer():
#     assert inc(3) == 5


def test_hyparam_combinations():
    gamma_list=[0.001,0.01]
    C_list=[1]
    # gamma_list=[0.001,0.01,0.1,1]
    # C_list=[1,10,100,1000]
    h_params={}
    h_params['gamma']=gamma_list
    h_params['C']=C_list
    h_params_combinations=get_hyperparameter_combinations(h_params)
    # assert len(h_params_combinations)==len(gamma_list)*len(C_list)
    expected_param_combo_1={'gamma':0.001,'C':1}
    expected_param_combo_2={'gamma':0.01,'C':1}
    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

