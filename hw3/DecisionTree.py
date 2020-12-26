import pickle
from dt import divide, entropy, info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test

with open('hw3_data/dt/data.pkl', 'rb') as f:
    train_data, test_data, attr_vals_list, attr_names = pickle.load(f)

    for dede in divide(train_data, 1, attr_vals_list):
        for nene in dede:
            print(nene)
