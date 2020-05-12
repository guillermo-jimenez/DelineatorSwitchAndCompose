from os.path import exists
from utils.data_structures import ExecutionInformation
from numpy.random import permutation
from math import ceil

def check_weights_exist(config, KFolds):
    for fold in range(len(KFolds)):
        (train_keys, test_keys) = KFolds[fold]

        execinfo = ExecutionInformation(config, fold, train_keys, test_keys, config.evaluate)

        if not exists(execinfo.state):
            raise FileNotFoundError("Weights file not found")


def shuffle_split_array(array, ptg=0.2):
    array           = permutation(array)
    end_array       = array[:ceil(ptg*len(array))]
    beginning_array = array[ceil(ptg*len(array)):]

    return beginning_array, end_array

    
