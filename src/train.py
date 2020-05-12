import time
import inspect, sys #Losses
import csv
import math
import tqdm
import os
import os.path
import numpy as np
import keras
import keras.backend
import tensorflow as tf

from utils.check import check_weights_exist
from utils.check import shuffle_split_array
from utils.logger import write_summary
from utils.data_structures import ExecutionInformation
# from utils.data_structures import FoldKeys
# from utils.data_structures import FoldPaths
from utils.data_structures import DataGenerator
from utils.evaluation import evaluate
from utils.disambiguator import select_optimizer
from utils.disambiguator import select_loss
import utils.architecture as arch
import utils.architecture2D as arch2D


def train_cross_val(KFolds, config, data, metrics, metrics_CV2, results, results_CV2):
    # Set random seed
    if (config.seed != None) and isinstance(config.seed, int): 
        tf.set_random_seed(config.seed)
        np.random.seed(config.seed)

    # Ensure that the model will fit in GPU memory
    good_batch_size = False 

    # Check whether weight files exist before starting, if evaluate flag is active
    if (config.evaluate): check_weights_exist(config, KFolds)

    for fold in range(len(KFolds)):
        # Keep track of training time
        t_start = time.time()
        (train_keys, test_keys)  = KFolds[fold]
        # (train_keys, valid_keys) = shuffle_split_array(train_keys, config.val_split)

        # Retrieve fold information
        # execinfo = FoldKeys(train_keys=train_keys, valid_keys=valid_keys, test_keys=test_keys)
        execinfo = ExecutionInformation(config, fold, train_keys, test_keys, config.evaluate)

        # Train a single fold
        good_batch_size = train_fold(config, data, execinfo, metrics, metrics_CV2, results, results_CV2, good_batch_size, True)

        # Check total training time
        time_elapsed = time.time() - t_start
        print('\n * Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    # Retrieve patient ID from lead data
    IDs = np.asarray(list(set(train_keys.tolist() + test_keys.tolist()))) # Avoid duplicates

    # Sanity check
    if len(IDs) != len(train_keys) + len(test_keys):
        raise ValueError("Some record shared between train and test!")

    # Evaluation of whole dataset
    execinfo = ExecutionInformation(config, None, None, IDs, True)
    evaluate(None, config, data, execinfo, metrics, metrics_CV2, results, results_CV2, False)


def train_all(IDs, config, data, metrics, metrics_CV2, results, results_CV2):
    # Set random seed
    if (config.seed != None) and isinstance(config.seed, int): 
        tf.set_random_seed(config.seed)
        np.random.seed(config.seed)

    # Keep track of training time
    t_start = time.time()

    # "iif" evaluate, test on whole database: useful for testing on other DB's
    if config.evaluate:
        # Define the model according to the configuration
        # if config.strategy == 'single': model = arch.FlatNet(config).create_model()
        # else:                           model = arch2D.FlatNet(config).create_model()
        model = arch.FlatNet(config).create_model()

        # Select optimizer and loss + compile
        optim   = select_optimizer(config.optimizer.lower())(lr=config.learning_rate)
        loss    = select_loss(config.loss.lower())
        model.compile(optimizer=optim, loss=loss)

        execinfo = ExecutionInformation(config, None, None, IDs, True) # Define execution
        evaluate(model, config, data, execinfo, metrics, metrics_CV2, results, results_CV2, True) # Evaluate
    else:
        # Retrieve execution information
        execinfo = ExecutionInformation(config, None, IDs, None, False)

        # Check whether weight files exist before starting, if evaluate flag is active
        if (config.evaluate) and (not os.path.exists(execinfo.state)):
            raise FileNotFoundError("Weights file not found")

        # Train a single ""fold""
        _ = train_fold(config, data, execinfo, metrics, metrics_CV2, results, results_CV2, False, True)

    # Check total training time
    time_elapsed = time.time() - t_start
    print('\n * Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))


def train_fold(config, data, execinfo, metrics, metrics_CV2, results, results_CV2, good_batch_size, recompute):
    ############################# MODEL CREATION ############################
    # Clear cache
    keras.backend.clear_session()

    # Define the model according to the configuration
    # if config.strategy == 'single': model = arch.FlatNet(config).create_model()
    # else:                           model = arch2D.FlatNet(config).create_model()
    model = arch.FlatNet(config).create_model()

    # Select optimizer and loss + compile
    optim   = select_optimizer(config.optimizer.lower())(lr=config.learning_rate)
    loss    = select_loss(config.loss.lower())
    model.compile(optimizer=optim, loss=loss)

    write_summary(execinfo.summary, model)

    # If chosen and exists, load weights (fine-tuning, etc.)
    if (config.load_weights or config.evaluate) and os.path.exists(execinfo.state):
        model.load_weights(execinfo.state)

    # If the flag to evaluate has not been set, train the model
    if not config.evaluate:
        # Data generators
        GeneratorTrain = DataGenerator(execinfo.train, config, data)
        GeneratorValid = DataGenerator(execinfo.valid, config, data)

        # keras-specific train
        good_batch_size = train_epochs(config, model, data, execinfo, results, results_CV2, good_batch_size, GeneratorTrain, GeneratorValid)

        # Evaluate model
        if config.splitting.lower() == "cross_validation":
            evaluate(model, config, data, execinfo, metrics, metrics_CV2, results, results_CV2, recompute)
    else:
        if config.splitting.lower() == "cross_validation":
            evaluate(model, config, data, execinfo, metrics, metrics_CV2, results, results_CV2, recompute)

    return good_batch_size


def train_epochs(config, model, data, execinfo, results, results_CV2, good_batch_size, GeneratorTrain, GeneratorValid):
    # Keras Callbacks
    pointer    = keras.callbacks.ModelCheckpoint(filepath=execinfo.state, save_best_only=True)
    csv_logger = keras.callbacks.CSVLogger(execinfo.logger, append=False, separator=',')
    stopper    = keras.callbacks.EarlyStopping(patience=config.patience, restore_best_weights=True)

    # Avoid issues with model not fitting in GPU RAM
    while not good_batch_size:
        try:
            print('Current batch size: ' + str(config.batch_size))
            model.fit_generator(GeneratorTrain, validation_data=GeneratorValid, epochs=config.n_epochs, 
                                shuffle=True, callbacks=[pointer, stopper, csv_logger])
            print(" ")
            good_batch_size = True
        except tf.errors.ResourceExhaustedError: # Adjust Batch Size dynamically for cluster
            if config.batch_size == 1: # Avoid infinite loops
                exit()
            else:
                # Update batch size, *also for future folds*
                config.batch_size //= 2

                # Create again Generators with updated batch size
                GeneratorTrain = DataGenerator(execinfo.train, config, data)
                GeneratorValid = DataGenerator(execinfo.valid, config, data)
    
    if not(execinfo.fold in [0, None]):
        model.fit_generator(GeneratorTrain, validation_data=GeneratorValid, epochs=config.n_epochs, 
                            shuffle=True, callbacks=[pointer, stopper, csv_logger])
    
    return good_batch_size






