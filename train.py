from argparse import ArgumentParser

import os
import os.path
import skimage
import skimage.segmentation
import sklearn.preprocessing
import sklearn.model_selection
import math
import shutil
import pathlib
import glob
import shutil
import uuid
import random
import platform
import ecgdetectors
import torch
import torchvision
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import pandas as pd
import networkx
import wfdb
import json
import tqdm
import dill
import pickle
import matplotlib.pyplot as plt

import scipy.stats

import src.data
import utils
import utils.wavelet
import utils.data
import utils.data.augmentation
import utils.visualization
import utils.visualization.plot
import utils.torch
import utils.torch.nn
import utils.torch.nn as nn
import utils.torch.loss
import utils.torch.train
import utils.torch.data
import utils.torch.preprocessing
import utils.torch.models
import utils.torch.models.lego
import utils.torch.models.variational
import utils.torch.models.classification

from utils.signal import StandardHeader

def smooth(x: np.ndarray, window_size: int, conv_mode: str = "same"):
    x = np.pad(np.copy(x),(window_size,window_size),"edge")
    window = np.hamming(window_size)/(window_size//2)
    x = np.convolve(x, window, mode=conv_mode)
    x = x[window_size:-window_size]
    return x


def main(config_file, model_name, input_files):
    ##### 1. Load data #####
    # 1.0. Open configuration file
    with open(config_file, "r") as f:
        execution = json.load(f)

    # 1.1. Load individual segments
    P = utils.pickleload(os.path.join(input_files,"Psignal_new.pkl"))
    PQ = utils.pickleload(os.path.join(input_files,"PQsignal_new.pkl"))
    QRS = utils.pickleload(os.path.join(input_files,"QRSsignal_new.pkl"))
    ST = utils.pickleload(os.path.join(input_files,"STsignal_new.pkl"))
    T = utils.pickleload(os.path.join(input_files,"Tsignal_new.pkl"))
    TP = utils.pickleload(os.path.join(input_files,"TPsignal_new.pkl"))

    Pamplitudes = utils.pickleload(os.path.join(input_files,"Pamplitudes_new.pkl"))
    PQamplitudes = utils.pickleload(os.path.join(input_files,"PQamplitudes_new.pkl"))
    QRSamplitudes = utils.pickleload(os.path.join(input_files,"QRSamplitudes_new.pkl"))
    STamplitudes = utils.pickleload(os.path.join(input_files,"STamplitudes_new.pkl"))
    Tamplitudes = utils.pickleload(os.path.join(input_files,"Tamplitudes_new.pkl"))
    TPamplitudes = utils.pickleload(os.path.join(input_files,"TPamplitudes_new.pkl"))

    # 1.2. Get amplitude distribution
    Pdistribution   = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(Pamplitudes.values()))))
    PQdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(PQamplitudes.values()))))
    QRSdistribution = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.hstack((np.array(list(QRSamplitudes.values())), 2-np.array(list(QRSamplitudes.values()))))))
    STdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(STamplitudes.values()))))
    Tdistribution   = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(Tamplitudes.values()))))
    TPdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(TPamplitudes.values()))))

    # 1.3. Smooth all
    window = 5
    P   = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth(  P[k],window)),metric=utils.signal.abs_max) for k in   P}
    PQ  = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth( PQ[k],window)),metric=utils.signal.abs_max) for k in  PQ}
    QRS = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth(QRS[k],window)),metric=utils.signal.abs_max) for k in QRS}
    ST  = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth( ST[k],window)),metric=utils.signal.abs_max) for k in  ST}
    T   = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth(  T[k],window)),metric=utils.signal.abs_max) for k in   T}
    TP  = {k: utils.data.ball_scaling(utils.signal.on_off_correction(smooth( TP[k],window)),metric=utils.signal.abs_max) for k in  TP}

    # 1.4. Split into train and test
    all_keys = {}
    for k in list(P) + list(PQ) + list(QRS) + list(ST) + list(T) + list(TP):
        uid = k.split("###")[0].split("_")[0].split("-")[0]
        if uid not in all_keys:
            all_keys[uid] = [k]
        else:
            all_keys[uid].append(k)
            
    # Get database and file
    filenames = []
    database = []
    for k in all_keys:
        filenames.append(k)
        if k.startswith("SOO"):
            database.append(0)
        elif k.startswith("sel"):
            database.append(1)
        else:
            database.append(2)
    filenames = np.array(filenames)
    database = np.array(database)

    # Set random seed for the execution and perform train/test splitting
    random.seed(execution["seed"])
    np.random.seed(execution["seed"])
    torch.random.manual_seed(execution["seed"])
    splitter = sklearn.model_selection.StratifiedKFold(5).split(filenames,database)

    ##### 2. Train folds #####
    ### Loss
    criterion = lambda X,y,y_pred: utils.torch.loss.DiceLoss()(y_pred, y)
    metric    = lambda X,y,y_pred: utils.torch.loss.DiceLoss()(y_pred, y)

    # Save model-generating files
    original_path = execution["save_directory"] # Store original output path for future usage
    original_length = execution["dataset"]["length"]
    if not os.path.isdir(os.path.join(original_path,model_name)):
        pathlib.Path(os.path.join(original_path,model_name)).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("./train.py",os.path.join(original_path,model_name,"train.py"))
    shutil.copyfile("./src/data.py",os.path.join(original_path,model_name,"data.py"))
    shutil.copyfile("./src/metrics.py",os.path.join(original_path,model_name,"metrics.py"))
    shutil.copyfile(config_file,os.path.join(original_path,model_name,os.path.split(config_file)[1]))
    
    # Structure to save splitter
    all_folds_test = {}
    
    # Iterate over folds
    for i,(ix_train,ix_valid) in enumerate(splitter):
        print("################# FOLD {} #################".format(i+1))
        train_keys, valid_keys = ([],[])
        for k in np.array(filenames)[ix_train]: train_keys += all_keys[k]
        for k in np.array(filenames)[ix_valid]: valid_keys += all_keys[k]

        # Save fold"s validation files for later usage
        all_folds_test["fold_{}".format(i+1)] = np.array(filenames)[ix_valid]
        
        # Divide train/valid segments
        Ptrain   = {k:   P[k] for k in   P if k in train_keys}
        PQtrain  = {k:  PQ[k] for k in  PQ if k in train_keys}
        QRStrain = {k: QRS[k] for k in QRS if k in train_keys}
        STtrain  = {k:  ST[k] for k in  ST if k in train_keys}
        Ttrain   = {k:   T[k] for k in   T if k in train_keys}
        TPtrain  = {k:  TP[k] for k in  TP if k in train_keys}

        Pvalid   = {k:   P[k] for k in   P if k in valid_keys}
        PQvalid  = {k:  PQ[k] for k in  PQ if k in valid_keys}
        QRSvalid = {k: QRS[k] for k in QRS if k in valid_keys}
        STvalid  = {k:  ST[k] for k in  ST if k in valid_keys}
        Tvalid   = {k:   T[k] for k in   T if k in valid_keys}
        TPvalid  = {k:  TP[k] for k in  TP if k in valid_keys}

        # Prepare folders
        execution["save_directory"] = os.path.join(original_path, model_name, "fold_{}".format(i+1))
        if not os.path.isdir(execution["save_directory"]):
            pathlib.Path(execution["save_directory"]).mkdir(parents=True, exist_ok=True)
        
        # Define datasets
        dataset_train = src.data.Dataset(Ptrain, QRStrain, Ttrain, PQtrain, STtrain, TPtrain, 
                                         Pdistribution, QRSdistribution, Tdistribution, PQdistribution, 
                                         STdistribution, TPdistribution, **execution["dataset"])
        execution["dataset"]["length"] = execution["dataset"]["length"]//4 # On synthetic data, not so useful to do intensive validation
        dataset_valid = src.data.Dataset(Pvalid, QRSvalid, Tvalid, PQvalid, STvalid, TPvalid, 
                                         Pdistribution, QRSdistribution, Tdistribution, PQdistribution, 
                                         STdistribution, TPdistribution, **execution["dataset"])
        execution["dataset"]["length"] = original_length # On synthetic data, not so useful to do intensive validation

        # Create dataloaders
        loader_train = torch.utils.data.DataLoader(dataset_train, **execution["loader"])
        loader_valid = torch.utils.data.DataLoader(dataset_valid, **execution["loader"])

        # Define model
        model = nn.ModelGraph(execution["model"]).float().cuda()
        
        # Train model
        state = {
            "epoch"         : 0,
            "device"        : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "optimizer"     : utils.class_selector("torch.optim",execution["optimizer"]["class"])(model.parameters(), **execution["optimizer"]["arguments"]),
            "root_dir"      : "./"
        }
        if "scheduler" in execution:
            state["scheduler"] = utils.class_selector("torch.optim.lr_scheduler",execution["scheduler"]["class"])(state["optimizer"], **execution["scheduler"]["arguments"])

        # Train model (auto-saves to same location as above)
        utils.torch.train.train_model(model,state,execution,loader_train,loader_valid,criterion,metric,smaller=True)

    utils.save_data(all_folds_test,os.path.join(original_path,model_name,"validation_files.csv"))
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="location of config file (.json)")
    parser.add_argument("--input_files", type=str, required=True, help="location of (pickled) input data")
    parser.add_argument("--model_name",  type=str, required=True, help="model name")
    args = parser.parse_args()

    main(args.config_file, args.model_name, args.input_files)


