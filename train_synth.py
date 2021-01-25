from argparse import ArgumentParser

import os
import os.path
import skimage
import skimage.segmentation
import sklearn.preprocessing
import sklearn.model_selection
import math
import pathlib
import glob
import shutil
import uuid
import random
import platform
import itertools
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
import sak
import sak.signal.wavelet
import sak.data
import sak.data.augmentation
import sak.data.preprocessing
import sak.visualization
import sak.visualization.signal
import sak.torch
import sak.torch.nn
import sak.torch.nn as nn
import sak.torch.train
import sak.torch.data
import sak.torch.models
import sak.torch.models.lego
import sak.torch.models.variational
import sak.torch.models.classification

from sak.signal import StandardHeader

def smooth(x: np.ndarray, window_size: int, conv_mode: str = "same"):
    x = np.pad(np.copy(x),(window_size,window_size),"edge")
    window = np.hamming(window_size)/(window_size//2)
    x = np.convolve(x, window, mode=conv_mode)
    x = x[window_size:-window_size]
    return x


def main(config_file, model_name, input_files, bool_hpc):
    ##### 1. Load configuration file #####
    with open(config_file, "r") as f:
        execution = json.load(f)

    execution["root_directory"] = os.path.expanduser(execution["root_directory"])
    execution["save_directory"] = os.path.expanduser(execution["save_directory"])

    # NO ITERATOR FOR HPC, WASTE OF MEMORY
    if bool_hpc:
        execution["iterator"] = "none"

    ##### 2. Load synthetic dataset #####
    # 2.1. Load individual segments
    P = sak.pickleload(os.path.join(input_files,"Psignal_new.pkl"))
    PQ = sak.pickleload(os.path.join(input_files,"PQsignal_new.pkl"))
    QRS = sak.pickleload(os.path.join(input_files,"QRSsignal_new.pkl"))
    ST = sak.pickleload(os.path.join(input_files,"STsignal_new.pkl"))
    T = sak.pickleload(os.path.join(input_files,"Tsignal_new.pkl"))
    TP = sak.pickleload(os.path.join(input_files,"TPsignal_new.pkl"))

    Pamplitudes = sak.pickleload(os.path.join(input_files,"Pamplitudes_new.pkl"))
    PQamplitudes = sak.pickleload(os.path.join(input_files,"PQamplitudes_new.pkl"))
    QRSamplitudes = sak.pickleload(os.path.join(input_files,"QRSamplitudes_new.pkl"))
    STamplitudes = sak.pickleload(os.path.join(input_files,"STamplitudes_new.pkl"))
    Tamplitudes = sak.pickleload(os.path.join(input_files,"Tamplitudes_new.pkl"))
    TPamplitudes = sak.pickleload(os.path.join(input_files,"TPamplitudes_new.pkl"))

    # 2.2. Get amplitude distribution
    Pdistribution   = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(Pamplitudes.values()))))
    PQdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(PQamplitudes.values()))))
    QRSdistribution = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.hstack((np.array(list(QRSamplitudes.values())), 2-np.array(list(QRSamplitudes.values()))))))
    STdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(STamplitudes.values()))))
    Tdistribution   = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(Tamplitudes.values()))))
    TPdistribution  = scipy.stats.lognorm(*scipy.stats.lognorm.fit(np.array(list(TPamplitudes.values()))))

    # 2.3. Smooth all
    window = 5
    P   = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth(  P[k],window)),metric=sak.signal.abs_max) for k in   P}
    PQ  = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth( PQ[k],window)),metric=sak.signal.abs_max) for k in  PQ}
    QRS = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth(QRS[k],window)),metric=sak.signal.abs_max) for k in QRS}
    ST  = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth( ST[k],window)),metric=sak.signal.abs_max) for k in  ST}
    T   = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth(  T[k],window)),metric=sak.signal.abs_max) for k in   T}
    TP  = {k: sak.data.ball_scaling(sak.signal.on_off_correction(smooth( TP[k],window)),metric=sak.signal.abs_max) for k in  TP}


    ##### 3. Generate random splits #####
    # 3.1. Split into train and test
    all_keys_synthetic = {}
    for k in list(P) + list(PQ) + list(QRS) + list(ST) + list(T) + list(TP):
        uid = k.split("###")[0].split("_")[0].split("-")[0]
        if uid not in all_keys_synthetic:
            all_keys_synthetic[uid] = [k]
        else:
            all_keys_synthetic[uid].append(k)

    # 3.2. Get database and file
    filenames = []
    database = []
    for k in all_keys_synthetic:
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
    splits = list(splitter)
    indices_train = [s[0] for s in splits]
    indices_valid = [s[1] for s in splits]

    ##### 5. Train folds #####
    # 5.1. Save model-generating files
    target_path = execution["save_directory"] # Store original output path for future usage
    original_length = execution["dataset"]["length"]
    if not os.path.isdir(os.path.join(target_path,model_name)):
        pathlib.Path(os.path.join(target_path,model_name)).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("./train_multi.py",os.path.join(target_path,model_name,"train_multi.py"))
    shutil.copyfile("./src/data.py",os.path.join(target_path,model_name,"data.py"))
    shutil.copyfile("./src/metrics.py",os.path.join(target_path,model_name,"metrics.py"))
    shutil.copyfile("./sak/torch/nn/modules/loss.py",os.path.join(target_path,model_name,"loss.py"))
    shutil.copyfile(config_file,os.path.join(target_path,model_name,os.path.split(config_file)[1]))
    
    # 5.2. Save folds of valid files
    all_folds_test = {"fold_{}".format(i+1): np.array(filenames)[ix_valid] for i,ix_valid in enumerate(indices_valid)}
    sak.save_data(all_folds_test,os.path.join(target_path,model_name,"validation_files.csv"))

    # 5.3. Iterate over folds
    for i,(ix_train,ix_valid) in enumerate(zip(indices_train,indices_valid)):
        print("################# FOLD {} #################".format(i+1))
        # Synthetic keys
        train_keys_synthetic, valid_keys_synthetic = ([],[])
        for k in np.array(filenames)[ix_train]: 
            train_keys_synthetic += all_keys_synthetic[k]
        for k in np.array(filenames)[ix_valid]: 
            valid_keys_synthetic += all_keys_synthetic[k]

        # Avoid repetitions
        train_keys_synthetic = list(set(train_keys_synthetic))
        valid_keys_synthetic = list(set(valid_keys_synthetic))

        # ~~~~~~~~~~~~~~~~~~~~ Refine synthetic set ~~~~~~~~~~~~~~~~~~~~
        # Divide train/valid segments
        Ptrain   = {k:   P[k] for k in   P if k in train_keys_synthetic}
        PQtrain  = {k:  PQ[k] for k in  PQ if k in train_keys_synthetic}
        QRStrain = {k: QRS[k] for k in QRS if k in train_keys_synthetic}
        STtrain  = {k:  ST[k] for k in  ST if k in train_keys_synthetic}
        Ttrain   = {k:   T[k] for k in   T if k in train_keys_synthetic}
        TPtrain  = {k:  TP[k] for k in  TP if k in train_keys_synthetic}

        Pvalid   = {k:   P[k] for k in   P if k in valid_keys_synthetic}
        PQvalid  = {k:  PQ[k] for k in  PQ if k in valid_keys_synthetic}
        QRSvalid = {k: QRS[k] for k in QRS if k in valid_keys_synthetic}
        STvalid  = {k:  ST[k] for k in  ST if k in valid_keys_synthetic}
        Tvalid   = {k:   T[k] for k in   T if k in valid_keys_synthetic}
        TPvalid  = {k:  TP[k] for k in  TP if k in valid_keys_synthetic}

        # Prepare folders
        execution["save_directory"] = os.path.join(target_path, model_name, "fold_{}".format(i+1))
        if not os.path.isdir(execution["save_directory"]):
            pathlib.Path(execution["save_directory"]).mkdir(parents=True, exist_ok=True)
        
        # Define synthetic datasets
        execution["dataset"]["length"] = original_length + 7086 # Size difference of oversampled multi dataset (equal computational budget)
        dataset_train_synthetic = src.data.Dataset(Ptrain, QRStrain, Ttrain, PQtrain, STtrain, TPtrain, 
                                                   Pdistribution, QRSdistribution, Tdistribution, PQdistribution, 
                                                   STdistribution, TPdistribution, **execution["dataset"])
        execution["dataset"]["length"] = original_length//4 + 7086 # Size difference of oversampled multi dataset (equal computational budget)
        dataset_valid_synthetic = src.data.Dataset(Pvalid, QRSvalid, Tvalid, PQvalid, STvalid, TPvalid, 
                                                   Pdistribution, QRSdistribution, Tdistribution, PQdistribution, 
                                                   STdistribution, TPdistribution, **execution["dataset"])
        execution["dataset"]["length"] = original_length

        # Create dataloaders
        execution["loader"]["shuffle"] = True # Force shuffle
        loader_train = torch.utils.data.DataLoader(dataset_train_synthetic, **execution["loader"])
        loader_valid = torch.utils.data.DataLoader(dataset_valid_synthetic, **execution["loader"])

        # Define model
        model = sak.from_dict(execution["model"]).float().cuda()
        
        # Train model
        state = {
            "epoch"         : 0,
            "device"        : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "optimizer"     : sak.class_selector(execution["optimizer"]["class"])(model.parameters(), **execution["optimizer"]["arguments"]),
            "root_dir"      : "./"
        }
        if "scheduler" in execution:
            state["scheduler"] = sak.class_selector(execution["scheduler"]["class"])(state["optimizer"], **execution["scheduler"]["arguments"])

        # Train model (auto-saves to same location as above)
        sak.torch.train.train_valid_model(model,state,execution,loader_train,loader_valid)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="location of config file (.json)")
    parser.add_argument("--input_files", type=str, required=True, help="location of (pickled) input data")
    parser.add_argument("--model_name",  type=str, required=True, help="model name")
    parser.add_argument("--hpc",         type=bool, default=False, help="mark if executed in HPC")
    args = parser.parse_args()

    main(args.config_file, args.model_name, args.input_files, args.hpc)


