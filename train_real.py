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

    ##### 2. Load QTDB #####
    dataset             = pd.read_csv(os.path.join(input_files,'QTDB','Dataset.csv'), index_col=0)
    dataset             = dataset.sort_index(axis=1)
    labels              = np.asarray(list(dataset)) # In case no data augmentation is applied
    description         = dataset.describe()
    group               = {k: '_'.join(k.split('_')[:-1]) for k in dataset}
    unique_ids          = list(set([k.split('_')[0] for k in dataset]))

    # Load validity
    validity            = sak.load_data(os.path.join(input_files,'QTDB','validity.csv'))

    # Load fiducials
    Pon_QTDB            = sak.load_data(os.path.join(input_files,'QTDB','PonNew.csv'))
    Poff_QTDB           = sak.load_data(os.path.join(input_files,'QTDB','PoffNew.csv'))
    QRSon_QTDB          = sak.load_data(os.path.join(input_files,'QTDB','QRSonNew.csv'))
    QRSoff_QTDB         = sak.load_data(os.path.join(input_files,'QTDB','QRSoffNew.csv'))
    Ton_QTDB            = sak.load_data(os.path.join(input_files,'QTDB','TonNew.csv'))
    Toff_QTDB           = sak.load_data(os.path.join(input_files,'QTDB','ToffNew.csv'))

    # Generate masks & signals
    signal_QTDB = {}
    segmentation_QTDB = {}
    for k in tqdm.tqdm(QRSon_QTDB):
        # Check file exists and all that
        if k not in validity:
            print("Issue with file {}, continuing...".format(k))
            continue

        # Store signal
        signal = dataset[k][validity[k][0]:validity[k][1]].values
        signal = sak.signal.on_off_correction(signal)
        amplitude = np.median(sak.signal.moving_lambda(signal,200,sak.signal.abs_max))
        signal = signal/amplitude
        signal_QTDB[k] = signal[None,]
        
        # Generate boolean mask
        segmentation = np.zeros((3,dataset.shape[0]),dtype=bool)
        if k in Pon_QTDB:
            for on,off in zip(Pon_QTDB[k],Poff_QTDB[k]):
                segmentation[0,on:off] = True
        if k in QRSon_QTDB:
            for on,off in zip(QRSon_QTDB[k],QRSoff_QTDB[k]):
                segmentation[1,on:off] = True
        if k in Ton_QTDB:
            for on,off in zip(Ton_QTDB[k],Toff_QTDB[k]):
                segmentation[2,on:off] = True
        
        segmentation_QTDB[k] = segmentation[:,validity[k][0]:validity[k][1]]
        
    ##### 4. Generate random splits #####
    all_keys_real = {}
    for k in list(signal_QTDB) + list(segmentation_QTDB):
        uid = k.split("###")[0].split("_")[0].split("-")[0]
        if uid not in all_keys_real:
            all_keys_real[uid] = [k]
        else:
            if k not in all_keys_real[uid]:
                all_keys_real[uid].append(k)

    # 4.2. Get database and file
    filenames = []
    database = []
    for k in all_keys_real:
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
    shutil.copyfile("./train_real.py",os.path.join(target_path,model_name,"train_real.py"))
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
        # Real keys
        train_keys_real, valid_keys_real = ([],[])
        for k in np.array(filenames)[ix_train]: 
            if k in all_keys_real: train_keys_real += all_keys_real[k]
        for k in np.array(filenames)[ix_valid]: 
            if k in all_keys_real: valid_keys_real += all_keys_real[k]

        # Avoid repetitions
        train_keys_real = list(set(train_keys_real))
        valid_keys_real = list(set(valid_keys_real))

        # ~~~~~~~~~~~~~~~~~~~~~~ Refine real set ~~~~~~~~~~~~~~~~~~~~~~~
        signal_QTDB_train       = {k:       signal_QTDB[k] for k in       signal_QTDB if k in train_keys_real}
        signal_QTDB_valid       = {k:       signal_QTDB[k] for k in       signal_QTDB if k in valid_keys_real}
        segmentation_QTDB_train = {k: segmentation_QTDB[k] for k in segmentation_QTDB if k in train_keys_real}
        segmentation_QTDB_valid = {k: segmentation_QTDB[k] for k in segmentation_QTDB if k in valid_keys_real}


        # Prepare folders
        execution["save_directory"] = os.path.join(target_path, model_name, "fold_{}".format(i+1))
        if not os.path.isdir(execution["save_directory"]):
            pathlib.Path(execution["save_directory"]).mkdir(parents=True, exist_ok=True)
        
        # Define real datasets
        dataset_train_real = src.data.OversampledDatasetQTDB(signal_QTDB_train,segmentation_QTDB_train,execution["dataset"]["N"],128,72622)
        dataset_valid_real = src.data.OversampledDatasetQTDB(signal_QTDB_valid,segmentation_QTDB_valid,execution["dataset"]["N"],128,72622)

        # Create dataloaders
        execution["loader"]["shuffle"] = True
        loader_train = torch.utils.data.DataLoader(dataset_train_real, **execution["loader"])
        loader_valid = torch.utils.data.DataLoader(dataset_valid_real, **execution["loader"])

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


