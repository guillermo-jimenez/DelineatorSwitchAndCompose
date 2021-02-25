from argparse import ArgumentParser

import os
import os.path
import sys
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
import cv2

import src.data
import src.metrics
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
import sak.torch.nn
import sak.torch.train
import sak.torch.data
import sak.torch.models
import sak.torch.models.lego
import sak.torch.models.variational
import sak.torch.models.classification

from sak.signal import StandardHeader

def smooth(x: np.ndarray, window_size: int, conv_mode: str = 'same'):
    x = np.pad(np.copy(x),(window_size,window_size),'edge')
    window = np.hamming(window_size)/(window_size//2)
    x = np.convolve(x, window, mode=conv_mode)
    x = x[window_size:-window_size]
    return x


# basedir = '/media/guille/DADES/DADES/Delineator'
# model_name = 'TestIncreasingF1LossNoSigmoidNoTPFPFNWithConstantValue1'
def main(basedir, model_name, hpc, model_type, batch_size, window_size):
    #########################################################################
    # Load validation distribution
    valid_folds = sak.load_data(os.path.join(basedir,'TrainedModels',model_name,'validation_files.csv'),dtype=None)
    fold_of_file = {fname: k for k in valid_folds for fname in valid_folds[k]}

    #########################################################################
    # Load models
    models = {}
    for i in range(5):
        if os.path.isfile(os.path.join(basedir,'TrainedModels',model_name,'fold_{}'.format(i+1),'{}.model'.format(model_type))):
            models['fold_{}'.format(i+1)] = torch.load(os.path.join(basedir,'TrainedModels',model_name,'fold_{}'.format(i+1),'{}.model'.format(model_type)),pickle_module=dill).eval().float()
        else:
            print("File for fold {} not found. Continuing...".format(i+1))

    #########################################################################
    # Load QT database
    dataset = pd.read_csv(os.path.join(basedir,'QTDB','Dataset.csv'), index_col=0)
    dataset = dataset.sort_index(axis=1)
    validity = sak.load_data(os.path.join(basedir, 'QTDB', 'validity.csv'))

    Pon    = sak.load_data(os.path.join(basedir, 'QTDB', 'PonNew.csv'))
    Poff   = sak.load_data(os.path.join(basedir, 'QTDB', 'PoffNew.csv'))
    QRSon  = sak.load_data(os.path.join(basedir, 'QTDB', 'QRSonNew.csv'))
    QRSoff = sak.load_data(os.path.join(basedir, 'QTDB', 'QRSoffNew.csv'))
    Ton    = sak.load_data(os.path.join(basedir, 'QTDB', 'TonNew.csv'))
    Toff   = sak.load_data(os.path.join(basedir, 'QTDB', 'ToffNew.csv'))

    #########################################################################
    # Predict signals
    predictions = {}
    stride = window_size//(2**2)
    for i,(sig_name,signal) in enumerate(tqdm.tqdm(dataset.iteritems(),total=dataset.shape[1])):
        # Retrieve signal name and fold
        fname = sig_name.split('_')[0]
        if fname not in fold_of_file:
            continue
        fold = fold_of_file[fname]
        
        if fold not in models:
            continue
        
        if fname in ['sel35','sel36','sel103','sel232','sel310']: continue
            
        # Copy signal just in case
        signal = np.copy(signal.values)
        
        # Filter signal
        signal = sp.signal.filtfilt(*sp.signal.butter(4,   0.5/250., 'high'),signal.T).T
        signal = sp.signal.filtfilt(*sp.signal.butter(4, 125.0/250.,  'low'),signal.T).T

        # Normalize and pad signal for inputing in algorithm
        pad_size = math.ceil(signal.size/window_size)*window_size - signal.size
        signal = np.pad(signal, (0,pad_size), mode='edge')
        normalization = np.median(sak.signal.moving_lambda(signal,256,sak.signal.abs_max))
        normalized_signal = signal/normalization
        
        # Window signal for meeting the input size
        windowed_signal = (skimage.util.view_as_windows(normalized_signal, window_size, stride) - 0)[:,None,:]
        windowed_mask  = torch.zeros((windowed_signal.shape[0],3,window_size),dtype=float)
        
        # Predict masks
        with torch.no_grad():
            for i in range(0,windowed_signal.shape[0],batch_size):
                inputs = {"x": torch.tensor(windowed_signal[i:i+batch_size]).cuda().float()}
                windowed_mask[i:i+batch_size] = models[fold].cuda()(inputs)["sigmoid"]
                models[fold] = models[fold].cpu()
        windowed_mask = windowed_mask.cpu().detach().numpy()

        # Retrieve mask as 1D
        counter = np.zeros_like(signal, dtype=int)
        signal_mask = np.zeros((3,signal.size))

        for i in range(windowed_mask.shape[0]):
            counter[i*stride:i*stride+window_size] += 1
            signal_mask[:,i*stride:i*stride+window_size] += windowed_mask[i]
        signal_mask = (signal_mask/counter) > 0.5
        
        # Store prediction
        predictions[sig_name] = signal_mask

    #########################################################################
    # Retrieve onsets and offsets
    pon, poff, qrson, qrsoff, ton, toff = {},{},{},{},{},{}
    for k in tqdm.tqdm(predictions):
        pon[k],poff[k]     = sak.signal.get_mask_boundary(predictions[k][0,],aslist=False)
        qrson[k],qrsoff[k] = sak.signal.get_mask_boundary(predictions[k][1,],aslist=False)
        ton[k],toff[k]     = sak.signal.get_mask_boundary(predictions[k][2,],aslist=False)

        # Refine results
        val_on = validity[k][0::2].tolist()
        val_off = validity[k][1::2].tolist()
        
        # P wave
        joint_on = []
        joint_off = []
        for v_on,v_off in zip(val_on,val_off):
            tmp_on, tmp_off = src.metrics.filter_valid(pon[k],poff[k],v_on,v_off,operation='and')
            joint_on.append(tmp_on)
            joint_off.append(tmp_off)
        pon[k],poff[k] = np.concatenate(joint_on),np.concatenate(joint_off)
        
        # QRS wave
        joint_on = []
        joint_off = []
        for v_on,v_off in zip(val_on,val_off):
            tmp_on, tmp_off = src.metrics.filter_valid(qrson[k],qrsoff[k],v_on,v_off,operation='and')
            joint_on.append(tmp_on)
            joint_off.append(tmp_off)
        qrson[k],qrsoff[k] = np.concatenate(joint_on),np.concatenate(joint_off)
        
        # T wave
        joint_on = []
        joint_off = []
        for v_on,v_off in zip(val_on,val_off):
            tmp_on, tmp_off = src.metrics.filter_valid(ton[k],toff[k],v_on,v_off,operation='and')
            joint_on.append(tmp_on)
            joint_off.append(tmp_off)
        ton[k],toff[k] = np.concatenate(joint_on),np.concatenate(joint_off)
    
    #########################################################################
    # Produce metrics, single lead
    metrics_single = {}
    metrics_multi = {}

    for wave in ['p','qrs','t']:
        metrics_single[wave] = {}
        metrics_single[wave]['truepositives'] = {}
        metrics_single[wave]['falsepositives'] = {}
        metrics_single[wave]['falsenegatives'] = {}
        metrics_single[wave]['onerrors'] = {}
        metrics_single[wave]['offerrors'] = {}

        input_on   = eval('{}on'.format(wave.lower()))
        input_off  = eval('{}off'.format(wave.lower()))
        target_on  = eval('{}on'.format(wave.upper()))
        target_off = eval('{}off'.format(wave.upper()))
        
        # Compute metrics_single
        for k in input_on:
            if fname in ['sel35','sel36','sel103','sel232','sel310']: continue
            try:
                # Refine input and output's regions w/ validity vectors
                (input_on[k],input_off[k]) = src.metrics.filter_valid(input_on[k],input_off[k], validity[k][0], validity[k][1],operation='and')
                (target_on[k],target_off[k]) = src.metrics.filter_valid(target_on[k],target_off[k], validity[k][0], validity[k][1],operation='and')
                tp,fp,fn,dice,onerror,offerror = src.metrics.compute_metrics(input_on[k],input_off[k],target_on[k],target_off[k])
            except:
                continue
            metrics_single[wave]['truepositives'][k] = tp
            metrics_single[wave]['falsepositives'][k] = fp
            metrics_single[wave]['falsenegatives'][k] = fn
            metrics_single[wave]['onerrors'][k] = np.copy(np.array(onerror))
            metrics_single[wave]['offerrors'][k] = np.copy(np.array(offerror))

    # Multi-lead
    for wave in ['p','qrs','t']:
        metrics_multi[wave] = {}
        metrics_multi[wave]['truepositives'] = {}
        metrics_multi[wave]['falsepositives'] = {}
        metrics_multi[wave]['falsenegatives'] = {}
        metrics_multi[wave]['onerrors'] = {}
        metrics_multi[wave]['offerrors'] = {}

        input_on   = eval('{}on'.format(wave.lower()))
        input_off  = eval('{}off'.format(wave.lower()))
        target_on  = eval('{}on'.format(wave.upper()))
        target_off = eval('{}off'.format(wave.upper()))

        listkeys = [k.split('_')[0] for k in input_on]

        for k in listkeys:
            if k in ['sel35','sel36','sel103','sel232','sel310']: continue
            try:
                # Refine input and output's regions w/ validity vectors
                (input_on[k+'_0'],input_off[k+'_0']) = src.metrics.filter_valid(input_on[k+'_0'],input_off[k+'_0'], validity[k+'_0'][0], validity[k+'_0'][1],operation='and')
                (target_on[k+'_0'],target_off[k+'_0']) = src.metrics.filter_valid(target_on[k+'_0'],target_off[k+'_0'], validity[k+'_0'][0], validity[k+'_0'][1],operation='and')
                (input_on[k+'_1'],input_off[k+'_1']) = src.metrics.filter_valid(input_on[k+'_1'],input_off[k+'_1'], validity[k+'_1'][0], validity[k+'_1'][1],operation='and')
                (target_on[k+'_1'],target_off[k+'_1']) = src.metrics.filter_valid(target_on[k+'_1'],target_off[k+'_1'], validity[k+'_1'][0], validity[k+'_1'][1],operation='and')
                tp,fp,fn,dice,on,off = src.metrics.compute_QTDB_metrics(input_on[k+'_0'],input_off[k+'_0'],
                                                                        input_on[k+'_1'],input_off[k+'_1'],
                                                                        target_on[k+'_0'],target_off[k+'_0'])
            except:
                continue
            metrics_multi[wave]['truepositives'][k] = tp
            metrics_multi[wave]['falsepositives'][k] = fp
            metrics_multi[wave]['falsenegatives'][k] = fn
            metrics_multi[wave]['onerrors'][k] = on
            metrics_multi[wave]['offerrors'][k] = off

    #########################################################################
    # Get stupid metric string
    metrics_string = ""
    metrics_string += "\n# {}".format(model_name)
    metrics_string += "\n\n--- SINGLE ---\n"

    for wave in ['p','qrs','t']:
        metrics_string += "\n######### {} wave #########".format(wave.upper())
        metrics_string += "\n"
        metrics_string += "\nPrecision:    {}%".format(np.round(src.metrics.precision(sum(metrics_single[wave]['truepositives'].values()),sum(metrics_single[wave]['falsepositives'].values()),sum(metrics_single[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\nRecall:       {}%".format(np.round(src.metrics.recall(sum(metrics_single[wave]['truepositives'].values()),sum(metrics_single[wave]['falsepositives'].values()),sum(metrics_single[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\nF1 score:     {}%".format(np.round(src.metrics.f1_score(sum(metrics_single[wave]['truepositives'].values()),sum(metrics_single[wave]['falsepositives'].values()),sum(metrics_single[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\n"
        metrics_string += "\nOnset Error:  {} ± {} ms".format(np.round(np.mean([v for l in metrics_single[wave]['onerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in metrics_single[wave]['onerrors'].values() for v in l])/250*1000,decimals=2))
        metrics_string += "\nOffset Error: {} ± {} ms".format(np.round(np.mean([v for l in metrics_single[wave]['offerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in metrics_single[wave]['offerrors'].values() for v in l])/250*1000,decimals=2))
        metrics_string += "\n\n"
        
    metrics_string += "\n---"
    metrics_string += "\n\n--- MULTI ---\n"


    for wave in ['p','qrs','t']:
        metrics_string += "\n######### {} wave #########".format(wave.upper())
        metrics_string += "\n"
        metrics_string += "\nPrecision:    {}%".format(np.round(src.metrics.precision(sum(metrics_multi[wave]['truepositives'].values()),sum(metrics_multi[wave]['falsepositives'].values()),sum(metrics_multi[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\nRecall:       {}%".format(np.round(src.metrics.recall(sum(metrics_multi[wave]['truepositives'].values()),sum(metrics_multi[wave]['falsepositives'].values()),sum(metrics_multi[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\nF1 score:     {}%".format(np.round(src.metrics.f1_score(sum(metrics_multi[wave]['truepositives'].values()),sum(metrics_multi[wave]['falsepositives'].values()),sum(metrics_multi[wave]['falsenegatives'].values()))*100,decimals=2))
        metrics_string += "\n"
        metrics_string += "\nOnset Error:  {} ± {} ms".format(np.round(np.mean([v for l in metrics_multi[wave]['onerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in metrics_multi[wave]['onerrors'].values() for v in l])/250*1000,decimals=2))
        metrics_string += "\nOffset Error: {} ± {} ms".format(np.round(np.mean([v for l in metrics_multi[wave]['offerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in metrics_multi[wave]['offerrors'].values() for v in l])/250*1000,decimals=2))
        metrics_string += "\n\n"
        
    #########################################################################
    # Produce per-fold metrics, single lead
    metrics_single_fold_1 = {wave: {m : {} for m in metrics_single[wave]} for wave in ['p','qrs','t']}
    metrics_single_fold_2 = {wave: {m : {} for m in metrics_single[wave]} for wave in ['p','qrs','t']}
    metrics_single_fold_3 = {wave: {m : {} for m in metrics_single[wave]} for wave in ['p','qrs','t']}
    metrics_single_fold_4 = {wave: {m : {} for m in metrics_single[wave]} for wave in ['p','qrs','t']}
    metrics_single_fold_5 = {wave: {m : {} for m in metrics_single[wave]} for wave in ['p','qrs','t']}
    metrics_multi_fold_1 = {wave: {m : {} for m in metrics_multi[wave]} for wave in ['p','qrs','t']}
    metrics_multi_fold_2 = {wave: {m : {} for m in metrics_multi[wave]} for wave in ['p','qrs','t']}
    metrics_multi_fold_3 = {wave: {m : {} for m in metrics_multi[wave]} for wave in ['p','qrs','t']}
    metrics_multi_fold_4 = {wave: {m : {} for m in metrics_multi[wave]} for wave in ['p','qrs','t']}
    metrics_multi_fold_5 = {wave: {m : {} for m in metrics_multi[wave]} for wave in ['p','qrs','t']}
    metrics_fold_string = ""

    for wave in ['p','qrs','t']:
        for m in metrics_single[wave]:
            for k in metrics_single[wave][m]:
                u_id = k.split('_')[0]
                m_fold = eval("metrics_single_{}".format(fold_of_file[u_id]))
                m_fold[wave][m][k] = metrics_single[wave][m][k]

    metrics_fold_string += "\n# {}".format(model_name)

    metrics_fold_string += "\n\n--- SINGLE ---\n"

    for wave in ['p','qrs','t']:
        print_header    = "                        "
        print_precision = "Precision:              "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_recall    = "Recall:                 "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_f1_score  = "F1 score:               "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_on_error  = "Onset error:  "#{:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms"
        print_off_error = "Offset error: "#{:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms"
        for fold in ['fold_1','fold_2','fold_3','fold_4','fold_5']:
            m = eval("metrics_single_{}".format(fold))
            print_header += fold + "             "
            try:
                print_precision += "{:3.4}%             ".format(np.round(src.metrics.precision(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_precision += "{:3.4}%             ".format(0.0)
            try:
                print_recall += "{:3.4}%             ".format(np.round(src.metrics.recall(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_recall += "{:3.4}%             ".format(0.0)
            try:
                print_f1_score += "{:3.4}%             ".format(np.round(src.metrics.f1_score(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_f1_score += "{:3.4}%             ".format(0.0)
            
            print_on_error +=  "{: 3.3} ± {: 3.3} ms   ".format(np.round(np.mean([v for l in m[wave]['onerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in m[wave]['onerrors'].values() for v in l])/250*1000,decimals=2))
            print_off_error += "{: 3.3} ± {: 3.3} ms   ".format(np.round(np.mean([v for l in m[wave]['offerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in m[wave]['offerrors'].values() for v in l])/250*1000,decimals=2))
        
        metrics_fold_string += "\n######### {} wave #########".format(wave.upper())
        metrics_fold_string += "\n"
        metrics_fold_string += "\n" + print_header
        metrics_fold_string += "\n"
        metrics_fold_string += "\n" + print_precision
        metrics_fold_string += "\n" + print_recall
        metrics_fold_string += "\n" + print_f1_score
        metrics_fold_string += "\n" + print_on_error
        metrics_fold_string += "\n" + print_off_error
        metrics_fold_string += "\n"
        metrics_fold_string += "\n"
    metrics_fold_string += "\n---"


    # Produce per-fold metrics, multi-lead
    for wave in ['p','qrs','t']:
        for m in metrics_multi[wave]:
            for k in metrics_multi[wave][m]:
                u_id = k.split('_')[0]
                m_fold = eval("metrics_multi_{}".format(fold_of_file[u_id]))
                m_fold[wave][m][k] = metrics_multi[wave][m][k]

    metrics_fold_string += "\n" + "\n--- MULTI ---\n"

    for wave in ['p','qrs','t']:
        print_header    = "                        "
        print_precision = "Precision:              "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_recall    = "Recall:                 "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_f1_score  = "F1 score:               "#{}%\t{}%\t{}%\t{}%\t{}%"
        print_on_error  = "Onset error:  "#{:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms"
        print_off_error = "Offset error: "#{:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms   {:>3.2f} ± {:>3.2f} ms"
        for fold in ['fold_1','fold_2','fold_3','fold_4','fold_5']:
            m = eval("metrics_multi_{}".format(fold))
            print_header += fold + "             "
            try:
                print_precision += "{:3.4}%             ".format(np.round(src.metrics.precision(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_precision += "{:3.4}%             ".format(0.0)
            try:
                print_recall += "{:3.4}%             ".format(np.round(src.metrics.recall(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_recall += "{:3.4}%             ".format(0.0)
            try:
                print_f1_score += "{:3.4}%             ".format(np.round(src.metrics.f1_score(sum(m[wave]['truepositives'].values()),sum(m[wave]['falsepositives'].values()),sum(m[wave]['falsenegatives'].values()))*100,decimals=2))
            except:
                print_f1_score += "{:3.4}%             ".format(0.0)
            
            print_on_error +=  "{: 3.3} ± {: 3.3} ms   ".format(np.round(np.mean([v for l in m[wave]['onerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in m[wave]['onerrors'].values() for v in l])/250*1000,decimals=2))
            print_off_error += "{: 3.3} ± {: 3.3} ms   ".format(np.round(np.mean([v for l in m[wave]['offerrors'].values() for v in l])/250*1000,decimals=2),np.round(np.std([v for l in m[wave]['offerrors'].values() for v in l])/250*1000,decimals=2))
        
        metrics_fold_string += "\n" + "######### {} wave #########".format(wave.upper())
        metrics_fold_string += "\n" + ""
        metrics_fold_string += "\n" + print_header
        metrics_fold_string += "\n" + ""
        metrics_fold_string += "\n" + print_precision
        metrics_fold_string += "\n" + print_recall
        metrics_fold_string += "\n" + print_f1_score
        metrics_fold_string += "\n" + print_on_error
        metrics_fold_string += "\n" + print_off_error
        metrics_fold_string += "\n" + ""
        metrics_fold_string += "\n" + ""
    metrics_fold_string += "\n" + "---"

    #########################################################################
    # Save predictions
    sak.save_data(pon,    os.path.join(basedir,'TrainedModels',model_name,'predicted_pon.csv'))
    sak.save_data(poff,   os.path.join(basedir,'TrainedModels',model_name,'predicted_poff.csv'))
    sak.save_data(qrson,  os.path.join(basedir,'TrainedModels',model_name,'predicted_qrson.csv'))
    sak.save_data(qrsoff, os.path.join(basedir,'TrainedModels',model_name,'predicted_qrsoff.csv'))
    sak.save_data(ton,    os.path.join(basedir,'TrainedModels',model_name,'predicted_ton.csv'))
    sak.save_data(toff,   os.path.join(basedir,'TrainedModels',model_name,'predicted_toff.csv'))

    # Save produced metrics
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(os.path.join(basedir,'TrainedModels',model_name,'metrics_string.txt'), 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(metrics_string)

    with open(os.path.join(basedir,'TrainedModels',model_name,'metrics_fold_string.txt'), 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(metrics_fold_string)
    sys.stdout = original_stdout # Reset the standard output to its original value



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--basedir",     type=str,  required=True,        help="folder containing the databases and the trained models")
    parser.add_argument("--model_name",  type=str,  required=True,        help="model name")
    parser.add_argument("--model_type",  type=str,  default='model_best', help="model checkpoint type ∈ ['model_best', 'checkpoint'] name")
    parser.add_argument("--batch_size",  type=int,  default=16,           help="number of elements to be predicted at the same time")
    parser.add_argument("--window_size", type=int,  default=2048,         help="samples in a window for skimage's view_as_windows")
    parser.add_argument("--hpc",         type=bool, default=False,        help="mark if executed in HPC")
    args = parser.parse_args()

    main(args.basedir, args.model_name, args.hpc, args.model_type, args.batch_size, args.window_size)

