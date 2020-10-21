from argparse import ArgumentParser

import os
import os.path
import skimage
import pathlib
import numpy as np
import scipy as sp
import scipy.signal
import pandas as pd
import tqdm

import cv2

import src.data
import src.metrics
import sak


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD INPUTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("LOAD INPUTS")
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str, required=True, help="location of data files")
    parser.add_argument("--outdir", type=str, required=True, help="location of output files")
    parser.add_argument("--signal_id",  type=int, required=True, help="specific signal to process")
    args = parser.parse_args()

    # Store as simple variables
    basedir = os.path.expanduser(args.basedir) # '/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/'
    outdir = os.path.expanduser(args.outdir) # '/media/guille/DADES/DADES/PhysioNet/QTDB/manual0/bias/'
    signal_id = args.signal_id # Variable, HPC array

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("LOAD DATASET")
    # Dataset
    dataset = pd.read_csv(os.path.join(basedir,'Dataset.csv'), index_col=0)
    dataset = dataset.sort_index(axis=1)

    # Validity
    try:
        validity = sak.load_data(os.path.join(basedir,'validity.csv'))
        appendage = "New"
        # Convert to old-style format
        val_tmp = {}
        for k in validity:
            val_tmp[k] = {}
            val_tmp[k]["on"] = [validity[k][0]]
            val_tmp[k]["off"] = [validity[k][1]]
        for k in ['sel35','sel36','sel103','sel232','sel310']:
            val_tmp[k+'_0'] = {}
            val_tmp[k+'_1'] = {}
            val_tmp[k+'_0']["on"] = []
            val_tmp[k+'_1']["on"] = []
            val_tmp[k+'_0']["off"] = []
            val_tmp[k+'_1']["off"] = []
        validity = pd.DataFrame(val_tmp)
            
    except FileNotFoundError:
        validity = pd.read_csv(os.path.join(basedir,'Validity.csv'), index_col=0, 
                            converters={"on": eval, "off": eval}).T
        appendage = ""
        
    # Fiducials
    Pon    = sak.load_data(os.path.join(basedir,'Pon{}.csv'.format(appendage)))
    Poff   = sak.load_data(os.path.join(basedir,'Poff{}.csv'.format(appendage)))
    QRSon  = sak.load_data(os.path.join(basedir,'QRSon{}.csv'.format(appendage)))
    QRSoff = sak.load_data(os.path.join(basedir,'QRSoff{}.csv'.format(appendage)))
    Ton    = sak.load_data(os.path.join(basedir,'Ton{}.csv'.format(appendage)))
    Toff   = sak.load_data(os.path.join(basedir,'Toff{}.csv'.format(appendage)))

    # Add all to validity
    for k in QRSon:
        if k not in validity:
            validity = validity.to_dict()
            validity[k] = {"on": [], "off": []}
            validity = pd.DataFrame(validity)
            
    # Exclude validity sections smaller than N = 2048
    for k in validity:
        exclude = []
        for i,(on,off) in enumerate(zip(*validity[k])):
            if off-on < 2048:
                exclude.append(i)
        # Sort list
        exclude = np.sort(exclude)[::-1]
        if len(exclude) > 0:
            for pos in exclude:
                validity = validity.to_dict()
                validity[k]["on"].pop(pos)
                validity[k]["off"].pop(pos)
                validity = pd.DataFrame(validity)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NORMALIZE DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("NORMALIZE DATASET")
    for k in tqdm.tqdm(list(dataset)):
        if (len(validity[k]['on']) == 0) or (len(validity[k]['off']) == 0):
            continue
        # Copy signal just in case
        signal = np.copy(dataset[k].values)
        
        # Filter signal
        signal = sp.signal.filtfilt(*sp.signal.butter(4,   0.5/250., 'high'),signal.T).T
        signal = sp.signal.filtfilt(*sp.signal.butter(4, 125.0/250.,  'low'),signal.T).T

        # Normalize and pad signal for inputing in algorithm
        normalization = np.median(sak.signal.moving_lambda(signal,256,sak.signal.abs_max))
        signal = signal/normalization
        
        # Store in dataset
        dataset[k] = signal

    # ~~~~~~~~~~~~~~~~~~~~~ RETRIEVE AVAILABLE SIGNAL NAMES ~~~~~~~~~~~~~~~~~~~~
    print("RETRIEVE AVAILABLE SIGNAL NAMES")
    sig_names = []

    for k in QRSon:
        max_n = max([len(Pon[k]),len(QRSon[k]),len(Ton[k])])
        for i in range(max_n):
            sig_names.append("{}-{}".format(k,i))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPUTE BIAS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("COMPUTE BIAS")
    # Retrieve 
    k1,i = sig_names[signal_id].split('-')
    i = int(i)

    for wave in ['P', 'QRS', 'T']:
        # Retrieve waves
        wave_on = eval("{}on".format(wave))
        wave_off = eval("{}off".format(wave))
        
        # Retrieve fundamental
        if i >= len(wave_on[k1]): continue
        fundamental = dataset[k1][wave_on[k1][i]:wave_off[k1][i]].values
        fundamental = sak.signal.on_off_correction(fundamental)
        
        for k2 in tqdm.tqdm(dataset,total=dataset.shape[1]):
            # if the specific wave has nothing for the key, pass
            if k2 not in wave_on: continue

            # Define output structures
            onsets95 = []
            offsets95 = []
            onsets99 = []
            offsets99 = []
            correlations = []

            for valid_on,valid_off in zip(*validity[k2]):
                # Retrieve data
                target_onset,target_offset = src.metrics.filter_valid(wave_on[k2],wave_off[k2],valid_on,valid_off)
                target_onset -= valid_on
                target_offset -= valid_on
                
                # Skip if no available target delineations
                if target_onset.size == 0: continue
                
                # Retrieve signal
                signal = np.copy(dataset[k2].values[valid_on:valid_off])
                    
                # Obtain windowing
                windowed_k2 = skimage.util.view_as_windows(signal,fundamental.size)
                filt = np.zeros((windowed_k2.shape[0],),dtype=bool)
                for on,off in zip(target_onset-(fundamental.size+1),target_offset+(fundamental.size-1)):
                    filt[on:off] = True

                # Compute correlations
                corrs = np.zeros((windowed_k2.shape[0],))
                for j,w in enumerate(windowed_k2):
                    if not filt[j]: continue
                    # Correct deviations w.r.t zero
                    w = sak.signal.on_off_correction(w)
                    c,_ = sak.signal.xcorr(fundamental,w,maxlags=0)
                    corrs[j] = c

                # Predict mask - threshold 95%
                mask95 = np.array(corrs) > 0.95
                mask95 = cv2.morphologyEx(mask95.astype(float), cv2.MORPH_CLOSE, np.ones((11,))).squeeze().astype(bool)
                input_onset95 = []
                input_offset95 = []
                for on,off in zip(*sak.signal.get_mask_boundary(mask95)):
                    if on!=off:
                        input_onset95.append(on+np.argmax(corrs[on:off]))
                        input_offset95.append(on+np.argmax(corrs[on:off])+fundamental.size)
                input_onset95 = np.array(input_onset95,dtype=int)
                input_offset95 = np.array(input_offset95,dtype=int)
                
                # Predict mask - threshold 99%
                mask99 = np.array(corrs) > 0.99
                mask99 = cv2.morphologyEx(mask99.astype(float), cv2.MORPH_CLOSE, np.ones((11,))).squeeze().astype(bool)
                input_onset99 = []
                input_offset99 = []
                for on,off in zip(*sak.signal.get_mask_boundary(mask99)):
                    if on!=off:
                        input_onset99.append(on+np.argmax(corrs[on:off]))
                        input_offset99.append(on+np.argmax(corrs[on:off])+fundamental.size)
                input_onset99 = np.array(input_onset99,dtype=int)
                input_offset99 = np.array(input_offset99,dtype=int)
                
                # Obtain the onsets and offses for the different correlations
                _,_,_,_,on95,off95 = src.metrics.compute_metrics(input_onset95,input_offset95,target_onset,target_offset)
                _,_,_,_,on99,off99 = src.metrics.compute_metrics(input_onset99,input_offset99,target_onset,target_offset)

                # Add to current structures
                correlations += corrs.tolist()
                onsets95 += on95
                offsets95 += off95
                onsets99 += on99
                offsets99 += off99

            # Save files
            out_dir = os.path.join(outdir,k1,str(i),"{}-{}.csv".format(k2,wave))
            pathlib.Path(os.path.split(out_dir)[0]).mkdir(parents=True, exist_ok=True) # Make necessary dirs
            sak.save_data({"onsets95": onsets95,"offsets95": offsets95, "onsets99": onsets95,"offsets99": offsets99}, out_dir)
