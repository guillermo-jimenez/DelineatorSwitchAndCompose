import cv2
import math
import tqdm
import pandas
import numpy as np
from utils.data_structures import series_to_supervised
from utils.data_structures import supervised_to_series



def predict(model, config, data, execinfo, results, results_CV2, retrain=False):
    # Load best model state
    # if (execinfo.state != None) and (model != None): backend.load_state(model, execinfo)

    # Predict evaluation on test set
    for key in tqdm.tqdm(execinfo.test, ascii=True, desc="Test "):
        key_0 = key + '_0'
        key_1 = key + '_1'

        if config.in_ch == 1: # Single lead
            if retrain or (key_0 not in results.keys) or (key_0 not in results_CV2.keys) or (key_1 not in results.keys) or (key_1 not in results_CV2.keys):
                predict_single(model, config, data, key_0, results, results_CV2)
                predict_single(model, config, data, key_1, results, results_CV2)

        elif config.in_ch == 2: # Multi lead
            if retrain or (key_0 not in results.keys) or (key_0 not in results_CV2.keys) or (key_1 not in results.keys) or (key_1 not in results_CV2.keys):
                predict_multi(model, config, data, key_0, key_1, results, results_CV2)

        else:
            raise NotImplementedError("No batch generation strategy for " + str(config.in_ch) + " channels has been devised")


def predict_single(model, config, data, key, results, results_CV2):
    # Step 1: Convert from series to supervised (with a finer stride for improved results in the window boundaries):
    series = series_to_supervised(data.dataset[key], config.window, config.window//8)

    # Step 2: Predict:
    seg = np.zeros((series.shape[0],series.shape[1],config.out_ch))
    sub_batches = math.ceil(series.shape[0]/(config.batch_size))
    
    # Avoid GPU RAM management problems
    for b in range(sub_batches):
        batch = series[b*config.batch_size:(b+1)*config.batch_size,...]
        batch = batch[...,np.newaxis]
        seg[b*config.batch_size:(b+1)*config.batch_size,...] = model.predict(batch)

        del batch

    # Step 3: Convert from supervised to series:
    lead_P   = supervised_to_series(seg[...,0].squeeze(), config.window, config.window//8)[:config.max_size].round()
    lead_QRS = supervised_to_series(seg[...,1].squeeze(), config.window, config.window//8)[:config.max_size].round()
    lead_T   = supervised_to_series(seg[...,2].squeeze(), config.window, config.window//8)[:config.max_size].round()
    
    # Step 4: Store in the output matrices:
    results.P.wave[key]   = lead_P
    results.QRS.wave[key] = lead_QRS
    results.T.wave[key]   = lead_T

    # Step 5.1: Apply morphological closing
    lead_P   = cv2.morphologyEx(lead_P,   cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations
    lead_QRS = cv2.morphologyEx(lead_QRS, cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations
    lead_T   = cv2.morphologyEx(lead_T,   cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations

    lead_P   = cv2.morphologyEx(lead_P,   cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations
    lead_QRS = cv2.morphologyEx(lead_QRS, cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations
    lead_T   = cv2.morphologyEx(lead_T,   cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations

    results_CV2.P.wave[key]   = lead_P
    results_CV2.QRS.wave[key] = lead_QRS
    results_CV2.T.wave[key]   = lead_T

    # Mark as computed
    results.keys.append(key)
    results_CV2.keys.append(key)

    # Step 6: Avoid GPU memory filled with old data
    del series, seg


def predict_multi(model, config, data, key_0, key_1, results, results_CV2):
    # Step 1: Convert from series to supervised (with a finer stride for improved results in the window boundaries):
    series_0 = series_to_supervised(data.dataset[key_0], config.window, config.window//8)
    series_1 = series_to_supervised(data.dataset[key_1], config.window, config.window//8)

    # Combine both series
    series = np.dstack((series_0, series_1)) # Stack on the channels

    # Step 2: Predict:
    seg = np.zeros((series.shape[0],series.shape[1],config.out_ch)) # Segmentation storage
    sub_batches = math.ceil(series.shape[0]/(config.batch_size))    # Number of batches

    # Avoid GPU RAM management problems
    for b in range(sub_batches):
        batch = series[b*config.batch_size:(b+1)*config.batch_size,...]
        # batch = batch[:,:,np.newaxis,:]
        seg[b*config.batch_size:(b+1)*config.batch_size,...] = model.predict(batch).squeeze()

        del batch

    # Step 3: Convert from supervised to series:
    lead_P   = supervised_to_series(seg[...,0].squeeze(), config.window, config.window//8)[:config.max_size].round()
    lead_QRS = supervised_to_series(seg[...,1].squeeze(), config.window, config.window//8)[:config.max_size].round()
    lead_T   = supervised_to_series(seg[...,2].squeeze(), config.window, config.window//8)[:config.max_size].round()
    
    # Step 4: Store in the output matrices:
    results.P.wave[key_0]   = lead_P
    results.QRS.wave[key_0] = lead_QRS
    results.T.wave[key_0]   = lead_T

    results.P.wave[key_1]   = lead_P
    results.QRS.wave[key_1] = lead_QRS
    results.T.wave[key_1]   = lead_T

    # Step 5.1: Apply morphological closing
    lead_P   = cv2.morphologyEx(lead_P,   cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations
    lead_QRS = cv2.morphologyEx(lead_QRS, cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations
    lead_T   = cv2.morphologyEx(lead_T,   cv2.MORPH_CLOSE, np.ones((config.element_size,))) # Close holes in segmentations

    lead_P   = cv2.morphologyEx(lead_P,   cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations
    lead_QRS = cv2.morphologyEx(lead_QRS, cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations
    lead_T   = cv2.morphologyEx(lead_T,   cv2.MORPH_OPEN, np.ones((5,))).squeeze()  # Erosion + Dilation to get rid of noisy activations

    results_CV2.P.wave[key_0]   = lead_P
    results_CV2.QRS.wave[key_0] = lead_QRS
    results_CV2.T.wave[key_0]   = lead_T

    results_CV2.P.wave[key_1]   = lead_P
    results_CV2.QRS.wave[key_1] = lead_QRS
    results_CV2.T.wave[key_1]   = lead_T

    # Mark as computed
    results.keys.append(key_0)
    results.keys.append(key_1)
    results_CV2.keys.append(key_0)
    results_CV2.keys.append(key_1)

    # Step 6: Avoid GPU memory filled with old data
    del series_0, series_1, series, seg


