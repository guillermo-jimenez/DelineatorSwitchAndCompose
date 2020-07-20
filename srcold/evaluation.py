import numpy as np
import os
import pandas
from utils.data_structures import MetricsStorage
from utils.inference import predict
import cv2


def evaluate(model, config, data, execinfo, metrics, metrics_CV2, results, results_CV2, recompute):
    """Wave evaluation"""
    predict(model, config, data, execinfo, results, results_CV2, recompute)

    # IF IT'S NOT THE OVERALL EVALUATION
    leads = np.concatenate((pandas.Index(execinfo.test) + '_0', pandas.Index(execinfo.test) + '_1'))

    retrieve_fiducials(results,     leads, recompute)
    retrieve_fiducials(results_CV2, leads, recompute)
    
    ### COMPUTE METRICS ###
    metrics     = metric_computation(config, data, metrics,     results,     execinfo.test, recompute)
    metrics_CV2 = metric_computation(config, data, metrics_CV2, results_CV2, execinfo.test, recompute)

    ### SAVE RESULTS ###
    path_CV2 = os.path.splitext(execinfo.results)[0] + '_CV2' + os.path.splitext(execinfo.results)[1]
    save_results(metrics,     config, execinfo.test, execinfo.results)
    save_results(metrics_CV2, config, execinfo.test, path_CV2)


def save_results(metrics, config, test, output_path):
    if output_path[-4:].lower() != '.csv': output_path += '.csv'
    res = dict()

    precision_P, recall_P, onset_P, offset_P, dice_P           = wave_evaluation_from_metrics(metrics.P,  test) 
    precision_QRS, recall_QRS, onset_QRS, offset_QRS, dice_QRS = wave_evaluation_from_metrics(metrics.QRS,test) 
    precision_T, recall_T, onset_T, offset_T, dice_T           = wave_evaluation_from_metrics(metrics.T,  test) 

    res['Precision (P+)']          = [precision_P,                                    precision_QRS,                                    precision_T]
    res['Recall (Se%)']            = [recall_P,                                       recall_QRS,                                       recall_T]
    res['Onset error (Mean), ms']  = [onset_P.mean()*(1./config.sampling_freq*1000),  onset_QRS.mean()*(1./config.sampling_freq*1000),  onset_T.mean()*(1./config.sampling_freq*1000)]
    res['Onset error (STD), ms']   = [onset_P.std()*(1./config.sampling_freq*1000),   onset_QRS.std()*(1./config.sampling_freq*1000),   onset_T.std()*(1./config.sampling_freq*1000)]
    res['Offset error (Mean), ms'] = [offset_P.mean()*(1./config.sampling_freq*1000), offset_QRS.mean()*(1./config.sampling_freq*1000), offset_T.mean()*(1./config.sampling_freq*1000)]
    res['Offset error (STD), ms']  = [offset_P.std()*(1./config.sampling_freq*1000),  offset_QRS.std()*(1./config.sampling_freq*1000),  offset_T.std()*(1./config.sampling_freq*1000)]
    res['Dice Coefficient']        = [dice_P,                                         dice_QRS,                                         dice_T]

    res = pandas.DataFrame(res, columns=['Precision (P+)','Recall (Se%)','Onset error (Mean), ms','Onset error (STD), ms','Offset error (Mean), ms','Offset error (STD), ms','Dice Coefficient'], index=['P wave', 'QRS wave', 'T wave'])
    res.to_csv(output_path)


def retrieve_fiducials(results, test, recompute=False):
    wave_fiducials(results.P,   test, recompute)
    wave_fiducials(results.QRS, test, recompute)
    wave_fiducials(results.T,   test, recompute)


def wave_fiducials(wave, test, recompute=False):
    for k in test:
        if recompute or (k not in wave.onset.keys()):
            seg = np.diff(np.pad(wave.wave[k].values,((1,1),),'constant', constant_values=0),axis=-1)

            wave.onset[k]  = np.where(seg ==  1.)[0]
            wave.offset[k] = np.where(seg == -1.)[0] - 1
            wave.peak[k]   = (wave.onset[k] + wave.offset[k])//2


def get_correspondence_between_gt_and_predicted(fiducials_data, fiducials_results, k, validity):
    mask_on    = (fiducials_data.onset[k]  >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_data.onset[k]  <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_peak  = (fiducials_data.peak[k]   >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_data.peak[k]   <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_off   = (fiducials_data.offset[k] >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_data.offset[k] <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_total = np.any(mask_on & mask_peak & mask_off, axis=0) # beat has to be found in every one

    d_on = fiducials_data.onset[k][mask_total]
    d_pk = fiducials_data.peak[k][mask_total]
    d_of = fiducials_data.offset[k][mask_total]

    mask_on    = (fiducials_results.onset[k]  >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_results.onset[k]  <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_peak  = (fiducials_results.peak[k]   >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_results.peak[k]   <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_off   = (fiducials_results.offset[k] >= np.asarray(validity[k][0])[:,np.newaxis]) & (fiducials_results.offset[k] <= np.asarray(validity[k][1])[:,np.newaxis])
    mask_total = np.any(mask_on & mask_peak & mask_off, axis=0) # beat has to be found in every one

    r_on = fiducials_results.onset[k][mask_total]
    r_pk = fiducials_results.peak[k][mask_total]
    r_of = fiducials_results.offset[k][mask_total]

    filtA =  (d_on <= r_on[:,np.newaxis]) & (r_on[:,np.newaxis] <= d_of)
    filtB =  (d_on <= r_pk[:,np.newaxis]) & (r_pk[:,np.newaxis] <= d_of)
    filtC =  (d_on <= r_of[:,np.newaxis]) & (r_of[:,np.newaxis] <= d_of)
    filtD = ((r_on <= d_on[:,np.newaxis]) & (d_on[:,np.newaxis] <= r_of)).T
    filtE = ((r_on <= d_pk[:,np.newaxis]) & (d_pk[:,np.newaxis] <= r_of)).T
    filtF = ((r_on <= d_of[:,np.newaxis]) & (d_of[:,np.newaxis] <= r_of)).T

    filt_all = filtA | filtB | filtC | filtD | filtE | filtF

    return filt_all, r_on, r_of, d_on, d_of


def get_correspondence_between_predicted_leads(fiducials_results, k, validity):
    mask_on    = (fiducials_results.onset[k + '_0']  >= np.asarray(validity[k + '_0'][0])[:,np.newaxis]) & (fiducials_results.onset[k + '_0']  <= np.asarray(validity[k + '_0'][1])[:,np.newaxis])
    mask_peak  = (fiducials_results.peak[k + '_0']   >= np.asarray(validity[k + '_0'][0])[:,np.newaxis]) & (fiducials_results.peak[k + '_0']   <= np.asarray(validity[k + '_0'][1])[:,np.newaxis])
    mask_off   = (fiducials_results.offset[k + '_0'] >= np.asarray(validity[k + '_0'][0])[:,np.newaxis]) & (fiducials_results.offset[k + '_0'] <= np.asarray(validity[k + '_0'][1])[:,np.newaxis])
    mask_total = np.any(mask_on & mask_peak & mask_off, axis=0) # beat has to be found in every one

    res_0_on = fiducials_results.onset[k + '_0'][mask_total]
    res_0_pk = fiducials_results.peak[k + '_0'][mask_total]
    res_0_of = fiducials_results.offset[k + '_0'][mask_total]

    mask_on    = (fiducials_results.onset[k + '_1']  >= np.asarray(validity[k + '_1'][0])[:,np.newaxis]) & (fiducials_results.onset[k + '_1']  <= np.asarray(validity[k + '_1'][1])[:,np.newaxis])
    mask_peak  = (fiducials_results.peak[k + '_1']   >= np.asarray(validity[k + '_1'][0])[:,np.newaxis]) & (fiducials_results.peak[k + '_1']   <= np.asarray(validity[k + '_1'][1])[:,np.newaxis])
    mask_off   = (fiducials_results.offset[k + '_1'] >= np.asarray(validity[k + '_1'][0])[:,np.newaxis]) & (fiducials_results.offset[k + '_1'] <= np.asarray(validity[k + '_1'][1])[:,np.newaxis])
    mask_total = np.any(mask_on & mask_peak & mask_off, axis=0) # beat has to be found in every one

    res_1_on = fiducials_results.onset[k + '_1'][mask_total]
    res_1_pk = fiducials_results.peak[k + '_1'][mask_total]
    res_1_of = fiducials_results.offset[k + '_1'][mask_total]

    filtA =  (res_0_on <= res_1_on[:,np.newaxis]) & (res_1_on[:,np.newaxis] <= res_0_of)
    filtB =  (res_0_on <= res_1_pk[:,np.newaxis]) & (res_1_pk[:,np.newaxis] <= res_0_of)
    filtC =  (res_0_on <= res_1_of[:,np.newaxis]) & (res_1_of[:,np.newaxis] <= res_0_of)
    filtD = ((res_1_on <= res_0_on[:,np.newaxis]) & (res_0_on[:,np.newaxis] <= res_1_of)).T
    filtE = ((res_1_on <= res_0_pk[:,np.newaxis]) & (res_0_pk[:,np.newaxis] <= res_1_of)).T
    filtF = ((res_1_on <= res_0_of[:,np.newaxis]) & (res_0_of[:,np.newaxis] <= res_1_of)).T

    filt_all = filtA | filtB | filtC | filtD | filtE | filtF

    return filt_all


def compute_dice_score(mask_1, mask_2):
    intersection = (mask_1 * mask_2).sum()
    union = mask_1.sum() + mask_2.sum()
    return 2.*intersection/(union + np.finfo('double').eps)


def compute_wave_metrics(config, fiducials_data, fiducials_results, wave_metrics, test, validity, recompute=False):
    for k in test:
        if (k not in wave_metrics.keys) or recompute:
            # If the prediction is different for every lead (single-lead strategy)
            filt_all_0, r_on_0, r_of_0, d_on_0, d_of_0 = get_correspondence_between_gt_and_predicted(fiducials_data, fiducials_results, k + '_0', validity)
            filt_all_1, r_on_1, r_of_1, d_on_1, d_of_1 = get_correspondence_between_gt_and_predicted(fiducials_data, fiducials_results, k + '_1', validity)
            filt_corr = get_correspondence_between_predicted_leads(fiducials_results, k, validity)

            # Check correspondence of GT beats to detected beats
            corr  = dict()
            
            # Account for already detected beats to calculate false positives
            chosen_0 = np.zeros((filt_all_0.shape[0],), dtype=bool)
            chosen_1 = np.zeros((filt_all_1.shape[0],), dtype=bool)

            for i in range(filt_all_0.shape[1]):
                corr[i]  = [np.where(filt_all_0[:,i])[0].tolist(), 
                            np.where(filt_all_1[:,i])[0].tolist()]
                chosen_0 = chosen_0 | filt_all_0[:,i]
                chosen_1 = chosen_1 | filt_all_1[:,i]
                
            # Retrieve beats detected that do not correspond to any GT beat (potential false positives)
            not_chosen_0 = np.where(np.logical_not(chosen_0))[0]
            not_chosen_1 = np.where(np.logical_not(chosen_1))[0]
                
            # Initialize metrics
            wave_metrics.truepositive[k] = 0
            wave_metrics.falsepositive[k] = 0
            wave_metrics.falsenegative[k] = 0
            wave_metrics.dice[k] = 0
            wave_metrics.onseterror[k] = []
            wave_metrics.offseterror[k] = []

            # Compute Dice coefficient
            for i in range(len(validity[k + '_0'][0])):
                on  = validity[k + '_0'][0][i]
                off = validity[k + '_0'][1][i]
                
                wave_metrics.dice[k] += compute_dice_score(fiducials_data.wave[k + '_0'][on:off], fiducials_results.wave[k + '_0'][on:off])
                wave_metrics.dice[k] += compute_dice_score(fiducials_data.wave[k + '_1'][on:off], fiducials_results.wave[k + '_1'][on:off])
                wave_metrics.dice[k] /= 2. # Average both leads
            
            # Compute metrics - Fusion strategy of results of both leads, following Martinez et al.
            for i in range(filt_all_0.shape[1]):
                # If any GT beat has a correspondence to any segmented beat, true positive + accounts for on/offset error
                if (len(corr[i][0]) != 0) or (len(corr[i][1]) != 0):
                    # Mark beat as true positive
                    wave_metrics.truepositive[k] += 1
                    
                    # To compute the onset-offset errors, check which is the lead with the least error to the GT (Martinez et al.)
                    if len(corr[i][0]) != 0:
                        onset_0  = (d_on_0[i] - r_on_0[corr[i][0]])
                        offset_0 = (d_of_0[i] - r_of_0[corr[i][0]])
                    else:
                        onset_0  = np.asarray([np.inf])
                        offset_0 = np.asarray([np.inf])
                    if len(corr[i][1]) != 0:
                        onset_1  = (d_on_1[i] - r_on_1[corr[i][1]])
                        offset_1 = (d_of_1[i] - r_of_1[corr[i][1]])
                    else:
                        onset_1  = np.asarray([np.inf])
                        offset_1 = np.asarray([np.inf])

                    # Concatenate errors in one error vector
                    onset_error  = np.hstack((onset_0,onset_1))
                    offset_error = np.hstack((offset_0,offset_1))
                    
                    # Onset/offset Error as the value resulting in the minimum absolute value
                    wave_metrics.onseterror[k]  += [int(onset_error[np.argmin(np.abs(onset_error))])]
                    wave_metrics.offseterror[k] += [int(offset_error[np.argmin(np.abs(offset_error))])]
                    
                # If any GT beat has a correspondence to more than one segmented beat, 
                #     the rest of the pairs have to be false positives (Martinez et al.)
                if (len(corr[i][0]) > 1) and (len(corr[i][1]) > 1):
                    wave_metrics.falsepositive[k] += min([len(corr[i][0]), len(corr[i][1])]) - 1
                
                # If any GT beat has no correspondence to any segmented beat, false negative
                if (len(corr[i][0]) == 0) and (len(corr[i][1]) == 0):
                    wave_metrics.falsenegative[k] += 1
                    
            # False positives will correspond to those existing in the results that do not correspond to any beat in the GT (the not chosen)
            wave_metrics.falsepositive[k] += len(not_chosen_0) + len(not_chosen_1) - (filt_corr[not_chosen_1,:][:,not_chosen_0]).sum()
                
            # Mark the key as computed
            wave_metrics.keys.append(k)


def wave_evaluation_from_metrics(wave_metrics, test):
    tp        = 0
    fp        = 0
    fn        = 0
    dice      = 0
    on        = []
    of        = []

    for k in test:
        tp   += wave_metrics.truepositive[k]
        fp   += wave_metrics.falsepositive[k]
        fn   += wave_metrics.falsenegative[k]
        on   += wave_metrics.onseterror[k]
        of   += wave_metrics.offseterror[k]
        dice += wave_metrics.dice[k]

    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    on        = np.asarray(on)
    of        = np.asarray(of)
    dice      = dice/len(test)

    return precision, recall, on, of, dice


def metric_computation(config, data, metrics, results, test, recompute=False):
    compute_wave_metrics(config, data.P,   results.P,   metrics.P,   test, data.validity, recompute)
    compute_wave_metrics(config, data.QRS, results.QRS, metrics.QRS, test, data.validity, recompute)
    compute_wave_metrics(config, data.T,   results.T,   metrics.T,   test, data.validity, recompute)
    
    return metrics


