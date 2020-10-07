from typing import List
import numpy as np


def dice_score(input: np.ndarray, target: np.ndarray) -> float:
    intersection = (input * target).sum()
    union = input.sum() + target.sum()
    return 2.*intersection/(union + np.finfo('double').eps)


def filter_valid(onset: np.ndarray, offset: np.ndarray, validity_on: int = 0, validity_off: int = np.inf):
    validity_on  = np.array( validity_on)[np.newaxis,np.newaxis]
    validity_off = np.array(validity_off)[np.newaxis,np.newaxis]

    mask_on    = (onset  >= validity_on) & (onset  <= validity_off)
    mask_off   = (offset >= validity_on) & (offset <= validity_off)
    mask_total = np.any(mask_on & mask_off, axis=0) # beat has to be found in every one

    onset = onset[mask_total]
    offset = offset[mask_total]

    return onset, offset


def correspondence(input_onsets, input_offsets, target_onsets, target_offsets):
    filtA =  (target_onsets <=   input_onsets[:,np.newaxis]) & (  input_onsets[:,np.newaxis] <= target_offsets)
    filtB =  (target_onsets <=  input_offsets[:,np.newaxis]) & ( input_offsets[:,np.newaxis] <= target_offsets)
    filtC = (( input_onsets <=  target_onsets[:,np.newaxis]) & ( target_onsets[:,np.newaxis] <=  input_offsets)).T
    filtD = (( input_onsets <= target_offsets[:,np.newaxis]) & (target_offsets[:,np.newaxis] <=  input_offsets)).T

    filter = filtA | filtB | filtC | filtD

    return filter


def cross_correspondence(input_onsets_A, input_offsets_A, input_onsets_B, input_offsets_B):
    filtA =  (input_onsets_A <=  input_onsets_B[:,None]) &  (input_onsets_B[:,None] <= input_offsets_A)
    filtB =  (input_onsets_A <= input_offsets_B[:,None]) & (input_offsets_B[:,None] <= input_offsets_A)
    filtC = ((input_onsets_B <=  input_onsets_A[:,None]) &  (input_onsets_A[:,None] <= input_offsets_B)).T
    filtD = ((input_onsets_B <= input_offsets_A[:,None]) & (input_offsets_A[:,None] <= input_offsets_B)).T

    filter = filtA | filtB | filtC | filtD

    return filter


def post_processing(input_onsets: np.ndarray,input_offsets: np.ndarray,
                    target_onsets: np.ndarray,target_offsets: np.ndarray,
                    validity_on: int,validity_off: int):
    input_onsets,input_offsets = filter_valid(input_onsets,input_offsets,validity_on,validity_off)
    target_onsets,target_offsets = filter_valid(target_onsets,target_offsets,validity_on,validity_off)
    
    return input_onsets,input_offsets,target_onsets,target_offsets


def compute_metrics(input_onsets: np.ndarray, input_offsets: np.ndarray, 
                    target_onsets: np.ndarray, target_offsets: np.ndarray,
                    return_not_chosen: bool = False):
    # Init output
    tp   = 0
    fp   = 0
    fn   = 0
    dice = 0
    onset_error  = []
    offset_error = []

    # Find correspondence between fiducials
    filter = correspondence(input_onsets, input_offsets, target_onsets, target_offsets)

    # Check correspondence of GT beats to detected beats
    corr  = dict()
    
    # Account for already detected beats to calculate false positives
    chosen = np.zeros((filter.shape[0],), dtype=bool)
    for i,column in enumerate(filter.T):
        corr[i] = np.where(column)[0]
        chosen = chosen | column

    # Retrieve beats detected that do not correspond to any GT beat (potential false positives)
    not_chosen = np.where(np.logical_not(chosen))[0]
    
    # Compute Dice coefficient
    mask_input  = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    mask_target = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    for (onset,offset) in zip(input_onsets,input_offsets):
        mask_input[onset:offset] = True
    for (onset,offset) in zip(target_onsets,target_offsets):
        mask_target[onset:offset] = True
    dice = dice_score(mask_input, mask_target)

    # Compute metrics - Fusion strategy of results of both leads, following Martinez et al.
    for i in range(filter.shape[1]):
        # If any GT beat has a correspondence to any segmented beat, true positive + accounts for on/offset error
        if len(corr[i]) != 0:
            # Mark beat as true positive
            tp += 1
            
            # Compute the onset-offset errors
            onset_error.append(int(target_onsets[i]  - input_onsets[corr[i]]))
            offset_error.append(int(target_offsets[i] - input_offsets[corr[i]]))
            
        # If any GT beat has a correspondence to more than one segmented beat, 
        #     the rest of the pairs have to be false positives (Martinez et al.)
        if len(corr[i]) > 1:
            fp += len(corr[i]) - 1
        
        # If any GT beat has no correspondence to any segmented beat, false negative
        if len(corr[i]) == 0:
            fn += 1
            
    # False positives will correspond to those existing in the results that do not correspond to any beat in the GT (the not chosen)
    fp += len(not_chosen)
    
    if return_not_chosen: return tp,fp,fn,dice,onset_error,offset_error,not_chosen
    else:                 return tp,fp,fn,dice,onset_error,offset_error
        



def compute_QTDB_metrics(input_onsets_0: np.ndarray, input_offsets_0: np.ndarray, 
                         input_onsets_1: np.ndarray, input_offsets_1: np.ndarray, 
                         target_onsets: np.ndarray, target_offsets: np.ndarray,
                         return_not_chosen: bool = False):
    # Init output
    tp   = 0
    fp   = 0
    fn   = 0
    dice = 0
    onset_error  = []
    offset_error = []

    # If the prediction is different for every lead (single-lead strategy)
    filter_0 = correspondence(input_onsets_0, input_offsets_0, target_onsets, target_offsets)
    filter_1 = correspondence(input_onsets_1, input_offsets_1, target_onsets, target_offsets)
    filt_corr = cross_correspondence(input_onsets_0,input_offsets_0,input_onsets_1,input_offsets_1)

    # Check correspondence of GT beats to detected beats
    corr  = dict()
    
    # Account for already detected beats to calculate false positives
    chosen_0 = np.zeros((filter_0.shape[0],), dtype=bool)
    chosen_1 = np.zeros((filter_1.shape[0],), dtype=bool)
    for i,(col0,col1) in enumerate(zip(filter_0.T,filter_1.T)):
        corr[i]  = [np.where(col0)[0], 
                    np.where(col1)[0]]
        chosen_0 = chosen_0 | col0
        chosen_1 = chosen_1 | col1
        
    # Retrieve beats detected that do not correspond to any GT beat (potential false positives)
    not_chosen_0 = np.where(np.logical_not(chosen_0))[0]
    not_chosen_1 = np.where(np.logical_not(chosen_1))[0]

    # Compute Dice coefficient - lead 0
    mask_input  = np.zeros((np.max(np.hstack((input_offsets_0,target_offsets)))+10,),dtype=bool)
    mask_target = np.zeros((np.max(np.hstack((input_offsets_0,target_offsets)))+10,),dtype=bool)
    for (onset,offset) in zip(input_onsets_0,input_offsets_0):
        mask_input[onset:offset] = True
    for (onset,offset) in zip(target_onsets,target_offsets):
        mask_target[onset:offset] = True
    dice += dice_score(mask_input, mask_target)
    # Compute Dice coefficient - lead 1
    mask_input  = np.zeros((np.max(np.hstack((input_offsets_1,target_offsets)))+10,),dtype=bool)
    mask_target = np.zeros((np.max(np.hstack((input_offsets_1,target_offsets)))+10,),dtype=bool)
    for (onset,offset) in zip(input_onsets_1,input_offsets_1):
        mask_input[onset:offset] = True
    for (onset,offset) in zip(target_onsets,target_offsets):
        mask_target[onset:offset] = True
    dice += dice_score(mask_input, mask_target)
    dice /= 2 # Halve each

    # Compute metrics - Fusion strategy of results of both leads, following Martinez et al.
    for i in range(filter_0.shape[1]):
        # If any GT beat has a correspondence to any segmented beat, true positive + accounts for on/offset error
        if (len(corr[i][0]) != 0) or (len(corr[i][1]) != 0):
            # Mark beat as true positive
            tp += 1
            
            # To compute the onset-offset errors, check which is the lead with the least error to the GT (Martinez et al.)
            if len(corr[i][0]) != 0:
                onset_0  = (target_onsets[i] - input_onsets_0[corr[i][0]])
                offset_0 = (target_offsets[i] - input_offsets_0[corr[i][0]])
            else:
                onset_0  = np.asarray([np.inf])
                offset_0 = np.asarray([np.inf])
            if len(corr[i][1]) != 0:
                onset_1  = (target_onsets[i] - input_onsets_1[corr[i][1]])
                offset_1 = (target_offsets[i] - input_offsets_1[corr[i][1]])
            else:
                onset_1  = np.asarray([np.inf])
                offset_1 = np.asarray([np.inf])

            # Concatenate errors in one error vector
            tmp_onerror  = np.hstack((onset_0,onset_1))
            tmp_offerror = np.hstack((offset_0,offset_1))
            
            # Onset/offset Error as the value resulting in the minimum absolute value
            onset_error  += [int(tmp_onerror[np.argmin(np.abs(tmp_onerror))])]
            offset_error += [int(tmp_offerror[np.argmin(np.abs(tmp_offerror))])]
            
        # If any GT beat has a correspondence to more than one segmented beat, 
        #     the rest of the pairs have to be false positives (Martinez et al.)
        if (len(corr[i][0]) > 1) and (len(corr[i][1]) > 1):
            fp += min([len(corr[i][0]), len(corr[i][1])]) - 1
        
        # If any GT beat has no correspondence to any segmented beat, false negative
        if (len(corr[i][0]) == 0) and (len(corr[i][1]) == 0):
            fn += 1
            
    # False positives will correspond to those existing in the results that do not correspond to any beat in the GT (the not chosen)
    fp += len(not_chosen_0) + len(not_chosen_1) - (filt_corr[not_chosen_1,:][:,not_chosen_0]).sum()
        
    # Return all
    return tp,fp,fn,dice,onset_error,offset_error





def compute_multilead_metrics(input_onsets: List[np.ndarray], input_offsets: List[np.ndarray], 
                              target_onsets: List[np.ndarray], target_offsets: List[np.ndarray]):
    # Init output
    truepositive = 0
    falsepositive = 0
    falsenegative = 0
    dice = []
    onset_error = []
    offset_error = []

    leads = len(input_onsets)

    # Find correspondence between different leads
    filters_corr = [correspondence(input_onsets[i],input_offsets[i],input_onsets[j],input_offsets[j]) for i in range(leads) for j in range(leads) if i != j]

    # For each lead, compute metrics
    not_chosen = []
    for i, (i_on, i_off, t_on, t_off) in enumerate(zip(input_onsets,input_offsets,target_onsets,target_offsets)):
        tp,fp,fn,dc,one,offe,notchsn = compute_metrics(i_on, i_off, t_on, t_off, return_not_chosen=True)
        truepositive += tp
        falsepositive += fp
        falsenegative += fn
        dice.append(dc)
        onset_error.append(one)
        offset_error.append(offe)
        not_chosen.append(notchsn)

    return falsepositive,filters_corr,not_chosen

    # # Compute Dice coefficients
    # mask_input  = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    # mask_target = np.zeros((np.max(np.hstack((input_offsets,target_offsets)))+10,),dtype=bool)
    # for (onset,offset) in zip(input_onsets,input_offsets):
    #     mask_input[onset:offset] = True
    # for (onset,offset) in zip(target_onsets,target_offsets):
    #     mask_target[onset:offset] = True
    # dice = dice_score(mask_input, mask_target)

    # # Compute metrics - Fusion strategy of results of both leads, following Martinez et al.
    # for i in range(filter.shape[1]):
    #     # If any GT beat has a correspondence to any segmented beat, true positive + accounts for on/offset error
    #     if len(corr[i]) != 0:
    #         # Mark beat as true positive
    #         tp += 1
            
    #         # Compute the onset-offset errors
    #         onset_error.append(int(target_onsets[corr[i]]  - input_onsets[i]))
    #         offset_error.append(int(target_offsets[corr[i]] - input_offsets[i]))
            
    #     # If any GT beat has a correspondence to more than one segmented beat, 
    #     #     the rest of the pairs have to be false positives (Martinez et al.)
    #     if len(corr[i]) > 1:
    #         fn += len(corr[i]) - 1
        
    #     # If any GT beat has no correspondence to any segmented beat, false negative
    #     if len(corr[i]) == 0:
    #         fp += 1
            
    # # False positives will correspond to those existing in the results that do not correspond to any beat in the GT (the not chosen)
    # fn += len(not_chosen)
    
    # return tp,fp,fn,dice,onset_error,offset_error
        

def precision(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fp)

def recall(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fn)

def f1_score(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+(fp+fn)/2)

