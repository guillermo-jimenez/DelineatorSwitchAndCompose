

if __name__ == '__main__':

k1,i = name.split('-')
i = int(i)

signal_k1 = np.copy(dataset[k1].values[validity[k1][0]:validity[k1][1]])

for wave in ['P', 'QRS', 'T']:
    # Retrieve waves
    wave_on = eval("{}on".format(wave))
    wave_off = eval("{}off".format(wave))
    
    # Retrieve fundamental
    if i >= len(wave_on[k1]): continue
    fundamental = dataset[k1][wave_on[k1][i]:wave_off[k1][i]].values
    fundamental = sak.data.ball_scaling(sak.signal.on_off_correction(fundamental), metric=sak.signal.abs_max)
    
    for j,k2 in enumerate(tqdm.tqdm(dataset,total=dataset.shape[1])):
        if k2 not in wave_on: continue
        # Retrieve data
        target_onset,target_offset = src.metrics.filter_valid(wave_on[k2],wave_off[k2],validity[k2][0],validity[k2][1])-validity[k2][0]
        
        # Retrieve signal
        signal = np.copy(dataset[k2].values[validity[k2][0]:validity[k2][1]])
            
        # Return windowed view
        windowed_k2 = skimage.util.view_as_windows(signal,fundamental.size)

        # Compute correlations
        correlations = []
        for w in windowed_k2:
            w = sak.data.ball_scaling(sak.signal.on_off_correction(w),metric=sak.signal.abs_max)
            c,l = sak.signal.xcorr(fundamental,w)
            correlations.append(c[l == 0])

        # Predict mask
        mask = np.array(correlations) > threshold
        mask = cv2.morphologyEx(mask.astype(float), cv2.MORPH_CLOSE, np.ones((11,))).squeeze().astype(bool)
        input_onset = []
        input_offset = []
        for on,off in zip(*sak.signal.get_mask_boundary(mask)):
            if on!=off:
                input_onset.append(on+np.argmax(correlations[on:off]))
                input_offset.append(on+np.argmax(correlations[on:off])+fundamental.size)
        input_onset = np.array(input_onset)
        input_offset = np.array(input_offset)
        if len(input_onset) != 0:
            _,_,_,_,on,off = src.metrics.compute_metrics(input_onset,input_offset,target_onset,target_offset)
        else:
            on = np.array([])
            off = np.array([])
        
        pd.DataFrame({"onset": on,"offset": off}).to_csv("./{}-{}-{}.{}".format(k1,i,k2,wave))
