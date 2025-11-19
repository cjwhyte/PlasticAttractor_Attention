"""
Decoding analysis of plastic attractor decision model

@author: Christopher Whyte

"""""

def plasticattractor_decode(num_trials, num_blocks, stim_period, c_label, s_label, r_label, conj, feat):

    import numpy as np
    
    # %% define decoding functions
    
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import KFold
    from imblearn.under_sampling import RandomUnderSampler 
    
    def decode_timeseries_CV(data,labels): # data shape: features x time x trials, blocks, labels shape: samples x 1
        
        classifier = LR()
        time = np.size(data,1)
        
        # reshape data 
        data = data.T
        
        # define undersample strategy
        undersample = RandomUnderSampler(sampling_strategy='majority')
        
        # cross val
        kfolds = 5
        kf = KFold(n_splits=kfolds)
        kf.get_n_splits(data)
        
        k = 0
        accuracy = np.zeros([time,kfolds])
        KFold(n_splits=kfolds, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(data):
             X_train, X_test = data[train_index,:,:], data[test_index,:,:]
             Y_train, Y_test = labels[train_index], labels[test_index]
             for t in range(time):
                # undersample data to balance classes
                X_train_bal, Y_train_bal = undersample.fit_resample(X_train[:,t,:], Y_train)
                X_test_bal, Y_test_bal = undersample.fit_resample(X_test[:,t,:], Y_test)
                # train
                classifier.fit(X_train_bal,Y_train_bal)
                # test on held out data
                pred_labels = classifier.predict(X_test_bal)
                # compute accuracy
                accuracy[t,k] = np.sum(Y_test_bal == pred_labels) / len(Y_test_bal)
             k += 1
        
        accuracy = np.sum(accuracy,axis=1)/np.size(accuracy,1)
        return accuracy
    
    # %% Sort data, generate labels
    
    # rule indexes
    rule1_idx = np.where(r_label == 0)
    rule2_idx = np.where(r_label == 1)
    
    # rule 1
    c_labels_R1 = c_label[rule1_idx]
    s_labels_R1 = s_label[rule1_idx]
    feat_act_R1 = np.squeeze(feat[:,:,rule1_idx])
    conj_act_R1 = np.squeeze(conj[:,:,rule1_idx])
    
    # rule 2
    c_labels_R2 = c_label[rule2_idx]
    s_labels_R2 = s_label[rule2_idx]
    feat_act_R2 = np.squeeze(feat[:,:,rule2_idx])
    conj_act_R2 = np.squeeze(conj[:,:,rule2_idx])
    
    # %% classify
    
    # feature unit decoding
    accuracy_colour_feat = decode_timeseries_CV(feat, c_label)
    accuracy_shape_feat = decode_timeseries_CV(feat, s_label)  
    
    # conjunction unit decoding 
    accuracy_colour_conj = decode_timeseries_CV(conj, c_label)
    accuracy_shape_conj = decode_timeseries_CV(conj, s_label)   
    
    ########### conjunction and feature decoding by rule
    
    ### rule 1
    
    # feature unit decoding
    accuracy_colour_feat_1 = decode_timeseries_CV(feat_act_R1, c_labels_R1)
    accuracy_shape_feat_1 = decode_timeseries_CV(feat_act_R1, s_labels_R1)  
    
    # conjunction unit decoding 
    accuracy_colour_conj_1 = decode_timeseries_CV(conj_act_R1, c_labels_R1)
    accuracy_shape_conj_1 = decode_timeseries_CV(conj_act_R1, s_labels_R1)   
      
    ### rule 2
    
    # feature unit decoding
    accuracy_colour_feat_2 = decode_timeseries_CV(feat_act_R2, c_labels_R2)
    accuracy_shape_feat_2 = decode_timeseries_CV(feat_act_R2, s_labels_R2)  
    
    # conjunction unit decoding 
    accuracy_colour_conj_2 = decode_timeseries_CV(conj_act_R2, c_labels_R2)
    accuracy_shape_conj_2 = decode_timeseries_CV(conj_act_R2, s_labels_R2)   
    
    return accuracy_colour_feat, accuracy_shape_feat, accuracy_colour_conj, accuracy_shape_conj, accuracy_colour_feat_1, accuracy_shape_feat_1,  \
           accuracy_colour_conj_1, accuracy_shape_conj_1, accuracy_colour_feat_2, accuracy_shape_feat_2, accuracy_colour_conj_2, accuracy_shape_conj_2

       