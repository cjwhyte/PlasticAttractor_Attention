"""
Behavioural analysis of plastic attractor decision model

@author: Christopher Whyte

"""""

def plasticattractor_beh(num_trials, num_blocks, acc_dict, rt_dict, c_label_dict, s_label_dict, r_label_dict,conj_dict, feat_dict):

    import numpy as np
    # import matplotlib.pyplot as plt
    
    ############ accuracy and reaction times
    
    rt = np.zeros([num_trials, num_blocks])
    accuracy = np.zeros([num_trials, num_blocks])
    for blk in range(num_blocks):
        # accuracy
        blk_acc = acc_dict[blk][2:]
        accuracy[:,blk] = blk_acc
        # rt
        blk_rt = rt_dict[blk][2:]
        rt[:,blk] = blk_rt
    
    # overall accuracy
    accuracy_overall = sum(accuracy.flatten())/(num_trials*num_blocks)

    # rt (for correct trials only)
    accuracy_flat = accuracy.flatten()
    rt = rt.flatten()
    correct_idx = (accuracy_flat == 1)
    correct_rt = rt[correct_idx]
    correct_rt = np.mean(correct_rt)
    
    
    ########### grab correct trial data
    
    # reshape data for easy indexing
    conj_correct = np.concatenate(np.array([conj_dict[i][:,:,2:] for i in range(num_blocks)]), axis = 2)
    conj_correct = conj_correct[:,:,correct_idx]
    feat_correct = np.concatenate(np.array([feat_dict[i][:4,:,2:] for i in range(num_blocks)]), axis = 2)
    feat_correct = feat_correct[:,:,correct_idx]
    
    c_labels_correct = np.concatenate([np.array(list(c_label_dict[i].values()))for i in range(num_blocks)],0)
    c_labels_correct = c_labels_correct[correct_idx]
    s_labels_correct = np.concatenate([np.array(list(s_label_dict[i].values()))for i in range(num_blocks)],0)
    s_labels_correct = s_labels_correct[correct_idx]
    r_labels_correct = np.concatenate([np.array(list(r_label_dict[i].values()))for i in range(num_blocks)],0)
    r_labels_correct = r_labels_correct[correct_idx]
    
    ########### firing rate differences between relevent and irrelevent
    
    # sort by rel vs irrel
    feat_rule1 = feat_correct[:,:,r_labels_correct==0] # colour
    feat_rule2 = feat_correct[:,:,r_labels_correct==1] # shape

    rel_diff1 = np.mean(abs(feat_rule1[0,:,:] - feat_rule1[1,:,:]),1)
    rel_diff2 = np.mean(abs(feat_rule2[2,:,:] - feat_rule2[3,:,:]),1)
    rel_diff = .5*(rel_diff1 + rel_diff2)
    
    irrel_diff1 = np.mean(abs(feat_rule1[2,:,:] - feat_rule1[3,:,:]),1)
    irrel_diff2 = np.mean(abs(feat_rule2[0,:,:] - feat_rule2[1,:,:]),1)
    irrel_diff = .5*(irrel_diff1 + irrel_diff2)

    
    ########### calculate reaction times and accuracy by congruency (only for correct trials)
    
    sorted_c_labels = np.zeros([num_trials,num_blocks])
    sorted_s_labels = np.zeros([num_trials,num_blocks])
    
    for blk in range(num_blocks):
        # labels
        c_labels = list(c_label_dict[blk].values())
        s_labels = list(s_label_dict[blk].values())
        sorted_c_labels[:,blk] = np.array(c_labels)
        sorted_s_labels[:,blk] = np.array(s_labels)
    
    # sort trials by congruency (green square, blue circle)
    c_labels = np.squeeze(sorted_c_labels.reshape(num_blocks * num_trials))
    s_labels = np.squeeze(sorted_s_labels.reshape(num_blocks * num_trials))
    grsq_idx = np.logical_and(c_labels == 1,s_labels == 1)
    blci_idx = np.logical_and(c_labels == 2,s_labels == 2)
    
    con_idx = np.logical_or(grsq_idx,blci_idx)
    con_idx = np.logical_and(correct_idx,con_idx)
    incon_idx = np.logical_and(correct_idx,con_idx==0)
    
    congruent_idx = np.squeeze(np.where(con_idx==True))
    incongruent_idx = np.squeeze(np.where(incon_idx==True))
    
    congruent_rt = np.mean(rt[congruent_idx])
    incongruent_rt = np.mean(rt[incongruent_idx])
    
    congruent_accuracy = sum(accuracy_flat[congruent_idx])/(.5*(num_trials*num_blocks))
    incongruent_accuracy = sum(accuracy_flat[incongruent_idx])/(.5*(num_trials*num_blocks))
    
    return accuracy_overall, correct_rt, congruent_rt, congruent_accuracy, incongruent_rt, incongruent_accuracy,\
        conj_correct,feat_correct, c_labels_correct, s_labels_correct, r_labels_correct, irrel_diff, rel_diff



