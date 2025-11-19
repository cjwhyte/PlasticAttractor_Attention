"""
Plastic attractor network simulation script

@author: Christopher Whyte

"""""

def plasticattractor_sim_SHFL(num_blocks, num_trials, rnd_seed, TMS_sim , TMS_start):
    
    import numpy as np
    
    # seed random number generator
    np.random.seed(rnd_seed)
    
    # %% params 
    
    # number of feature units
    num_features = 6 
    
    # number of conjection units 
    num_conjunction = 4
    
    # col1 = conj, col2 = feat
    # alpha(0), alpha(3) = mutual lateral inhibition
    # alpha(1), alpha(4) = self excitation / temporal decay 
    # alpha(2), alpha(5) = synaptic gain
    alpha = np.array([-.45, 1, .08, -.28, .73, .04]) 

    # baseline activity/lateral inhibition threshold
    beta = .175
    
    # learning rate for hebbian updates
    gamma = .02 
    
    # feature -> feature weights. This is a num_features x num_features matrix with
    # 2x2 matrices down the diagonal so that only neurons within each feature compete
    # via lateral inhibition. 
    W_ff = alpha[4] * np.eye(num_features) + alpha[3] * np.array([[1, 1, 0, 0, 0, 0],
                                                                  [1, 1, 0, 0, 0, 0],
                                                                  [0, 0, 1, 1, 0, 0],
                                                                  [0, 0, 1, 1, 0, 0],
                                                                  [0, 0, 0, 0, 1, 1],
                                                                  [0, 0, 0, 0, 1, 1]])
        
    # conjunction -> conjection weights. This is a num_conjection x
    # num_conjection matrix with alpha(0) + alpha(1) on the diagonal and
    # alpha(0) in the off diagonal elements. When multiplied with a vectors
    # whose elements are [Conjection_neuron - beta] this creates lateral
    # inhibition, or self excitation, depending on whether the conjunction
    # neurons are > or < beta.
    W_cc = alpha[1]*np.eye(num_conjunction) + alpha[0]*np.ones(num_conjunction)
    
    # initialise conjection <-> feature weights. These are what get modified
    # through hebbian learning.
    W = np.random.rand(num_features, num_conjunction)
        
    # %% specify stimulus and conditions
    
    # rules
    rules = np.zeros([2,num_features,2]) 
    rules[0,:,0] = [1, -1, .5, .5, 1, -1] # green -> button 1
    rules[1,:,0] = [-1, 1, .5, .5, -1, 1] # blue  -> button 2
    rules[0,:,1] = [.5, .5, 1, -1, 1, -1] # square -> button 1
    rules[1,:,1] = [.5, .5, -1, 1, -1, 1] # circle  -> button 2
    
    # stimuli 
    stimuli = np.zeros([4,num_features]) 
    stimuli[0,:] = [1, 0, 1, 0, 0, 0]   # green square
    stimuli[1,:] = [1, 0, 0, 1, 0, 0]   # green circle
    stimuli[2,:] = [0, 1, 1, 0, 0, 0]   # blue square
    stimuli[3,:] = [0, 1, 0, 1, 0, 0]   # blue circle
    
    # rule order and stim order
    rule_set = np.tile([0,1],int(num_blocks/2))
    rule_order = np.array([0,1])
    
    # number of trials
    trial_order = np.tile([0,1,2,3],int(num_trials/4))
    
    # rule period 
    rule_period = 400
    # stim period 
    stim_period = 400
    
    # isi
    isi = np.array([-1, -1, -1, -1, -1, -1])
    
    # resp period
    resp_period = np.array([0, 0, 0, 0, 0, 0])
      
    def stim_generator(rule_set_idx):
        # initialise stimulus dictionary
        stimulus = {}; c_labels = {}; s_labels = {}; r_labels = {}; ground_truth = {} 
        
        trial_counter = 0
        
        # rule period
        np.random.shuffle(rule_order) # shuffle
        for rule in range(2):
            rule_idx = rule_order[rule]
            rule_input = np.zeros([num_features,rule_period])
            for t in range(rule_period):
                # present rule
                if t <= 250:
                    rule_input[:,t] = rules[rule_idx,:,rule_set_idx]
                # isi
                elif t >= 250 and t <= 400:
                    rule_input[:,t] = isi   
            stimulus[trial_counter] = rule_input
            
            # ground truth action
            ground_truth[trial_counter] = 4 
            trial_counter += 1
                
        # stimulus period 
        np.random.shuffle(trial_order) # shuffle
        for trl in range(num_trials):
            cond_idx = trial_order[trl]
            stim_input = np.zeros([num_features,stim_period])
            for t in range(stim_period):
                
                # isi
                if t <= 50:
                    stim_input[:,t] = isi
                # present stimulus
                elif t >= 50 and t <= 100:
                    stim_input[:,t] = stimuli[cond_idx,:]
                # resp period
                elif t >= 100 and t <= 350:
                    stim_input[:,t] = resp_period
                # isi
                elif t >= 350 and t <= 400:
                    stim_input[:,t] = isi
                    
            stimulus[trial_counter] = stim_input
            
            # rules
            r_labels[trl] = rule_set_idx
            
            # labels for classifier
            if cond_idx==0:
                c_labels[trl] = 1
                s_labels[trl] = 1
            elif cond_idx==1:
                c_labels[trl] = 1
                s_labels[trl] = 2
            elif cond_idx==2:
                c_labels[trl] = 2
                s_labels[trl] = 1
            elif cond_idx==3:
                c_labels[trl] = 2
                s_labels[trl] = 2
                    
            # ground truth action
            if rule_set_idx == 0:
                if cond_idx == 0 or cond_idx == 1:
                    ground_truth[trial_counter] = 0
                else:
                    ground_truth[trial_counter] = 1
            elif rule_set_idx == 1:
                if cond_idx == 0 or cond_idx == 2:
                    ground_truth[trial_counter] = 0
                else:
                    ground_truth[trial_counter] = 1  
                    
            trial_counter += 1
        
        return stimulus, ground_truth, c_labels, s_labels, r_labels
                
    # %% simulate task
    
    # initialise dicts to store network activity and labels
    conj_dict = {}; feat_dict = {}; choice_dict = {}; acc_dict = {}; weight_dict = {}
    rt_dict = {}; s_label_dict = {}; r_label_dict = {}; c_label_dict = {}; 
    
    # initialise conjection <-> feature weights. These are what get modified
    # through hebbian learning.
    w_short = np.random.rand(num_features, num_conjunction);
    w_long = np.random.rand(num_features, num_conjunction);
    # W = np.random.rand(num_features, num_conjunction,stim_period);
    W = np.random.rand(num_features, num_conjunction);
    
    for blk in range(num_blocks):
        
            # initialise conjunction unit firing rates
            conjunction_units = np.zeros((num_conjunction,stim_period,num_trials+2));
            
            # initialise feature unit firing rates
            feature_units = np.zeros((num_features,stim_period,num_trials+2));
            
            # initialise choice and reaction time arrays
            choice = np.zeros(num_trials+2);  rt = np.zeros(num_trials+2); accuracy = np.zeros(num_trials+2)
                
            # grab rule set for this trial
            rule_idx = int(rule_set[blk])
            
            # call stimulus function
            stimulus_dict, ground_truth, c_labels, s_labels, rule = stim_generator(rule_idx)
            
            for trl in range(num_trials+2):
                
                # grab stimulus for this trial
                stimulus = stimulus_dict[trl]
                            
                for t in range(stim_period):
                    
                    # feature neuron
                    feature_units[:,t,trl] = beta + W_ff @ (feature_units[:,t-1,trl] - beta) + alpha[2] * \
                                             W @ (conjunction_units[:,t-1,trl] - beta) + stimulus[:,t]
                                         
                    # apply non-linearity to feature neuron 
                    feature_units[:,t,trl] = np.maximum(0,np.minimum(1,feature_units[:,t,trl]))
                
                    # conjunction neuron   
                    if TMS_sim[blk,trl] == 1 and t >= TMS_start and t <= 100 and trl >= 2 and TMS_start!= 100:
                        conjunction_units[:,t,trl] = np.ones(num_conjunction)
                    else:
                        conjunction_units[:,t,trl] = beta + W_cc @ (conjunction_units[:,t-1,trl] - beta) + alpha[5] \
                                                     * W.T @ (feature_units[:,t-1,trl] - beta) \
                                                     + 0.005 * np.random.randn(num_conjunction,1).T    
                    
                    # apply non-linearity to conjection neuron
                    conjunction_units[:,t,trl] = np.maximum(0,np.minimum(1,conjunction_units[:,t,trl]))
                    
                    # calculate delta term for weight update
                    delta_w = np.outer((feature_units[:,t,trl] - beta),(conjunction_units[:,t,trl] - beta))
                    
                    # weight update
                    w_short = np.maximum(0,np.minimum(1, w_short + gamma * delta_w))
                    w_long  = np.maximum(0,np.minimum(.2, w_long  + .01*gamma * delta_w)) # evolves at 2 orders of magnitude the rate
                    W = w_short + w_long
                    
                    # # turn off plasticity after rule presentation
                    # if trl < 2:
                    #     w_short = np.maximum(0,np.minimum(1, w_short + gamma * delta_w))
                    #     w_long  = np.maximum(0,np.minimum(.2, w_long  + .01*gamma * delta_w)) # evolves at 2 orders of magnitude the rate
                    #     W = w_short + w_long
                    
                    
                # find max activated "action" feature after stimulus offset (offset is padded by +10 timesteps)
                max_action = np.amax(feature_units[4:,110:,trl])
                # action_index = np.argwhere(feature_units[4:,110:,trl] == max_action)
                threshold = 98*(max_action/100)
                action_index = np.argwhere(feature_units[4:,110:,trl] > threshold)
                
                # choice
                choice[trl] = action_index[0,0]
                
                # reaction time 
                rt[trl] = action_index[0,1]
                
                # accuracy   
                if choice[trl] == ground_truth[trl]:
                    accuracy[trl] = 1
                else:
                    accuracy[trl] = 0
                    
                weight_dict[trl,blk] = W
                                    
            # store trial information in relevent dictionaries
            conj_dict[blk] = conjunction_units
            feat_dict[blk] = feature_units
            choice_dict[blk] = choice
            acc_dict[blk] = accuracy
            rt_dict[blk] = rt
            c_label_dict[blk] = c_labels
            s_label_dict[blk] = s_labels
            r_label_dict[blk] = rule
            # weight_dict[blk] = W
            
    return conj_dict, feat_dict, choice_dict, acc_dict, rt_dict, c_label_dict, s_label_dict, r_label_dict, weight_dict

