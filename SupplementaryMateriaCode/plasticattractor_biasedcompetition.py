"""
Plastic attractor model of WM/attention based on psuedo code described in the appendix of Manohar et al (2019)
with additional long term plasticity. 

Biased competition simulation

Created on 8/19/2021

@author: Christopher Whyte

"""

import numpy as np
import matplotlib.pyplot as plt

# seed random number generator
np.random.seed(0)

###################
# Params 
###################

# simulation length
sim_length = 700

# number of feature units
num_features = 2

# number of conjection units 
num_conjunction = 4

# alpha(0), alpha(3) = mutual lateral inhibition
# alpha(1), alpha(4) = self excitation / temporal decay
# alpha(2), alpha(5) = synaptic gain
alpha = np.array([-.45, 1.02, 0.08, -.28, .73, .05])
    
# baseline activity/lateral inhibition threshold
beta = .175
    

# learning rate for hebbian updates
gamma = .02

# feature -> feature weights. This is a num_features x num_features matrix with
# 2x2 matrices down the diagonal so that only neurons within each feature compete
# via lateral inhibition. 
W_ff = alpha[4] * np.eye(num_features) + alpha[3] * np.array([[1, 1],\
                                                              [1, 1]])

    
# conjunction -> conjection weights. This is a num_conjection x
# num_conjection matrix with alpha(0) + alpha(1) on the diagonal and
# alpha(0) in the off diagonal elements. When multiplied with a vectors
# whose elements are [Conjection_neuron - beta] this creates lateral
# inhibition, or self excitation, depending on whether the conjunction
# neurons are > or < beta.
W_cc = alpha[1]*np.eye(num_conjunction) + alpha[0]*np.ones(num_conjunction);

# initialise conjection <-> feature weights. These are what get modified
# through hebbian learning.
w_short = np.random.rand(num_features, num_conjunction);
w_long = np.random.rand(num_features, num_conjunction);
W = np.random.rand(num_features, num_conjunction);

# initialise conjunction unit firing rates
conjunction_units = np.zeros((num_conjunction,sim_length,2));

# initialise feature unit firing rates
feature_units = np.zeros((num_features,sim_length,2));


    
###################
# Specify stimulus
###################  

def stim_generator(trial):
    
    stimulus = np.zeros((num_features,sim_length))
    
    if trial == 0:
        a = 1; b = 0
    elif trial == 1:
        a = 0; b = 1
    
    # foreperiod: 200ms
    for t in range(0,100):
        stimulus[:,t] = [0, 0] # no stimulus   
    # present cue
    for t in range(100,200):
         stimulus[:,t] = [a, b]    # cue
    # isi: 100ms
    for t in range(200,400):
        stimulus[:,t] = [0, 0]     # delay    
    # present object 2
    for t in range(400,450):
        stimulus[:,t] = [1, 1]     # probes
    # post stim
    for t in range(500,sim_length):
        stimulus[:,t] = [0, 0]     # post-stim
        
    return stimulus


###################
# Simulate task
###################  

for trial in range(2):
    
    stimulus = stim_generator(trial)

    # simulate network
    for t in range(1,sim_length):
        
        # feature neuron
        feature_units[:,t,trial] = beta + W_ff @ (feature_units[:,t-1,trial] - beta) + alpha[2] * W @ (conjunction_units[:,t-1,trial] - beta) + stimulus[:,t]
        
        # apply non-linearity to feature neuron 
        feature_units[:,t,trial] = np.maximum(0,np.minimum(1,feature_units[:,t,trial]))
    
        # conjunction neuron
        conjunction_units[:,t,trial] = beta + W_cc @ (conjunction_units[:,t-1,trial] - beta) + alpha[5] * W.T @ (feature_units[:,t-1,trial] - beta) + 0.005 * np.random.randn(num_conjunction,1).T
        
        # apply non-linearity to conjection neuron
        conjunction_units[:,t,trial] = np.maximum(0,np.minimum(1,conjunction_units[:,t,trial]))
        
        # calculate delta term for weight update
        delta_w = np.outer((feature_units[:,t,trial] - beta),(conjunction_units[:,t,trial] - beta))
        
        # weight update
        w_short = np.maximum(0,np.minimum(1, w_short + gamma * delta_w))
        w_long  = np.maximum(0,np.minimum(.2, w_long  + .01*gamma * delta_w))
        W = w_short + w_long
        
        [U, S, V] = np.linalg.svd(W.T @ W)
        S = np.sqrt(S)
    

###################
# Figure
###################

time = np.array(range(-100,sim_length-100))
inhibition_threshold = np.ones_like(time)*beta

fig = plt.figure()
plt.margins(x=0)
plt.plot(time,feature_units[0,:,0].T,'k')
plt.plot(time,feature_units[0,:,1].T,'k--')
plt.xlabel('Time from cue onset (time steps)')
plt.ylabel('Firing rate')
plt.show()




  