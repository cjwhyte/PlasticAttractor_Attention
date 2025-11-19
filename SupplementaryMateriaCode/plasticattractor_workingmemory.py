"""
Plastic attractor model of WM/attention based on psuedo code described in the appendix of Manohar et al (2019)
with additional long term plasticity. 

Created on 8/19/2021

@author: Christopher Whyte

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seed random number generator
np.random.seed(11)

###################
# Params 
###################

# simulation length
sim_length = 1700

# number of feature units
num_features = 8

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
W_ff = alpha[4] * np.eye(num_features) + alpha[3] * np.array([[1, 1, 1, 1, 0, 0, 0, 0],\
                                                              [1, 1, 1, 1, 0, 0, 0, 0],\
                                                              [1, 1, 1, 1, 0, 0, 0, 0],\
                                                              [1, 1, 1, 1, 0, 0, 0, 0],\
                                                              [0, 0, 0, 0, 1, 1, 1, 1],\
                                                              [0, 0, 0, 0, 1, 1, 1, 1],\
                                                              [0, 0, 0, 0, 1, 1, 1, 1],\
                                                              [0, 0, 0, 0, 1, 1, 1, 1]])
    
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
conjunction_units = np.zeros((num_conjunction,sim_length));

# initialise feature unit firing rates
feature_units = np.zeros((num_features,sim_length));
    
###################
# Specify stimulus
###################  

# this is a feature neuron x time matrix. When a stimulus is presented the
# active features = 1, and the inactive features = -1. Features are organised 
# according to [green, blue, square, triangle]

stimulus = np.zeros((num_features,sim_length))

# foreperiod: 200ms
for t in range(0,200):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0] # no stimulus   
# present first object: 120ms
for t in range(200,350):
     stimulus[:,t] = [1, 0, 0, 0, 1, 0, 0, 0]    # stim 1
# isi: 100ms
for t in range(350,500):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0]     # blank screen       
# present object 2
for t in range(500,650):
    stimulus[:,t] = [0, 1, 0, 0, 0, 1, 0, 0]     # stim 2
# isi: 100ms
for t in range(650,800):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0]     # blank screen
# present object 3 120ms
for t in range(800,950):
    stimulus[:,t] = [0, 0, 1, 0, 0, 0, 1, 0]     # stim 3
# isi: 100ms
for t in range(950,1100):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0]     # blank screen
# present object 4
for t in range(1100,1250):
    stimulus[:,t] = [0, 0, 0, 1, 0, 0, 0, 1]     # stim 4
# delay: 100ms  
for t in range(1250,1400):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0]     # blank screen
# probe
for t in range(1400,1550):
    stimulus[:,t] = [1, 0, 0, 0, 0, 0, 0, 0]     # present green bar
# choice period
for t in range(1550,sim_length):
    stimulus[:,t] = [0, 0, 0, 0, 0, 0, 0, 0]     # blank screen

###################
# Simulate task
###################  

# simulate network
for t in range(1,sim_length):
    
    # feature neuron
    feature_units[:,t] = beta + W_ff @ (feature_units[:,t-1] - beta) + alpha[2] * W @ (conjunction_units[:,t-1] - beta) + stimulus[:,t]
    
    # apply non-linearity to feature neuron 
    feature_units[:,t] = np.maximum(0,np.minimum(1,feature_units[:,t]))

    # conjunction neuron
    conjunction_units[:,t] = beta + W_cc @ (conjunction_units[:,t-1] - beta) + alpha[5] * W.T @ (feature_units[:,t-1] - beta) + 0.005 * np.random.randn(num_conjunction,1).T
    
    # apply non-linearity to conjection neuron
    conjunction_units[:,t] = np.maximum(0,np.minimum(1,conjunction_units[:,t]))
    
    # calculate delta term for weight update
    delta_w = np.outer((feature_units[:,t] - beta),(conjunction_units[:,t] - beta))
    
    # weight update
    w_short = np.maximum(0,np.minimum(1, w_short + gamma * delta_w))
    w_long  = np.maximum(0,np.minimum(.2, w_long  + .01*gamma * delta_w))
    W = w_short + w_long
    
    [U, S, V] = np.linalg.svd(W.T @ W)
    S = np.sqrt(S)
    

# model choice
choice_index = max(feature_units[4:,t])
choice = np.where(feature_units[:,t] == choice_index);

print(choice)
print(feature_units[:,t])

###################
# Figures
###################

plt.rcParams['font.size'] = 12

time = np.array(range(1,sim_length+1))
inhibition_threshold = np.ones_like(time)*beta
fig, ax = plt.subplots(2)
ax[0].plot(time,conjunction_units[0,:].T,color = 'lightseagreen')
ax[0].plot(time,conjunction_units[1,:].T,color = 'teal')
ax[0].plot(time,conjunction_units[2,:].T,color = 'darkturquoise')
ax[0].plot(time,conjunction_units[3,:].T,color = 'dodgerblue')
ax[0].set_ylim([0, 1])
ax[0].plot(time,inhibition_threshold, 'k--')
ax[0].margins(x=0)
ax[1] = sns.heatmap(feature_units,cmap="Greys",cbar=False, xticklabels=False)
plt.show()



  