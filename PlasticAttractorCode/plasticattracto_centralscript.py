"""
Central script for plastic attractor model of feature-based attention 

This script calls the relevent functions to reproduce all simulations
and figures described in the first two simulations of the thesis.

@author: Christopher Whyte

"""""

import numpy as np
# simulate network
from plasticattractor_networksims import plasticattractor_sim
# behavioural analysis 
from plasticattractor_behaviouralanalysis import plasticattractor_beh
# decoding analysis 
from plasticattractor_decoding import plasticattractor_decode

# run decoding analysis (set to 1 to run decoding - be warned it will take some
# time to run)
decoding = 0

# number of blocks
num_blocks = 20

# trials per block (must be a multiple of 4)
num_trials = 12

# number of random seeds 
n_seeds = 2

# %% basic simulation

c_lab = {}; s_lab = {}; r_lab = {}; conj = {}; feat = {}; 
choice = {}; acc = {}; rule_set = {}; rt = {}; weights = {}

for rnd in range(n_seeds):
    conj[rnd], feat[rnd], choice[rnd], acc[rnd],rt[rnd],\
    c_lab[rnd], s_lab[rnd], r_lab[rnd], weights[rnd]\
    = plasticattractor_sim(num_blocks, num_trials, rnd, TMS_sim = False, TMS_start = None)
    print('Basic task simulation progress', ((rnd+1)/n_seeds)*100, '%')
        
# %% behavioural analysis and data sorting

accuracy_overall = {}; correct_rt = {}; congruent_rt = {}; congruent_acc = {}; incongruent_rt = {}; 
incongruent_acc = {}; conj_corr = {}; feat_corr = {}; c_lab_corr = {}; s_lab_corr = {}; r_lab_corr = {};
rel_feat_diff= {}; irrel_feat_diff= {};

for rnd in range(n_seeds):
    accuracy_overall[rnd], correct_rt[rnd], congruent_rt[rnd], congruent_acc[rnd], incongruent_rt[rnd],incongruent_acc[rnd],\
    conj_corr[rnd],feat_corr[rnd],c_lab_corr[rnd], s_lab_corr[rnd], r_lab_corr[rnd], irrel_feat_diff[rnd], rel_feat_diff[rnd]\
    = plasticattractor_beh(num_trials, num_blocks, acc[rnd], rt[rnd], c_lab[rnd], s_lab[rnd], r_lab[rnd], conj[rnd], feat[rnd])
    
# compute mean 
mean_accuracy = np.mean(np.array(list(accuracy_overall.values())))
mean_rt = np.mean(np.array(list(correct_rt.values())))

mean_congruent_acc = np.mean(np.array(list(congruent_acc.values())))
mean_incongruent_acc = np.mean(np.array(list(incongruent_acc.values())))

mean_congruent_rt = np.mean(np.array(list(congruent_rt.values())))
mean_incongruent_rt = np.mean(np.array(list(incongruent_rt.values())))

# standard error
ster_accuracy = np.std(np.array(list(accuracy_overall.values())))/np.sqrt(n_seeds)
ster_rt = np.std(np.array(list(correct_rt.values())))/np.sqrt(n_seeds)

ster_congruent_acc = np.std(np.array(list(congruent_acc.values())))/np.sqrt(n_seeds)
ster_incongruent_acc = np.std(np.array(list(incongruent_acc.values())))/np.sqrt(n_seeds)

ster_congruent_rt = np.std(np.array(list(congruent_rt.values())))/np.sqrt(n_seeds)
ster_incongruent_rt = np.std(np.array(list(incongruent_rt.values())))/np.sqrt(n_seeds)

# %% TMS simulation 

s_lab_tms = {}; r_lab_tms = {}; rule_set_tms = {}; conj_tms = {}; feat_tms = {}
choice_tms = {}; acc_tms = {}; rt_tms = {}; c_lab_tms = {}; weights_tms = {}

np.zeros([400,n_seeds]);

prog_counter = 0
tms_vals = np.arange(0,20,1)
for rnd_seed in range(n_seeds):
    for i, tms_start in enumerate(tms_vals):
        conj_tms[rnd_seed,i], feat_tms[rnd_seed,i], choice_tms[rnd_seed,i], acc_tms[rnd_seed,i],rt_tms[rnd_seed,i],\
        c_lab_tms[rnd_seed,i], s_lab_tms[rnd_seed,i], r_lab_tms[rnd_seed,i], weights_tms[rnd_seed,i], \
        = plasticattractor_sim(num_blocks, num_trials, rnd_seed, TMS_sim = True, TMS_start = (100-tms_start))
        prog_counter += 1
        print('TMS simulation progress', (prog_counter/(n_seeds*len(tms_vals)))*100, '%')

# %% behavioural analysis and data sorting TMS 

# # intermediate dictionaries
accuracy_overall_tms = {}; correct_rt_tms = {}; congruent_rt_tms = {}; congruent_acc_tms = {}; 
incongruent_rt_tms = {}; incongruent_acc_tms = {}; congruent_rt_tms = {}; incongruent_rt_tms = {}
# dictionaries for each value of the tms param
mean_accuracy_tms = {}; mean_rt_tms = {}; mean_congruent_rt_tms = {}; mean_congruent_acc_tms = {}
mean_incongruent_rt_tms = {}; mean_incongruent_acc_tms = {}; ster_accuracy_tms = {}; ster_congruent_acc_tms = {}
ster_incongruent_acc_tms = {}; ster_rt_tms = {}; ster_congruent_rt_tms = {}; ster_incongruent_rt_tms = {}
rel_feat_diff_tms = {}; irrel_feat_diff_tms = {};

# dictionaries for activity
conj_corr_tms = {}; feat_corr_tms = {}; c_labels_corr_tms = {}; s_labels_corr_tms = {}; r_labels_corr_tms = {}

# dict for raw reaction time and accuracy data
congruent_rt_tms_store = {} ; congruent_acc_tms_store = {} ; incongruent_rt_tms_store = {}; incongruent_acc_tms_store = {}

for tms in range(len(tms_vals)):
    for rnd in range(n_seeds):
        accuracy_overall_tms[rnd], correct_rt_tms[rnd], congruent_rt_tms[rnd],congruent_acc_tms[rnd], incongruent_rt_tms[rnd],incongruent_acc_tms[rnd],\
        conj_corr_tms[rnd,tms], feat_corr_tms[rnd,tms], c_labels_corr_tms[rnd,tms], s_labels_corr_tms[rnd,tms], r_labels_corr_tms[rnd,tms], irrel_feat_diff_tms[rnd,tms],rel_feat_diff_tms[rnd,tms] \
        = plasticattractor_beh(num_trials, num_blocks, acc_tms[rnd,tms], rt_tms[rnd,tms], c_lab_tms[rnd,tms], s_lab_tms[rnd,tms], r_lab_tms[rnd,tms], conj_tms[rnd,tms], feat_tms[rnd,tms])
        
    # store raw data
    congruent_rt_tms_store[tms] = np.array(list(congruent_rt_tms.values()))
    congruent_acc_tms_store[tms] = np.array(list(congruent_acc_tms.values()))
    incongruent_rt_tms_store[tms]  = np.array(list(incongruent_rt_tms.values()))
    incongruent_acc_tms_store[tms] = np.array(list(incongruent_acc_tms.values()))   
        
    # compute mean 
    mean_accuracy_tms[tms] = np.mean(np.array(list(accuracy_overall_tms.values())))
    
    mean_congruent_acc_tms[tms] = np.mean(np.array(list(congruent_acc_tms.values())))
    mean_incongruent_acc_tms[tms] = np.mean(np.array(list(incongruent_acc_tms.values())))
    
    mean_rt_tms[tms] = np.mean(np.array(list(correct_rt_tms.values())))
    mean_congruent_rt_tms[tms] = np.mean(np.array(list(congruent_rt_tms.values())))
    mean_incongruent_rt_tms[tms] = np.mean(np.array(list(incongruent_rt_tms.values())))
    
    mean_cong_diff_rt = np.mean(np.array(list(incongruent_rt_tms.values())) - np.array(list(congruent_rt_tms.values())))
        
    # standard error
    ster_accuracy_tms[tms] = np.std(np.array(list(accuracy_overall_tms.values())))/np.sqrt(n_seeds)
    
    ster_congruent_acc_tms[tms] = np.std(np.array(list(congruent_acc_tms.values())))/np.sqrt(n_seeds)
    ster_incongruent_acc_tms[tms] = np.std(np.array(list(incongruent_acc_tms.values())))/np.sqrt(n_seeds)
    
    ster_rt_tms[tms] = np.std(np.array(list(correct_rt_tms.values())))/np.sqrt(n_seeds)
    ster_congruent_rt_tms[tms] = np.std(np.array(list(congruent_rt_tms.values())))/np.sqrt(n_seeds)
    ster_incongruent_rt_tms[tms] = np.std(np.array(list(incongruent_rt_tms.values())))/np.sqrt(n_seeds)
    
    ster_cong_diff_rt = np.std(np.array(list(incongruent_rt_tms.values())) - np.array(list(congruent_rt_tms.values())))/np.sqrt(n_seeds)
        
# %% Weight matrix eigenvalue analysis

# basic simulations
sat_vals = np.zeros([n_seeds,num_trials, num_blocks])
largest_val =  np.zeros([n_seeds, num_trials, num_blocks])
for rnd in range(n_seeds):
    for blk in range(num_blocks):
        for trl in range(num_trials):
            W = weights[rnd][trl,blk]
            [vals, vecs] = np.linalg.eig(W@W.T)
            # number of saturating eigen values (i.e. >1)
            sat_vals[rnd,trl,blk] = np.sum(vals>1)
            largest_val[rnd,trl,blk] = np.max(vals)
        
mean_sat_vals = np.mean(np.mean(sat_vals,0),1)
mean_largest_val = np.mean(np.mean(largest_val,0),1)

# tms simulations
sat_vals_tms = np.zeros([n_seeds, num_trials, num_blocks, len(tms_vals)])
largest_val_tms =  np.zeros([n_seeds, num_trials, num_blocks, len(tms_vals)])
W_spread =  np.zeros([n_seeds, num_trials, num_blocks, len(tms_vals)])
for tms in range(len(tms_vals)):
    for rnd in range(n_seeds):
        for blk in range(num_blocks):
            for trl in range(num_trials):
                W = weights_tms[rnd,tms][trl,blk]
                W_spread[rnd,trl,blk,tms] = np.mean(np.std(W[0:3,:],0))
                [vals, vecs] = np.linalg.eig(W@W.T)
                # number of saturating eigen values (i.e. >1)
                sat_vals_tms[rnd,trl,blk,tms] = np.sum(vals>1)
                largest_val_tms[rnd,trl,blk,tms] = np.max(vals)
            
mean_W_spread = np.mean(np.mean(W_spread,0),1)
mean_sat_vals_tms = np.mean(np.mean(sat_vals_tms,0),1)
mean_largest_val_tms = np.mean(np.mean(largest_val_tms,0),1)  
ster_sat_vals_tms = np.std(np.mean(sat_vals_tms,0),1)/np.sqrt(n_seeds)
ster_largest_val_tms = np.std(np.mean(largest_val_tms,0),1)/np.sqrt(n_seeds)

# %% Firing rate differences for irrelevant feature dimension

irrel_diff = np.zeros([len(tms_vals),n_seeds,400])
irrel_meandiff = np.zeros([len(tms_vals),n_seeds])
rel_diff = np.zeros([len(tms_vals),n_seeds,400])
rel_meandiff = np.zeros([len(tms_vals),n_seeds])

# average over time points
for tms in range(len(tms_vals)):
    for rnd in range(n_seeds):
        rel_diff[tms,rnd,:] = rel_feat_diff_tms[rnd,tms]
        irrel_diff[tms,rnd,:] = irrel_feat_diff_tms[rnd,tms]
        diff = rel_feat_diff_tms[rnd,tms]
        rel_meandiff[tms,rnd] = np.mean(diff)
        diff = irrel_feat_diff_tms[rnd,tms]
        irrel_meandiff[tms,rnd] = np.mean(diff)

# average over seeds
rel_diff = np.mean(rel_diff,1)
irrel_diff = np.mean(irrel_diff,1)


# %% behavioural plots

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

plt.rcParams['font.size'] = 12

num_conj = 4
num_feat = 6
stim_period = 400
rule_period = 400
rule_length = 2*rule_period 
stim_length = num_trials*stim_period
sim_length = rule_length + stim_length
time = np.arange(0,stim_period,1)

###### example plot
example_blk = 15
# extract and reshape example conjunction units
conjunction_units = conj[0][example_blk]
conjunction_units = conjunction_units.reshape([num_conj,stim_period*(num_trials+2)], order='F')
# extract and reshape example feature units
feature_units = feat[1][example_blk]
feature_units = feature_units.reshape([num_feat,stim_period*(num_trials+2)], order='F')

# stimulus period
rule_time = rule_period*2
time = np.array(range(1,sim_length+1))
inhibition_threshold = np.ones_like(time)*.175
ylabels = ("Green", "Blue", "Square", "Circle")

fig, ax = plt.subplots(3)
ax[0].plot(time,feature_units[4,:].T, color = 'darkviolet')
ax[0].plot(time,feature_units[5,:].T, color = 'orchid')
ax[0].margins(x=0)
ax[1].plot(time,conjunction_units[0,:].T,color = 'lightseagreen')
ax[1].plot(time,conjunction_units[1,:].T,color = 'teal')
ax[1].plot(time,conjunction_units[2,:].T,color = 'darkturquoise')
ax[1].plot(time,conjunction_units[3,:].T,color = 'dodgerblue')
ax[1].plot(time,inhibition_threshold, 'k--')
ax[1].margins(x=0)
ax[2] = sns.heatmap(feature_units[:4,:],cmap="Greys",cbar=False, xticklabels=False, yticklabels = ylabels)
plt.show()

###### example plot TMS
tms = 1
example_blk = 10
# extract and reshape example conjunction units
conjunction_units = conj_tms[1,tms][example_blk]
conjunction_units = conjunction_units.reshape([num_conj,stim_period*(num_trials+2)], order='F')
# extract and reshape example feature units
feature_units = feat_tms[1,tms][example_blk]
feature_units = feature_units.reshape([num_feat,stim_period*(num_trials+2)], order='F')

rule_time = rule_period*2
time = np.array(range(1,sim_length+1))
inhibition_threshold = np.ones_like(time)*.175
ylabels = ("Green", "Blue", "Square", "Circle")

fig, ax = plt.subplots(3)
ax[0].plot(time,feature_units[4,:].T, color = 'darkviolet')
ax[0].plot(time,feature_units[5,:].T, color = 'orchid')
ax[0].margins(x=0)
ax[1].plot(time,conjunction_units[0,:].T,color = 'lightseagreen')
ax[1].plot(time,conjunction_units[1,:].T,color = 'teal')
ax[1].plot(time,conjunction_units[2,:].T,color = 'darkturquoise')
ax[1].plot(time,conjunction_units[3,:].T,color = 'dodgerblue')
ax[1].plot(time,inhibition_threshold, 'k--')
ax[1].margins(x=0)
ax[2] = sns.heatmap(feature_units[:4,:],cmap="Greys", cbar=False, xticklabels=False, yticklabels = ylabels)
plt.show()

###### accuracy, rt, and congruency plots

# accuracy vs TMS strength
tms_strength = np.arange(0,len(tms_vals),1)
mean_accuracy_tms = np.array(list(mean_accuracy_tms.values()))
ster_accuracy_tms = np.array(list(ster_accuracy_tms.values()))
fig = plt.figure()
# plt.title('TMS train length vs accuracy')
plt.errorbar(tms_strength[1:20],100*mean_accuracy_tms[1:20], yerr = 100*ster_accuracy_tms[1:20], color = 'black', capsize=3, linestyle="-",marker="o")
plt.xlabel("Train length")
plt.ylabel("Accuracy %")
plt.xticks(np.arange(0,20,1))

# rt vs TMS strength
tms_strength = np.arange(0,len(tms_vals),1)
mean_rt_tms = np.array(list(mean_rt_tms.values()))
ster_rt_tms = np.array(list(ster_rt_tms.values()))
fig = plt.figure()
plt.title('TMS train length vs reaction time')
plt.errorbar(tms_strength[1:20],mean_rt_tms[1:20], yerr = ster_rt_tms[1:20],color = 'black', capsize=3, linestyle="-",marker="o")
plt.xlabel("Train length")
plt.ylabel("Reaction time")
plt.xticks(np.arange(0,20,1))

# congruency vs TMS line acccuracy
mean_congruent_acc_tms = np.array(list(mean_congruent_acc_tms.values()))
mean_incongruent_acc_tms = np.array(list(mean_incongruent_acc_tms.values()))
ster_congruent_acc_tms = np.array(list(ster_congruent_acc_tms.values()))
ster_incongruent_acc_tms = np.array(list(ster_incongruent_acc_tms.values()))
fig = plt.figure()
# plt.title('Train length vs congrunecy effect')
plt.errorbar(tms_strength[1:20],100*mean_congruent_acc_tms[1:20], yerr = 100*ster_congruent_acc_tms[1:20], color = 'indianred', capsize=3, linestyle="-",marker="o")
plt.errorbar(tms_strength[1:20],100*mean_incongruent_acc_tms[1:20], yerr = 100*ster_incongruent_acc_tms[1:20], color = 'darkred', capsize=3, linestyle="-",marker="o")
plt.xticks(np.arange(0,20,1))
# fig.legend(['Congruent','Incongruent'])
plt.xlabel("Train length")
plt.ylabel("Accuracy %")

# congruency vs TMS reaction time
mean_congruent_rt_tms = np.array(list(mean_congruent_rt_tms.values()))
mean_incongruent_rt_tms = np.array(list(mean_incongruent_rt_tms.values()))
ster_congruent_rt_tms = np.array(list(ster_congruent_rt_tms.values()))
ster_incongruent_rt_tms = np.array(list(ster_incongruent_rt_tms.values()))
fig = plt.figure()
# plt.title('Train length vs congrunecy effect')
plt.errorbar(tms_strength[1:20],mean_congruent_rt_tms[1:20], yerr = ster_congruent_rt_tms[1:20], color = 'indianred', capsize=3, linestyle="-",marker="o")
plt.errorbar(tms_strength[1:20],mean_incongruent_rt_tms[1:20], yerr = ster_incongruent_rt_tms[1:20], color = 'darkred', capsize=3, linestyle="-",marker="o")
plt.xticks(np.arange(0,20,1))
fig.legend(['Congruent','Incongruent'])
plt.xlabel("Train length")
plt.ylabel("Reaction times (time-steps)")

# incongruent - congruent rt box plot vs TMS
labels = ['1','2','3','4','5','6','7','8','9', '10',\
      '11','12','13','14','15','16','17','18', '19']
data = mean_incongruent_rt_tms[1:20] - mean_congruent_rt_tms[1:20]
error = ster_incongruent_rt_tms[1:20] - ster_congruent_rt_tms[1:20]
fig = plt.figure()
plt.title('Congruency effect vs TMS train length')
plt.bar(labels,data, color = 'lightgrey', capsize=3)
plt.xlabel("Train length")
plt.ylabel("Reaction times (time-steps)")

# # incongruent - congruent accuracy box plot vs TMS
# labels = ['1','2','3','4','5','6','7','8','9', '10',\
#       '11','12','13','14','15','16','17','18', '19']
# data =  mean_congruent_acc_tms[1:20] - mean_incongruent_acc_tms[1:20]
# error = ster_congruent_acc_tms[1:20] - ster_incongruent_acc_tms[1:20] 
# fig = plt.figure()
# plt.title('Congruency effect vs TMS train length')
# plt.bar(labels,data, color = 'lightgrey', capsize=3)
# plt.ylim([-.5, .5])

# congruent vs incongruent rt box plot TMS = 0
labels = ['Congruent','Incongruent']
data = np.array([list(congruent_rt.values()),list(incongruent_rt.values())])
fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
bplot = ax.boxplot(data.T,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               showfliers = False,
               labels=labels)  # will be used to label x-ticks

colors = ['indianred', 'darkred']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for median in bplot['medians']:
    median.set_color('black')
ax.set_ylabel('Reaction time (time-steps)')

# congruent vs incongruent reaction time box plot TMS = 11
tms = 10
labels = ['Congruent','Incongruent','TMS: congruent','TMS: incongruent']
data = np.array([list(congruent_rt.values()),list(incongruent_rt.values()),\
             congruent_rt_tms_store[tms],incongruent_rt_tms_store[tms]])
fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

bplot = ax.boxplot(data.T,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               showfliers = False,
               labels=labels)  # will be used to label x-ticks

colors = ['indianred', 'darkred','indianred', 'darkred']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for median in bplot['medians']:
    median.set_color('black')
    
# congruent vs incongruent accuracy box plot TMS = 10
labels = ['Congruent','Incongruent','TMS: congruent','TMS: incongruent']
data = np.array([list(congruent_acc.values()),list(incongruent_acc.values()),\
             congruent_acc_tms_store[tms],incongruent_acc_tms_store[tms]])
fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

bplot = ax.boxplot(data.T,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               showfliers = False,
               labels=labels)  # will be used to label x-ticks

colors = ['indianred', 'darkred','indianred', 'darkred']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

for median in bplot['medians']:
    median.set_color('black')
    
data = np.zeros([len(tms_vals),n_seeds])
for tms in range(len(tms_vals)):
        diff = incongruent_rt_tms_store[tms] - congruent_rt_tms_store[tms]
        data[tms,:] = diff
        
###### Eigen value plots

# eigen value vs TMS line plot
fig = plt.figure()
# plt.title('TMS train length vs number of saturating eigenvalues')
plt.errorbar(tms_strength[0:20],mean_sat_vals_tms[0,0:20], yerr= ster_sat_vals_tms[0,0:20], color = 'black', capsize=3, linestyle="-",marker="o")
plt.errorbar(tms_strength[0:20],mean_sat_vals_tms[5,0:20], yerr= ster_sat_vals_tms[5,0:20], color = 'blue', capsize=3, linestyle="-.",marker="o")
plt.errorbar(tms_strength[0:20],mean_sat_vals_tms[11,0:20], yerr= ster_sat_vals_tms[11,0:20], color = 'red', capsize=3, linestyle="-.",marker="o")
plt.xlabel("Strength")
plt.ylabel("Eigenvalues > 1")
plt.legend(['Trial 1', 'Trial 6', 'Trial 12' ])
plt.xticks(np.arange(0,20,1))

# weight spread vs TMS line plot
fig = plt.figure()
# plt.title('TMS train length vs number of saturating eigenvalues')
plt.plot(tms_strength[0:20],mean_W_spread[0,0:20], color = 'black', linestyle="-",marker="o")
plt.plot(tms_strength[0:20],mean_W_spread[5,0:20], color = 'blue', linestyle="-.",marker="o")
plt.plot(tms_strength[0:20],mean_W_spread[11,0:20], color = 'red', linestyle="-.",marker="o")
plt.xlabel("Strength")
plt.ylabel("W std")
plt.legend(['Trial 1', 'Trial 6', 'Trial 12' ])
plt.xticks(np.arange(0,20,1))


###### Difference in rel and irrel feat dims vs congruency

# scatter plots

# relevant
fig = plt.figure()
plt.scatter(np.mean(data[0:20],1),np.mean(rel_meandiff[0:20],1), color = 'lightgrey')
plt.xlabel("Average congruency effect")
plt.ylabel("Average difference in relevent firing rate (abs)")

# irrelevant
fig = plt.figure()
plt.scatter(np.mean(data[0:20],1),np.mean(irrel_meandiff[0:20],1), color = 'lightgrey')
plt.xlabel("Average congruency effect")
plt.ylabel("Average difference in irrelevent firing rate (abs)")

# difference plot
cmap_2 = mpl.cm.get_cmap('Greys')
time = np.arange(0,400)

# relevant
fig = plt.figure()
for i in range(len(tms_vals)):
    plt.plot(time,rel_diff[i,:].T, color = cmap_2((i+5)/30))
norm = mpl.colors.Normalize(vmin=0, vmax=20)
cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('Greys'))
cmap2.set_array([])
plt.colorbar(cmap2, ticks=np.arange(0,20)).ax.invert_yaxis() 
plt.xlabel("Time")
plt.ylabel("Difference in relevent firing rate (abs)")
    
# irrelevant
fig = plt.figure()
for i in range(len(tms_vals)):
    plt.plot(time,irrel_diff[i,:].T, color = cmap_2((i+5)/30))
norm = mpl.colors.Normalize(vmin=0, vmax=20)
cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('Greys'))
cmap2.set_array([])
plt.colorbar(cmap2, ticks=np.arange(0,20)).ax.invert_yaxis() 
plt.xlabel("Time")
plt.ylabel("Difference in irrelevent firing rate (abs)")

# correlation between incongruent dimension and congruent dimension
congruent_irreldiff_correlation = np.corrcoef(np.mean(data[0:20],1),np.mean(irrel_meandiff[0:20],1))

# %% Decoding

if decoding == 1:
    
    stim_period = 400

    accuracy_colour_feat = np.zeros([400,n_seeds]); accuracy_shape_feat = np.zeros([400,n_seeds])
    accuracy_colour_conj = np.zeros([400,n_seeds]); accuracy_shape_conj = np.zeros([400,n_seeds])
    accuracy_colour_feat_1 = np.zeros([400,n_seeds]); accuracy_shape_feat_1 = np.zeros([400,n_seeds])
    accuracy_colour_conj_1 = np.zeros([400,n_seeds]); accuracy_shape_conj_1 = np.zeros([400,n_seeds])
    accuracy_colour_feat_2 = np.zeros([400,n_seeds]); accuracy_shape_feat_2 = np.zeros([400,n_seeds])
    accuracy_colour_conj_2 = np.zeros([400,n_seeds]); accuracy_shape_conj_2 = np.zeros([400,n_seeds])
    
    print('Start decoding analysis')
    for rnd in range(n_seeds):
        accuracy_colour_feat[:,rnd], accuracy_shape_feat[:,rnd], accuracy_colour_conj[:,rnd], accuracy_shape_conj[:,rnd], \
        accuracy_colour_feat_1[:,rnd], accuracy_shape_feat_1[:,rnd], accuracy_colour_conj_1[:,rnd], accuracy_shape_conj_1[:,rnd], \
        accuracy_colour_feat_2[:,rnd], accuracy_shape_feat_2[:,rnd], accuracy_colour_conj_2[:,rnd], accuracy_shape_conj_2[:,rnd] \
        = plasticattractor_decode(num_trials, num_blocks, stim_period, c_lab_corr[rnd], s_lab_corr[rnd], r_lab_corr[rnd], conj_corr[rnd], feat_corr[rnd])
        print('Decoding progress', ((rnd+1)/n_seeds)*100, '%')
    
    # adecoding accuracy
    # mean
    accuracy_colour_feat_mean = np.mean(accuracy_colour_feat,1)
    accuracy_shape_feat_mean = np.mean(accuracy_shape_feat,1)
    accuracy_colour_feat_1_mean = np.mean(accuracy_colour_feat_1,1)
    accuracy_colour_feat_2_mean = np.mean(accuracy_colour_feat_2,1)
    accuracy_shape_feat_1_mean = np.mean(accuracy_shape_feat_1,1)
    accuracy_shape_feat_2_mean = np.mean(accuracy_shape_feat_2,1)
    accuracy_colour_conj_mean = np.mean(accuracy_colour_conj,1)
    accuracy_shape_conj_mean = np.mean(accuracy_shape_conj,1)
    accuracy_colour_conj_1_mean = np.mean(accuracy_colour_conj_1,1)
    accuracy_colour_conj_2_mean = np.mean(accuracy_colour_conj_2,1)
    accuracy_shape_conj_1_mean = np.mean(accuracy_shape_conj_1,1)
    accuracy_shape_conj_2_mean = np.mean(accuracy_shape_conj_2,1)
    
    # sterror
    accuracy_colour_feat_std = np.std(accuracy_colour_feat,1)/np.sqrt(n_seeds)
    accuracy_shape_feat_std = np.std(accuracy_shape_feat,1)/np.sqrt(n_seeds)
    accuracy_colour_feat_1_std = np.std(accuracy_colour_feat_1,1)/np.sqrt(n_seeds)
    accuracy_colour_feat_2_std = np.std(accuracy_colour_feat_2,1)/np.sqrt(n_seeds)
    accuracy_shape_feat_1_std = np.std(accuracy_shape_feat_1,1)/np.sqrt(n_seeds)
    accuracy_shape_feat_2_std = np.std(accuracy_shape_feat_2,1)/np.sqrt(n_seeds)
    accuracy_colour_conj_std = np.std(accuracy_colour_conj,1)/np.sqrt(n_seeds)
    accuracy_shape_conj_std = np.std(accuracy_shape_conj,1)/np.sqrt(n_seeds)
    accuracy_colour_conj_1_std = np.std(accuracy_colour_conj_1,1)/np.sqrt(n_seeds)
    accuracy_colour_conj_2_std = np.std(accuracy_colour_conj_2,1)/np.sqrt(n_seeds)
    accuracy_shape_conj_1_std = np.std(accuracy_shape_conj_1,1)/np.sqrt(n_seeds)
    accuracy_shape_conj_2_std = np.std(accuracy_shape_conj_2,1)/np.sqrt(n_seeds)
    
    # average decoding accuracy
    # mean
    accuracy_colour_feat_avg = np.mean(np.mean(accuracy_colour_feat,0))
    accuracy_shape_feat_avg = np.mean(np.mean(accuracy_shape_feat,0))
    accuracy_colour_feat_1_avg = np.mean(np.mean(accuracy_colour_feat_1,0))
    accuracy_colour_feat_2_avg = np.mean(np.mean(accuracy_colour_feat_2,0))
    accuracy_shape_feat_1_avg = np.mean(np.mean(accuracy_shape_feat_1,0))
    accuracy_shape_feat_2_avg = np.mean(np.mean(accuracy_shape_feat_2,0))
    accuracy_colour_conj_avg = np.mean(np.mean(accuracy_colour_conj,0))
    accuracy_shape_conj_avg = np.mean(np.mean(accuracy_shape_conj,0))
    accuracy_colour_conj_1_avg = np.mean(np.mean(accuracy_colour_conj_1,0))
    accuracy_colour_conj_2_avg = np.mean(np.mean(accuracy_colour_conj_2,0))
    accuracy_shape_conj_1_avg = np.mean(np.mean(accuracy_shape_conj_1,0))
    accuracy_shape_conj_2_avg = np.mean(np.mean(accuracy_shape_conj_2,0))
    
    # sterror
    accuracy_colour_feat_avgstd = np.std(np.mean(accuracy_colour_feat,0))/np.sqrt(n_seeds)
    accuracy_colour_feat_avgstd = np.std(np.mean(accuracy_colour_feat,0))/np.sqrt(n_seeds)
    accuracy_colour_feat_1_avgstd = np.std(np.mean(accuracy_colour_feat_1,0))/np.sqrt(n_seeds)
    accuracy_colour_feat_2_avgstd = np.std(np.mean(accuracy_colour_feat_2,0))/np.sqrt(n_seeds)
    accuracy_shape_feat_1_avgstd = np.std(np.mean(accuracy_shape_feat_1,0))/np.sqrt(n_seeds)
    accuracy_shape_feat_2_avgstd = np.std(np.mean(accuracy_shape_feat_2,0))/np.sqrt(n_seeds)
    accuracy_colour_conj_avgstd = np.std(np.mean(accuracy_colour_conj,0))/np.sqrt(n_seeds)
    accuracy_colour_conj_avgstd = np.std(np.mean(accuracy_colour_conj,0))/np.sqrt(n_seeds)
    accuracy_colour_conj_1_avgstd = np.std(np.mean(accuracy_colour_conj_1,0))/np.sqrt(n_seeds)
    accuracy_colour_conj_2_avgstd = np.std(np.mean(accuracy_colour_conj_2,0))/np.sqrt(n_seeds)
    accuracy_shape_conj_1_avgstd = np.std(np.mean(accuracy_shape_conj_1,0))/np.sqrt(n_seeds)
    accuracy_shape_conj_2_avgstd = np.std(np.mean(accuracy_shape_conj_2,0))/np.sqrt(n_seeds)
    
    # %% Decoding TMS
    
    accuracy_colour_feat_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_feat_tms = np.zeros([400,len(tms_vals),n_seeds])
    accuracy_colour_conj_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_conj_tms = np.zeros([400,len(tms_vals),n_seeds])
    accuracy_colour_feat_1_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_feat_1_tms = np.zeros([400,len(tms_vals),n_seeds])
    accuracy_colour_conj_1_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_conj_1_tms = np.zeros([400,len(tms_vals),n_seeds])
    accuracy_colour_feat_2_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_feat_2_tms = np.zeros([400,len(tms_vals),n_seeds])
    accuracy_colour_conj_2_tms = np.zeros([400,len(tms_vals),n_seeds]); accuracy_shape_conj_2_tms = np.zeros([400,len(tms_vals),n_seeds])
    
    prog_counter = 0
    print('Start TMS decoding analysis')
    for rnd in range(n_seeds):
        for i in range(len(tms_vals)):
            accuracy_colour_feat_tms[:,i,rnd], accuracy_shape_feat_tms[:,i,rnd], accuracy_colour_conj_tms[:,i,rnd], accuracy_shape_conj_tms[:,i,rnd], \
            accuracy_colour_feat_1_tms[:,i,rnd], accuracy_shape_feat_1_tms[:,i,rnd], accuracy_colour_conj_1_tms[:,i,rnd], accuracy_shape_conj_1_tms[:,i,rnd], \
            accuracy_colour_feat_2_tms[:,i,rnd], accuracy_shape_feat_2_tms[:,i,rnd], accuracy_colour_conj_2_tms[:,i,rnd], accuracy_shape_conj_2_tms[:,i,rnd] \
            = plasticattractor_decode(num_trials, num_blocks, stim_period, c_labels_corr_tms[rnd,i], s_labels_corr_tms[rnd,i], r_labels_corr_tms[rnd,i], conj_corr_tms[rnd,i], feat_corr_tms[rnd,i])
            prog_counter += 1
            print('TMS decoding progress', (prog_counter/(n_seeds*len(tms_vals)))*100, '%')
        
    # decoding accuracy
    # mean
    accuracy_colour_feat_mean_tms = np.mean(accuracy_colour_feat_tms,2)
    accuracy_shape_feat_mean_tms = np.mean(accuracy_shape_feat_tms,2)
    accuracy_colour_feat_1_mean_tms = np.mean(accuracy_colour_feat_1_tms,2)
    accuracy_colour_feat_2_mean_tms = np.mean(accuracy_colour_feat_2_tms,2)
    accuracy_shape_feat_1_mean_tms = np.mean(accuracy_shape_feat_1_tms,2)
    accuracy_shape_feat_2_mean_tms = np.mean(accuracy_shape_feat_2_tms,2)
    accuracy_colour_conj_mean_tms = np.mean(accuracy_colour_conj_tms,2)
    accuracy_shape_conj_mean_tms = np.mean(accuracy_shape_conj_tms,2)
    accuracy_colour_conj_1_mean_tms = np.mean(accuracy_colour_conj_1_tms,2)
    accuracy_colour_conj_2_mean_tms = np.mean(accuracy_colour_conj_2_tms,2)
    accuracy_shape_conj_1_mean_tms = np.mean(accuracy_shape_conj_1_tms,2)
    accuracy_shape_conj_2_mean_tms = np.mean(accuracy_shape_conj_2_tms,2)
    
    # sterror
    accuracy_colour_feat_std_tms = np.std(accuracy_colour_feat_tms,2)/np.sqrt(n_seeds)
    accuracy_shape_feat_std_tms = np.std(accuracy_shape_feat_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_feat_1_std_tms = np.std(accuracy_colour_feat_1_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_feat_2_std_tms= np.std(accuracy_colour_feat_2_tms,2)/np.sqrt(n_seeds)
    accuracy_shape_feat_1_std_tms = np.std(accuracy_shape_feat_1_tms,2)/np.sqrt(n_seeds)
    accuracy_shape_feat_2_std_tms = np.std(accuracy_shape_feat_2_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_conj_std_tms = np.std(accuracy_colour_conj_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_conj_std_tms = np.std(accuracy_colour_conj_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_conj_1_std_tms = np.std(accuracy_colour_conj_1_tms,2)/np.sqrt(n_seeds)
    accuracy_colour_conj_2_std_tms = np.std(accuracy_colour_conj_2_tms,2)/np.sqrt(n_seeds)
    accuracy_shape_conj_1_std_tms = np.std(accuracy_shape_conj_1_tms,2)/np.sqrt(n_seeds)
    accuracy_shape_conj_2_std_tms = np.std(accuracy_shape_conj_2_tms,2)/np.sqrt(n_seeds)
    
    # average over time
    accuracy_colour_feat_avg_tms = np.mean(accuracy_colour_feat_tms,0)
    accuracy_shape_feat_avg_tms = np.mean(accuracy_shape_feat_tms,0)
    accuracy_colour_feat_1_avg_tms = np.mean(accuracy_colour_feat_1_tms,0)
    accuracy_colour_feat_2_avg_tms = np.mean(accuracy_colour_feat_2_tms,0)
    accuracy_shape_feat_1_avg_tms = np.mean(accuracy_shape_feat_1_tms,0)
    accuracy_shape_feat_2_avg_tms = np.mean(accuracy_shape_feat_2_tms,0)
    accuracy_colour_conj_avg_tms = np.mean(accuracy_colour_conj_tms,0)
    accuracy_shape_conj_avg_tms = np.mean(accuracy_shape_conj_tms,0)
    accuracy_colour_conj_1_avg_tms = np.mean(accuracy_colour_conj_1_tms,0)
    accuracy_colour_conj_2_avg_tms = np.mean(accuracy_colour_conj_2_tms,0)
    accuracy_shape_conj_1_avg_tms = np.mean(accuracy_shape_conj_1_tms,0)
    accuracy_shape_conj_2_avg_tms = np.mean(accuracy_shape_conj_2_tms,0)
    
    # change in mean decoding accuracy as a function of TMS train length
    delta_decode_ConjRel = 100*(.5*(accuracy_colour_conj_1_avg_tms + accuracy_shape_conj_2_avg_tms) - .5*(accuracy_colour_conj_1_avg + accuracy_shape_conj_2_avg))
    delta_decode_ConjIrel = 100*(.5*(accuracy_colour_conj_2_avg_tms + accuracy_shape_conj_1_avg_tms) - .5*(accuracy_colour_conj_2_avg + accuracy_shape_conj_1_avg))
    delta_decode_FeatRel = 100*(.5*(accuracy_colour_feat_1_avg_tms + accuracy_shape_feat_2_avg_tms) - .5*(accuracy_colour_feat_1_avg + accuracy_shape_feat_2_avg))
    delta_decode_FeatIrel = 100*(.5*(accuracy_colour_feat_2_avg_tms + accuracy_shape_feat_1_avg_tms) - .5*(accuracy_colour_feat_2_avg + accuracy_shape_feat_1_avg))
    
    # standard error of mean decoding accuracy change in decoding accuracy
    delta_decode_ConjRel_ster = np.std(delta_decode_ConjRel,1)/np.sqrt(n_seeds)
    delta_decode_ConjIrel_ster = np.std(delta_decode_ConjIrel,1)/np.sqrt(n_seeds)
    delta_decode_FeatRel_ster = np.std(delta_decode_FeatRel,1)/np.sqrt(n_seeds)
    delta_decode_FeatIrel_ster = np.std(delta_decode_FeatIrel,1)/np.sqrt(n_seeds)
    
    # average over subjects
    delta_decode_ConjRel = np.mean(delta_decode_ConjRel,1)
    delta_decode_ConjIrel = np.mean(delta_decode_ConjIrel,1)
    delta_decode_FeatRel = np.mean(delta_decode_FeatRel ,1)
    delta_decode_FeatIrel =np.mean(delta_decode_FeatIrel,1)
    
    # %%  Decoding plots
    
    cmap_1 = mpl.cm.get_cmap('Blues')
    cmap_2 = mpl.cm.get_cmap('Greys')
    time = np.array(range(400))
     
    # decoding all trails
    fig = plt.figure()
    plt.title('Feature unit decoding')
    plt.plot(time,accuracy_colour_feat_mean, color = cmap_1(13/len(tms_vals)))
    plt.plot(time,accuracy_shape_feat_mean, color = cmap_2(13/len(tms_vals)))
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    fig.legend(['Colour','Shape'])
     
    fig = plt.figure()
    plt.title('Conjunction unit decoding')
    plt.plot(time,accuracy_colour_conj_mean, color = cmap_1(13/len(tms_vals)))
    plt.plot(time,accuracy_shape_conj_mean, color = cmap_2(13/len(tms_vals)))
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    fig.legend(['Colour','Shape'])
     
    # decoding sorted by rule
    fig, ax  = plt.subplots()
    plt.title('Feature units')
    plt.plot(time,.5*(accuracy_colour_feat_1_mean+accuracy_shape_feat_2_mean)*100, color = 'crimson', linewidth = 2)
    plt.plot(time,.5*(accuracy_shape_feat_1_mean+accuracy_colour_feat_2_mean)*100, color = 'darkviolet', linewidth = 2)
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Classification accuracy %", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # fig.savefig('PrioritisationPlot_feat.png', format='png', dpi=1200)

    fig, ax  = plt.subplots()
    plt.title('Conjunction units')
    plt.plot(time,.5*(accuracy_colour_conj_1_mean+accuracy_shape_conj_2_mean)*100, color = 'crimson', linewidth = 2)
    plt.plot(time,.5*(accuracy_shape_conj_1_mean+accuracy_colour_conj_2_mean)*100, color = 'darkviolet', linewidth = 2)
    plt.xlabel("Time")
    plt.ylabel("Classification accuracy %")
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Classification accuracy %", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # fig.savefig('PrioritisationPlot_conj.png', format='png', dpi=1200)

    
    # TMS: decoding sorted by rule
    norm = mpl.colors.Normalize(vmin=0, vmax=13)
    cmap1 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('Blues'))
    cmap1.set_array([])
    cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('Greys'))
    cmap2.set_array([])
    
    fig = plt.figure()
    # plt.title('TMS: feature unit decoding accuracy')
    for i in range(13):
        plt.plot(time,.5*(accuracy_shape_feat_1_mean_tms[:,i]+accuracy_colour_feat_2_mean_tms[:,i]), color = cmap_2(i/len(tms_vals)))
        plt.plot(time,.5*(accuracy_colour_feat_1_mean_tms[:,i]+accuracy_shape_feat_2_mean_tms[:,i]), color = cmap_1(i/len(tms_vals)))
    plt.colorbar(cmap1, ticks=np.arange(1,13)).ax.invert_yaxis() 
    plt.colorbar(cmap2, ticks=np.arange(1,13)).ax.invert_yaxis()
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
     
    fig = plt.figure()
    # plt.title('TMS: conjunction unit decoding accuracy')
    for i in range(13):
        plt.plot(time,100*(.5*(accuracy_colour_conj_1_mean_tms[:,i]+accuracy_shape_conj_2_mean_tms[:,i])), color = cmap_1(i/len(tms_vals)))
        plt.plot(time,100*(.5*(accuracy_shape_conj_1_mean_tms[:,i]+accuracy_colour_conj_2_mean_tms[:,i])), color = cmap_2(i/len(tms_vals)))
    plt.colorbar(cmap1, ticks=np.arange(1,13)).ax.invert_yaxis() 
    plt.colorbar(cmap2, ticks=np.arange(1,13)).ax.invert_yaxis() 
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
     
    # Delta decoding plots
    fig = plt.figure()
    plt.plot(tms_vals,delta_decode_ConjRel, color = cmap_1(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_ConjIrel, color = cmap_2(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_FeatRel,'--', color = cmap_1(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_FeatIrel,'--', color = cmap_2(13/len(tms_vals)))
    plt.xlabel("TMS train length")
    plt.ylabel("Change in decoding accuracy")
    fig.legend(['Conj - relevant','Conj - irrelevant', 'Feat - relevant','Feat - irrelevant', ])
    
    # Delta decoding plots just conj
    chance_threshold = np.zeros_like(tms_vals)
        
    fig = plt.figure()
    plt.errorbar(tms_vals,delta_decode_ConjRel,yerr = delta_decode_ConjRel_ster, color = cmap_1(13/len(tms_vals)),linestyle="-",marker="o", capsize = 3)
    plt.errorbar(tms_vals,delta_decode_ConjIrel,yerr = delta_decode_ConjIrel_ster, color = cmap_2(13/len(tms_vals)),linestyle="-",marker="o", capsize = 3)
    plt.xlabel("TMS train length")
    plt.ylabel("Change in decoding accuracy %")
    fig.legend(['Conjunction relevant','Conjunction irrelevant'])
    plt.plot(tms_vals,chance_threshold, 'k--')
    plt.xticks(np.arange(0,20,1))
    plt.ylim([-45,10])
    # fig.savefig('DeltaTMS_conj.png', format='png', dpi=1200)
    
    
    # Delta decoding plots just feature
    fig = plt.figure()
    plt.errorbar(tms_vals,delta_decode_FeatRel,yerr = delta_decode_FeatRel_ster,color = cmap_1(13/len(tms_vals)), linestyle="-",marker="o", capsize = 3)
    plt.errorbar(tms_vals,delta_decode_FeatIrel,yerr = delta_decode_FeatIrel_ster,color = cmap_2(13/len(tms_vals)), linestyle="-",marker="o", capsize = 3)
    plt.xlabel("TMS train length")
    plt.ylabel("Change in decoding accuracy %")
    fig.legend(['Feature relevant','Feature irrelevant'])
    plt.plot(tms_vals,chance_threshold, 'k--')
    plt.xticks(np.arange(0,20,1))
    plt.ylim([-45,10])    
    # fig.savefig('DeltaTMS_feat.png', format='png', dpi=1200)
  
    
    # Delta decoding plots as a function normalised by peak decoding accuracy
    fig = plt.figure()
    plt.plot(tms_vals,delta_decode_ConjRel/(.5*(accuracy_colour_conj_1_avg + accuracy_shape_conj_2_avg)), color = cmap_1(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_ConjIrel/(.5*(accuracy_colour_conj_2_avg + accuracy_shape_conj_1_avg)), color = cmap_2(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_FeatRel/(.5*(accuracy_colour_feat_1_avg + accuracy_shape_feat_2_avg)),'--', color = cmap_1(13/len(tms_vals)))
    plt.plot(tms_vals,delta_decode_FeatIrel/(.5*(accuracy_colour_feat_2_avg + accuracy_shape_feat_1_avg)),'--', color = cmap_2(13/len(tms_vals)))
    plt.xlabel("TMS train length")
    plt.ylabel("Change in decoding accuracy")
    fig.legend(['Conj - relevant','Conj - irrelevant', 'Feat - relevant','Feat - irrelevant', ])
    


