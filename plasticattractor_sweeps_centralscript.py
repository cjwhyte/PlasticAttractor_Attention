"""
Central script for the plastic attractor parameter sensitivity and scalability
analyses

This script calls the relevant functions to produce the analyses requested by
Reviewer 2 (comments 3 and 4). Set the flags below to select which sweeps to
run; each cell can also be run on its own.

Every sweep checkpoints to sweep_results/_cache_<name>.npz every 200 runs, so
an interrupted run can be restarted by simply running the cell again, and it
will resume rather than start over.

@author: Christopher Whyte

"""""

import os
import numpy as np
import matplotlib

# Display figures as well as writing them to disk. Set to 1 when working
# interactively (VS Code interactive window, Spyder, Jupyter): figures appear
# in the plot pane and are still saved to outdir. Set to 0 for an unattended
# batch run, which selects a non-interactive backend so the sweeps do not
# block waiting for a display.
#
# This has to be decided before pyplot is imported anywhere, which is why it
# sits above the imports rather than down in the settings cell.
show_figures = 1

if not show_figures:
    matplotlib.use('Agg')

# sweep functions
from plasticattractor_sweeps import (sweep_dose, sweep_marginal, sweep_grid,
                                     sweep_dimensions, sweep_noise)
# configuration builder. Every sweep uses one of these internally; calling the
# same builder here gives the exact list of settings that sweep ran, which is
# what the example activity blocks are generated from. Builders for the other
# sweeps (configs_dose, configs_marginal, configs_grid, configs_noise) are
# available in the sweeps module if example blocks are ever wanted for them.
from plasticattractor_sweeps import (configs_marginal,
                                     configs_dimensions,
                                     configs_noise)
# figures
from plasticattractor_sweeps import (plot_dose, plot_marginal_compare,
                                     plot_seizure_boundary, plot_grid,
                                     plot_dimensions, plot_noise,
                                     plot_example_blocks)

# %% settings

# which sweeps to run (set to 1 to run). Start with the dose response: it is
# the validation test for the simulation code
dose_response = 1
marginal_sensitivity = 0
alpha_grid = 0
dimensions = 0
noise = 0

# Produce figures showing an example block of network activity at each task
# size in the capacity sweep, so the summary numbers can be checked against
# what the network is actually doing. 
example_blocks = 12

# number of blocks per run
num_blocks = 20

# blocks for the dimensionality sweep. Rules cycle over the dimensions, so
# this has to divide by 2, 3 and 4 for the rules to be used equally often at
# every D in the sweep.
num_blocks_dims = 24

# trials per block (must be a multiple of the number of stimulus combinations)
num_trials = 12

# number of random seeds per setting
n_seeds = 40

# parallel workers (-1 uses every core)
n_jobs = -1

# TMS dose at which the sensitivity sweeps are run. This is the dose at which
# the constellation of effects reported by Jackson et al. (2021) is
# reproduced, and where the amplifying eigenvalues begin to fall.
operating_dose = 11

# where figures and .npz files are written
outdir = 'sweep_results'
os.makedirs(outdir, exist_ok=True)

# %% TMS dose response
#
# Reproduces Figures 4 and 5. Run this first: it confirms that the refactored
# simulation still gives the published pattern (accuracy at ceiling out to a
# dose of about 11, collapse over 12-14, amplifying eigenvalues falling from 2
# to 1 in step with accuracy, congruency effect shrinking), and it locates the
# operating dose used everywhere below.

if dose_response:

    dose_list, dose_summ = sweep_dose(dose_list=list(range(0, 20)),
                                      num_blocks=num_blocks,
                                      num_trials=num_trials,
                                      n_seeds=n_seeds,
                                      n_jobs=n_jobs,
                                      name='dose_response')

    plot_dose(dose_list=dose_list, summ=dose_summ,
              fname=os.path.join(outdir, 'dose_response.png'),
              show=show_figures,
              title='TMS dose response  |  publication parameters, '
                    '%d blocks, %d seeds' % (num_blocks, n_seeds))


    # the legacy path returns LEGACY_METRICS, which is behaviour plus the
    # eigenvalue count; there is no seizure index or prioritisation here,
    # because the original analysis function does not compute them
    print('accuracy by dose:   ' + str(np.round(dose_summ['accuracy'], 3)))
    print('eigenvalues > 1:    ' + str(np.round(dose_summ['mean_eig'], 2)))
    print('congruency effect:  '
          + str(np.round(dose_summ['congruency_effect'], 1)))

# %% Reviewer 2, comment 3: marginal parameter sweeps
#
# Each parameter varied alone over +/- 25% of its published value, with the
# others held fixed. The range is deliberately narrow: 11 points over +/- 25%
# resolves alpha1 to steps of 0.0225, which is fine enough to locate the
# seizure boundary at alpha1 = -0.5 (a factor of 1.111, so still inside the
# range). Run at two TMS doses, because two different ranges are
# being conflated in the reviewer's comment: the range over which the model
# performs the TASK (dose 0, expected to be broad) and the range over which it
# RECOVERS FROM PERTURBATION (the operating dose, narrower, and the only one
# that was ever hand tuned).

if marginal_sensitivity:

    # no TMS: does the model do the task
    lab_task, val_task, fac_task, summ_task = sweep_marginal(
        param_names=None,          # None uses SWEEP_PARAMS: alpha1-6, beta
        rel_range=.25,
        n_points=11,
        tms_dose=0,
        num_blocks=num_blocks,
        num_trials=num_trials,
        n_seeds=n_seeds,
        n_jobs=n_jobs,
        name='marginal_tms0')

    # operating dose: does the model recover from perturbation
    lab_tms, val_tms, fac_tms, summ_tms = sweep_marginal(
        param_names=None,          # as above
        rel_range=.25,
        n_points=11,
        tms_dose=operating_dose,
        num_blocks=num_blocks,
        num_trials=num_trials,
        n_seeds=n_seeds,
        n_jobs=n_jobs,
        name='marginal_tms%d' % operating_dose)

    # overlay both doses on the same axes: this is the figure that carries the
    # argument, since plotting only one invites the reviewer to keep reading
    # the two ranges as the same thing
    for measure in ['accuracy', 'congruency_effect', 'prioritisation']:
        plot_marginal_compare(
            summ_task=summ_task,
            summ_tms=summ_tms,
            labels=lab_task,
            factors=fac_task,
            measure=measure,
            fname=os.path.join(outdir, 'marginal_%s.png' % measure),
            show=show_figures,
            title='Marginal sensitivity: %s  |  +/-25%%, %d points, '
                  'TMS dose 0 vs %d, %d blocks, %d seeds'
                  % (measure, 11, operating_dose, num_blocks, n_seeds))

    # quantified seizure boundary for the one parameter whose hand tuning the
    # reviewer singled out
    plot_seizure_boundary(
        labels=lab_tms,
        values=val_tms,
        factors=fac_tms,
        summ=summ_tms,
        param='alpha1',
        fname=os.path.join(outdir, 'seizure_boundary_alpha1.png'),
        show=show_figures,
        title='Seizure boundary in alpha1  |  TMS dose %d, %d blocks, '
              '%d seeds'
              % (operating_dose, num_blocks, n_seeds))

    plot_seizure_boundary(
        labels=lab_tms,
        values=val_tms,
        factors=fac_tms,
        summ=summ_tms,
        param='alpha2',
        fname=os.path.join(outdir, 'seizure_boundary_alpha2.png'),
        show=show_figures,
        title='Seizure boundary in alpha2  |  TMS dose %d, %d blocks, '
              '%d seeds' % (operating_dose, num_blocks, n_seeds))

    # Example activity across each parameter, at the operating dose. One
    # figure per parameter so the panels stay legible: three stacked rows per
    # setting (motor units, conjunction units, sensory feature heatmap).
    if example_blocks:
        for pname in ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5',
                      'alpha6', 'beta']:
            cfgs, _ = configs_marginal(param_names=[pname],
                                       rel_range=.25,
                                       n_points=11,
                                       tms_dose=operating_dose,
                                       num_blocks=num_blocks,
                                       num_trials=num_trials)
            plot_example_blocks(
                configs=cfgs,
                max_configs=5,
                num_blocks=num_blocks,
                # last block, once the long term weights have settled
                blk=num_blocks - 1,
                n_trials_shown=6,
                fname=os.path.join(outdir,
                                   'example_marginal_%s.png' % pname),
                show=show_figures,
                title='Example activity across %s, TMS dose %d'
                      % (pname, operating_dose))

# %% Reviewer 2, comment 3: the alpha1 x alpha5 plane
#
# The two parameters actually adjusted by hand relative to Manohar et al.
# (2019): alpha1 (conjunction lateral inhibition, -0.5 -> -0.45) and alpha5
# (feature self excitation, 0.7 -> 0.73). Run at the operating dose, since
# that is the regime the tuning was for.
#
# This grid is also the joint perturbation analysis: it varies two parameters
# simultaneously rather than one at a time, and they are the two whose values
# we set by hand. Broader joint sampling of the remaining parameters is
# covered by the sweep reported in Manohar et al. (2019).

if alpha_grid:

    a_vals, b_vals, grid_summ = sweep_grid(
        param_a='alpha1',
        param_b='alpha5',
        rel_range=.25,
        n_points=13,
        tms_dose=operating_dose,
        num_blocks=num_blocks,
        num_trials=num_trials,
        n_seeds=n_seeds,
        n_jobs=n_jobs,
        name='grid_alpha1_alpha5')

    plot_grid(a_vals=a_vals, b_vals=b_vals, summ=grid_summ,
              measure='accuracy', param_a='alpha1', param_b='alpha5',
              fname=os.path.join(outdir, 'grid_alpha1_alpha5_accuracy.png'),
              show=show_figures,
              title='alpha1 x alpha5, accuracy  |  +/-25%%, 13 x 13, '
                    'TMS dose %d, %d blocks, %d seeds'
                    % (operating_dose, num_blocks, n_seeds))

    plot_grid(a_vals=a_vals, b_vals=b_vals, summ=grid_summ,
              measure='seizure_index', param_a='alpha1', param_b='alpha5',
              fname=os.path.join(outdir, 'grid_alpha1_alpha5_seizure.png'),
              show=show_figures,
              title='alpha1 x alpha5, seizure index  |  +/-25%%, 13 x 13, '
                    'TMS dose %d, %d blocks, %d seeds'
                    % (operating_dose, num_blocks, n_seeds))

# %% Reviewer 2, comment 4: higher dimensional stimulus spaces
#
# How does the model scale as stimulus dimensions are added? The network is
# held at its published size throughout -- 4 conjunction units, two features
# per dimension, two responses -- and only the number of dimensions grows.
# One dimension is relevant per rule and the rest are ignored, so a rule
# always contains two mappings and chance stays at 0.50 whatever D is. Raw
# accuracy is therefore directly comparable across D.
#
# The network is deliberately NOT rescaled. It is non-linear, so its
# parameters do not scale linearly with population size.
#
# THE ONE CHANGE IS irrel_drive, REDUCED FROM THE PUBLISHED 0.5 TO 0.25.
# Note this is a property of how the RULE IS PRESENTED rather than of the
# network's dynamics, which is why it is a defensible change: it is the
# analogue of how salient the irrelevant dimension is in the instruction.

if dimensions:

    dims, dim_summ = sweep_dimensions(
        dims_list=(2, 3, 4),
        irrel_drive=.25,
        num_conjunction=4,
        tms_dose=0,
        num_blocks=num_blocks_dims,
        num_trials=num_trials,
        n_seeds=n_seeds,
        n_jobs=n_jobs,
        name='dimensions')

    for measure in ['accuracy', 'prioritisation', 'congruency_effect',
                    'mean_eig']:
        plot_dimensions(dims=dims, summ=dim_summ,
                        measure=measure,
                        fname=os.path.join(outdir,
                                           'dimensions_%s.png' % measure),
                        show=show_figures,
                        title='%s by stimulus dimensionality  |  4 '
                              'conjunction units, irrel_drive 0.25, no TMS, '
                              '%d blocks, %d seeds'
                              % (measure, num_blocks_dims, n_seeds))

    for i, nd in enumerate(dims):
        print('D = %d: accuracy %.3f, prioritisation %+.3f, '
              'eigenvalues %.2f, congruency %.1f'
              % (nd, dim_summ['accuracy'][i], dim_summ['prioritisation'][i],
                 dim_summ['mean_eig'][i], dim_summ['congruency_effect'][i]))

    # example activity at each dimensionality, one row per D
    if example_blocks:
        plot_example_blocks(
            configs=configs_dimensions(dims_list=(2, 3, 4),
                                       irrel_drive=.25,
                                       num_conjunction=4,
                                       tms_dose=0,
                                       num_blocks=num_blocks_dims,
                                       num_trials=num_trials),
            max_configs=3,
            num_blocks=num_blocks_dims,
            # last block, once the long term weights have settled
            blk=num_blocks_dims - 1,
            n_trials_shown=6,
            fname=os.path.join(outdir, 'example_dimensions.png'),
            show=show_figures,
            title='Example activity by stimulus dimensionality  |  4 '
                  'conjunction units, irrel_drive 0.25')

# %% Reviewer 2, comment 4: noise sensitivity
#
# Noise scale at the publication network size, spanning a 40-fold range above
# the publication value of 0.005. 

if noise:

    eps_vals, noise_summ = sweep_noise(
        eps_list=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        tms_dose=0,
        num_blocks=num_blocks,
        num_trials=num_trials,
        n_seeds=n_seeds,
        n_jobs=n_jobs,
        name='noise')

    # accuracy, prioritisation and the size of the congruency effect
    for measure in ['accuracy', 'prioritisation', 'congruency_effect']:
        plot_noise(eps_vals=eps_vals, summ=noise_summ,
                   measure=measure,
                   fname=os.path.join(outdir, 'noise_%s.png' % measure),
                   show=show_figures,
                   title='%s by noise scale  |  publication network, no TMS, '
                         '%d blocks, %d seeds'
                         % (measure, num_blocks, n_seeds))

    for i, ev in enumerate(eps_vals):
        print('epsilon = %.3f: accuracy %.3f, prioritisation %+.3f, '
              'congruency %.1f'
              % (ev, noise_summ['accuracy'][i],
                 noise_summ['prioritisation'][i],
                 noise_summ['congruency_effect'][i]))

    # example activity at each noise scale
    if example_blocks:
        plot_example_blocks(
            configs=configs_noise(eps_list=(0.005, 0.01, 0.02, 0.05, 0.1,
                                            0.2),
                                  tms_dose=0,
                                  num_blocks=num_blocks,
                                  num_trials=num_trials),
            max_configs=6,
            num_blocks=num_blocks,
            blk=num_blocks - 1,
            n_trials_shown=6,
            fname=os.path.join(outdir, 'example_noise.png'),
            show=show_figures,
            title='Example activity by noise scale')
