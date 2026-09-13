"""
Plastic attractor network simulation script -- generalised implementation

Refactor of plasticattractor_networksims.py written to support the parameter
sensitivity (Reviewer 2, comment 3) and scalability (Reviewer 2, comment 4)
analyses. The dynamics are unchanged.

Summary of changes relative to the original script:

    1. TASK IS NOW BUILT PROGRAMMATICALLY. The original hard coded a 6 feature
       unit network (2 colours, 2 shapes, 2 responses) with a 4 x 4 stimulus
       array and a 2 x 6 x 2 rule array. Here the task is specified by the
       number of stimulus dimensions, the number of features per dimension, and
       the number of responses, and the rule vectors, stimulus vectors, and the
       block structure of W_ff are all constructed from that specification.

    2. FREE PARAMETERS ARE PASSED IN A DICTIONARY. alpha1-alpha6, beta,
       gamma_short, gamma_long, epsilon and the weight bounds can all be swept.
       NOTE ON NAMING: the parameter names here follow the code, i.e. alpha3 is
       the conjunction -> feature gain (0.08) and alpha6 is the feature ->
       conjunction gain (0.04). 

    3. QUANTITATIVE OUTCOME MEASURES ARE COMPUTED IN-SIMULATION. Accuracy, RT,
       the congruency effect, the relevant vs irrelevant prioritisation index,
       the number of amplifying eigenvalues, and a quantitative seizure-like
       dynamics index (see seizure_index below) are all returned, so that a
       sweep does not have to hold on to the full activity arrays.

    4. TMS IS SPECIFIED AS A DOSE. The original took a TMS_start timestep and
       computed the dose implicitly as (100 - TMS_start). Here tms_dose is the
       number of timesteps of maximal conjunction unit activation ending at
       stimulus offset, which is the quantity plotted in Figures 4 and 5.

    5. A DEGENERATE RESPONSE GUARD. If the response units never rise above zero
       (which can happen in the far corners of the parameter space) the
       original would raise an IndexError on the empty argwhere. Such trials
       are now scored as incorrect with an undefined RT.

@author: Christopher Whyte
"""

import numpy as np


# %% helpers for the seizure criterion

def _lag1_corr(x):
    """
    Lag-1 autocorrelation of a timeseries. Near +1 for smooth persistent
    activity, near -1 when the signal alternates on successive timesteps.
    Returns 0 for a constant signal, which is neither.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.
    a, b = x[:-1], x[1:]
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))


def _mean_pairwise_corr(x):
    """
    Mean correlation between the rows of x (units x time). Near 1 when the
    population moves as one, low when a single unit has won the competition.
    Units with no variance are dropped.
    """
    x = np.asarray(x, dtype=float)
    keep = np.std(x, axis=1) > 1e-12
    if np.sum(keep) < 2:
        return 0.
    c = np.corrcoef(x[keep])
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else 0.


# %% default parameter and task specifications

def default_params():
    """Free parameters of the network, at the values used in the manuscript."""

    params = {}

    # col1 = conj, col2 = feat
    # alpha1, alpha4 = mutual lateral inhibition
    # alpha2, alpha5 = self excitation / temporal decay
    # alpha3, alpha6 = synaptic gain
    params['alpha'] = np.array([-.45, 1., .08, -.28, .73, .04])

    # baseline activity/lateral inhibition threshold
    params['beta'] = .175

    # learning rates for the two timescales of hebbian plasticity
    params['gamma_short'] = .02
    params['gamma_long'] = .0002

    # upper bounds on the two weight matrices
    params['w_short_max'] = 1.
    params['w_long_max'] = .2

    # scale of the gaussian noise injected into the conjunction units
    params['epsilon'] = .005

    return params


def default_task():
    """Task specification. The defaults reproduce the manuscript task."""

    task = {}

    # number of stimulus dimensions (e.g. colour, shape) and the number of
    # features within each dimension (e.g. green, blue). Every stimulus
    # dimension has the same number of features.
    task['num_dims'] = 2
    task['num_features_per_dim'] = 2

    # number of motor response units. Must be >= num_features_per_dim so that
    # every feature of the relevant dimension can be given its own response.
    task['num_responses'] = 2

    # number of conjunction units
    task['num_conjunction'] = 4

    # input to the features of the irrelevant dimension(s) during the rule
    # period (the partial coactivation, i_ext = +0.5, of the manuscript)
    task['irrel_drive'] = .5

    # trial timing (in timesteps). Both the rule epochs and the stimulus trials
    # last trial_period timesteps.
    task['trial_period'] = 400
    task['rule_on'] = 250        # rule presented for t <= rule_on, then isi
    task['stim_on'] = 50         # isi for t <= stim_on
    task['stim_off'] = 100       # stimulus presented for stim_on < t <= stim_off
    task['resp_off'] = 350       # response period, then isi to trial_period
    task['resp_pad'] = 10        # RT is read out from stim_off + resp_pad

    # fraction of stimulus trials that may show synchronous period-2
    # oscillation of the conjunction population before a parameter setting is
    # called seizure-like (see the seizure criterion in the trial loop)
    task['seizure_thresh'] = .05

    return task


def build_task(task):
    """
    Expand a task specification into the arrays needed by the simulation.

    Returns a dictionary containing the rule input vectors, the stimulus input
    vectors, the per dimension index slices, the congruency of each stimulus,
    and the ground truth response for each stimulus under each rule.
    """

    num_dims = task['num_dims']
    k = task['num_features_per_dim']
    num_responses = task['num_responses']

    # feature units are laid out as [dim0 features, dim1 features, ...,
    # responses], exactly as in the original 6 unit network
    num_stim_features = num_dims * k
    num_features = num_stim_features + num_responses

    # index slices for each stimulus dimension and for the response units
    dim_slices = [slice(d * k, (d + 1) * k) for d in range(num_dims)]
    resp_slice = slice(num_stim_features, num_features)

    # mapping from feature index within the relevant dimension to response
    # index. With num_responses == k this is the identity, i.e. the first
    # feature maps to the first response, generalising "green -> button 1,
    # blue -> button 2". With num_responses < k it wraps, so several features
    # share a response and the task becomes a categorisation rather than a
    # one-to-one stimulus-response mapping. That second case is useful for
    # separating the number of mappings from the size of the response set in
    # the capacity analysis, but note that it changes what the task is.
    response_map = np.arange(k) % num_responses

    # ---- rules -----------------------------------------------------------
    # For each rule (i.e. for each choice of relevant dimension) there are k
    # stimulus-response mappings, each presented as its own rule epoch. Within
    # a mapping the relevant feature and its response are driven to +1, the
    # other features of the relevant dimension and the other responses to -1,
    # and every feature of every irrelevant dimension to irrel_drive (+0.5).
    # For the default task this reproduces [1, -1, .5, .5, 1, -1] etc.
    rules = np.zeros([num_dims, k, num_features])
    for rule_dim in range(num_dims):
        for m in range(k):
            vec = np.zeros(num_features)
            for d in range(num_dims):
                if d == rule_dim:
                    vec[dim_slices[d]] = -1.
                    vec[d * k + m] = 1.
                else:
                    vec[dim_slices[d]] = task['irrel_drive']
            vec[resp_slice] = -1.
            vec[num_stim_features + response_map[m]] = 1.
            rules[rule_dim, m, :] = vec

    # ---- stimulus space --------------------------------------------------
    # A stimulus is one feature from each dimension, so the space has
    # k ** num_dims members. That is fine at the published 2 x 2 task (four
    # stimuli) but explodes once dimensions and features grow together: at
    # k = D = 6 it is 46656. Stimuli are therefore SAMPLED per trial in
    # stim_generator rather than enumerated here.
    #
    # Nothing about the task needs the full factorial. Testing "respond to the
    # relevant feature, ignore the rest" requires each relevant feature to be
    # paired with VARIED irrelevant features, not with all of them. The
    # sampler balances the relevant feature across trials and draws the
    # irrelevant ones at random, which also keeps trials per block a number we
    # choose rather than one dictated by k ** D.
    num_stimuli = k ** num_dims

    # ---- fixed weight matrices -------------------------------------------
    # feature -> feature. Blocks of ones down the diagonal so that units only
    # compete within their own dimension (and responses compete with
    # responses).
    blocks = np.zeros([num_features, num_features])
    for sl in dim_slices + [resp_slice]:
        blocks[sl, sl] = 1.

    built = {}
    built['num_features'] = num_features
    built['num_stim_features'] = num_stim_features
    built['num_stimuli'] = num_stimuli   # size of the space, not enumerated
    built['num_rule_epochs'] = k
    built['dim_slices'] = dim_slices
    built['resp_slice'] = resp_slice
    built['response_map'] = response_map
    built['rules'] = rules
    built['ff_blocks'] = blocks

    return built


# %% stimulus generation

def stim_generator(rule_dim, num_trials, task, built, rng):
    """
    Build the input timeseries for one block: k rule epochs followed by
    num_trials sampled stimulus trials.

    STIMULI ARE SAMPLED, NOT ENUMERATED. The stimulus space has k ** num_dims
    members, which is four at the published task but 46656 once k and num_dims
    both reach six. Presenting all of them would make trials per block a
    function of task size rather than something we control. Instead:

        - the feature on the RELEVANT dimension is balanced: each of the k
          features appears num_trials / k times, in shuffled order, so every
          stimulus-response mapping is tested equally often
        - the features on the IRRELEVANT dimensions are drawn per trial, half
          the trials congruent (every dimension carrying the feature that maps
          to the same response as the relevant one) and half incongruent

    Forcing the congruent/incongruent split to 50/50 keeps the congruency
    effect measurable at any task size. Left to chance it would not be: with D
    dimensions of k features the probability that all agree is k^-(D-1), which
    is one trial in 7776 at k = D = 6, so the congruency measure would simply
    vanish as the task grew.

    Returns the input array (trials x features x time), the ground truth
    response per trial (-1 for rule epochs, which are never scored), the
    feature index chosen on each dimension per trial, and a congruency flag
    per trial.
    """

    num_features = built['num_features']
    num_rule_epochs = built['num_rule_epochs']
    num_dims = task['num_dims']
    k = task['num_features_per_dim']
    response_map = built['response_map']
    trial_period = task['trial_period']

    assert num_trials % (2 * k) == 0, \
        'num_trials must be a multiple of 2 * num_features_per_dim so that ' \
        'relevant feature and congruency can be balanced jointly'

    num_total = num_rule_epochs + num_trials
    inputs = np.zeros([num_total, num_features, trial_period])
    truth = -np.ones(num_total, dtype=int)
    stim_feats = -np.ones([num_total, num_dims], dtype=int)
    congruent = np.zeros(num_total, dtype=bool)

    # isi drives every feature unit to -1, as in the original
    isi = -np.ones(num_features)

    # ---- rule period -----------------------------------------------------
    # one epoch per stimulus-response mapping, in shuffled order
    rule_order = rng.permutation(num_rule_epochs)
    for e in range(num_rule_epochs):
        rule_input = np.zeros([num_features, trial_period])
        # rule presented, then isi (boundaries follow the original script)
        rule_input[:, :task['rule_on'] + 1] = \
            built['rules'][rule_dim, rule_order[e], :][:, None]
        rule_input[:, task['rule_on'] + 1:] = isi[:, None]
        inputs[e, :, :] = rule_input

    # ---- stimulus period -------------------------------------------------
    # Relevant feature and congruency are balanced JOINTLY, not independently:
    # each of the 2k (feature, congruent) cells appears num_trials / 2k times.
    # Balancing them separately would leave the cells unequal by chance, and
    # at the published 2 x 2 task the joint version reproduces the original
    # design exactly -- three presentations of each of the four stimuli.
    cells = [(f, con) for f in range(k) for con in (True, False)]
    order = np.tile(np.arange(len(cells)), num_trials // len(cells))
    rng.shuffle(order)
    rel_order = np.array([cells[i][0] for i in order])
    con_order = np.array([cells[i][1] for i in order])

    for trl in range(num_trials):

        rel_feat = rel_order[trl]
        want_congruent = con_order[trl]

        # choose the feature on every dimension
        feats = np.zeros(num_dims, dtype=int)
        for d in range(num_dims):
            if d == rule_dim:
                feats[d] = rel_feat
            elif want_congruent:
                # same feature index, and so the same mapped response
                feats[d] = rel_feat
            else:
                # any feature mapping to a different response. Falls back to a
                # free draw if the response set is too small for a mismatch to
                # exist (num_responses = 1), which never happens in practice.
                options = [f for f in range(k)
                           if response_map[f] != response_map[rel_feat]]
                feats[d] = rng.choice(options) if options \
                    else rng.integers(k)

        # a trial counts as congruent only if every dimension agrees on the
        # response, which is what was asked for but is worth recomputing
        # rather than assuming
        is_con = len(set(response_map[feats])) == 1

        # build the input vector: active features at +1, everything else 0
        stim_vec = np.zeros(num_features)
        for d in range(num_dims):
            stim_vec[d * k + feats[d]] = 1.

        stim_input = np.zeros([num_features, trial_period])
        # isi -> stimulus -> response period (zero input) -> isi
        stim_input[:, :task['stim_on'] + 1] = isi[:, None]
        stim_input[:, task['stim_on'] + 1:task['stim_off'] + 1] = \
            stim_vec[:, None]
        stim_input[:, task['stim_off'] + 1:task['resp_off'] + 1] = 0.
        stim_input[:, task['resp_off'] + 1:] = isi[:, None]

        inputs[num_rule_epochs + trl, :, :] = stim_input
        truth[num_rule_epochs + trl] = response_map[rel_feat]
        stim_feats[num_rule_epochs + trl, :] = feats
        congruent[num_rule_epochs + trl] = is_con

    return inputs, truth, stim_feats, congruent


# %% main simulation

def plasticattractor_sim(num_blocks, num_trials, rnd_seed, tms_dose=0,
                         params=None, task=None, tms_trial_frac=1.0,
                         freeze_plasticity=False, return_activity=False):
    """
    Simulate the plastic attractor network on the rule based selective
    attention task.

    Arguments
    ---------
    num_blocks       : number of blocks (rules alternate across blocks)
    num_trials       : stimulus trials per block, must divide by num_stimuli
    rnd_seed         : seed for this run
    tms_dose         : number of timesteps of maximal conjunction unit
                       activation ending at stimulus offset (0 = no TMS)
    params           : dict from default_params(), or None for defaults
    task             : dict from default_task(), or None for defaults
    tms_trial_frac   : fraction of stimulus trials receiving TMS (1 = every
                       trial, as in the main analysis; 0.5 reproduces the
                       interleaved simulation of Figure S3)
    freeze_plasticity: if True, weights stop updating after the rule epochs
                       (the Figure S4 control)
    return_activity  : if True, also return the full conjunction and feature
                       unit activity. Off by default so that sweeps do not
                       accumulate large arrays.

    Returns a dictionary of behaviour, weights and summary measures.
    """

    if params is None:
        params = default_params()
    if task is None:
        task = default_task()

    rng = np.random.default_rng(rnd_seed)

    built = build_task(task)
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']

    num_features = built['num_features']
    num_conjunction = task['num_conjunction']
    num_rule_epochs = built['num_rule_epochs']
    num_total = num_rule_epochs + num_trials
    trial_period = task['trial_period']
    resp_slice = built['resp_slice']

    # ---- fixed weight matrices -------------------------------------------
    # feature -> feature: self excitation on the diagonal plus lateral
    # inhibition within each dimension
    W_ff = alpha[4] * np.eye(num_features) + alpha[3] * built['ff_blocks']

    # conjunction -> conjunction: alpha2 + alpha1 on the diagonal, alpha1 off
    # diagonal, giving blanket competition above beta and self excitation
    # below. 
    W_cc = alpha[0] * np.ones(num_conjunction) \
        + alpha[1] * np.eye(num_conjunction)

    # ---- plastic weights -------------------------------------------------
    # initialised once and carried across blocks, as in the original: the long
    # term weights are what stabilise conjunction unit selectivity over blocks
    w_short = rng.random((num_features, num_conjunction))
    w_long = rng.random((num_features, num_conjunction))
    W = w_short + w_long

    # rule for each block, alternating across the stimulus dimensions
    rule_set = np.tile(np.arange(task['num_dims']),
                       int(np.ceil(num_blocks / task['num_dims'])))[:num_blocks]

    # ---- storage ---------------------------------------------------------
    choice = np.full([num_blocks, num_total], -1, dtype=int)
    rt = np.full([num_blocks, num_total], np.nan)
    accuracy = np.zeros([num_blocks, num_total])
    stim_labels = np.full([num_blocks, num_total, task['num_dims']], -1,
                          dtype=int)
    con_labels = np.zeros([num_blocks, num_total], dtype=bool)
    rule_labels = np.full([num_blocks, num_total], -1, dtype=int)
    tms_labels = np.zeros([num_blocks, num_total], dtype=bool)
    n_eig = np.zeros(num_blocks)
    seizure_frac = np.zeros([num_blocks, num_total])
    sat_frac = np.zeros([num_blocks, num_total])
    osc_index = np.zeros([num_blocks, num_total])
    sync_index = np.zeros([num_blocks, num_total])
    rel_index = np.full([num_blocks, num_total], np.nan)
    weight_store = np.zeros([num_blocks, num_features, num_conjunction])

    if return_activity:
        conj_store = np.zeros([num_blocks, num_conjunction, trial_period, num_total])
        feat_store = np.zeros([num_blocks, num_features, trial_period, num_total])

    # TMS is applied over the last tms_dose timesteps of stimulus presentation
    tms_start = task['stim_off'] - tms_dose

    for blk in range(num_blocks):

        rule_dim = int(rule_set[blk])

        # call stimulus function
        inputs, truth, stim_feats, con_flags = stim_generator(
            rule_dim, num_trials, task, built, rng)

        # which stimulus trials receive TMS
        tms_on_trial = np.zeros(num_total, dtype=bool)
        if tms_dose > 0:
            n_tms = int(round(tms_trial_frac * num_trials))
            pick = rng.permutation(num_trials)[:n_tms]
            tms_on_trial[num_rule_epochs + pick] = True

        for trl in range(num_total):

            stimulus = inputs[trl]

            # network state at t-1. Explicitly zero at trial onset rather than
            # relying on the wrap around of the t-1 index (see header).
            f_prev = np.zeros(num_features)
            c_prev = np.zeros(num_conjunction)

            f_trace = np.zeros([num_features, trial_period])
            c_trace = np.zeros([num_conjunction, trial_period])

            for t in range(trial_period):

                # feature neuron
                f = beta + W_ff @ (f_prev - beta) \
                    + alpha[2] * W @ (c_prev - beta) + stimulus[:, t]

                # apply non-linearity to feature neuron
                f = np.maximum(0, np.minimum(1, f))

                # conjunction neuron
                if tms_on_trial[trl] and tms_start <= t <= task['stim_off']:
                    # simulated TMS: maximal activation of all conjunction units
                    c = np.ones(num_conjunction)
                else:
                    c = beta + W_cc @ (c_prev - beta) \
                        + alpha[5] * W.T @ (f_prev - beta) \
                        + epsilon * rng.standard_normal(num_conjunction)

                # apply non-linearity to conjunction neuron
                c = np.maximum(0, np.minimum(1, c))

                # weight update. Frozen after the rule epochs if requested.
                if not (freeze_plasticity and trl >= num_rule_epochs):
                    # calculate delta term for weight update
                    delta_w = np.outer((f - beta), (c - beta))
                    w_short = np.maximum(0, np.minimum(
                        params['w_short_max'],
                        w_short + params['gamma_short'] * delta_w))
                    w_long = np.maximum(0, np.minimum(
                        params['w_long_max'],
                        w_long + params['gamma_long'] * delta_w))
                    W = w_short + w_long

                f_trace[:, t] = f
                c_trace[:, t] = c
                f_prev, c_prev = f, c

            # ---- behaviour ----------------------------------------------
            # find max activated "action" feature after stimulus offset
            # (offset is padded by + resp_pad timesteps)
            read_from = task['stim_off'] + task['resp_pad']
            resp_trace = f_trace[resp_slice, read_from:]
            max_action = np.amax(resp_trace)

            if max_action > 0:
                threshold = .98 * max_action
                action_index = np.argwhere(resp_trace > threshold)
                choice[blk, trl] = action_index[0, 0]
                rt[blk, trl] = action_index[0, 1]
            else:
                # degenerate case: no response unit ever leaves zero. Scored as
                # an error with an undefined RT rather than raising.
                choice[blk, trl] = -1

            if trl >= num_rule_epochs:
                accuracy[blk, trl] = float(choice[blk, trl] == truth[trl])
                stim_labels[blk, trl, :] = stim_feats[trl, :]
                con_labels[blk, trl] = con_flags[trl]
                rule_labels[blk, trl] = rule_dim
                tms_labels[blk, trl] = tms_on_trial[trl]

                # ---- prioritisation index --------------------------------
                # mean absolute firing rate difference between the two features
                # of the relevant dimension minus the same quantity averaged
                # over the irrelevant dimensions, taken over the response
                # period. This is the firing rate version of the relevant vs
                # irrelevant effect in Figure 2.
                win = slice(task['stim_off'], task['resp_off'])
                diffs = []
                for d in range(task['num_dims']):
                    blk_act = f_trace[built['dim_slices'][d], win]
                    # spread within the dimension: max minus mean of the rest
                    diffs.append(np.mean(np.max(blk_act, 0)
                                         - np.mean(blk_act, 0)))
                rel = diffs[rule_dim]
                irrel = np.mean([diffs[d] for d in range(task['num_dims'])
                                 if d != rule_dim])
                rel_index[blk, trl] = rel - irrel

            # ---- seizure-like dynamics ----------------------------------
            # The pathology the lateral inhibition parameter was tuned to
            # avoid is a synchronous period-2 oscillation: every conjunction
            # unit alternating in phase between a positive value and zero on
            # successive timesteps, with all selectivity destroyed. It is a
            # discrete-time instability. 
            
            # Two quantities identify it. The lag-1 autocorrelation of the
            # population mean rate is strongly positive for healthy persistent
            # activity and strongly negative when the population alternates.
            # The mean pairwise correlation between units is low when one unit
            # has won the competition and near 1 when they move together. A
            # trial is counted as seizure-like when both criteria are met.
            #
            # Saturation is tracked separately as sat_index: it catches
            # runaway excitation, which is a different failure mode, and in
            # practice stays at zero across the ranges swept here.
            win = slice(task['stim_off'] + 1, task['resp_off'])
            cw = c_trace[:, win]

            pop = np.mean(cw, axis=0)
            osc = -_lag1_corr(pop)
            sync = _mean_pairwise_corr(cw)
            osc_index[blk, trl] = osc
            sync_index[blk, trl] = sync
            seizure_frac[blk, trl] = float(osc > .5 and sync > .5)

            n_sat = np.sum(cw >= 1. - 1e-9, axis=0)
            sat_frac[blk, trl] = np.mean(n_sat >= 2)

            if return_activity:
                conj_store[blk, :, :, trl] = c_trace
                feat_store[blk, :, :, trl] = f_trace

        # ---- amplifying eigenvalues at the end of the block --------------
        # W W^T is symmetric positive semi-definite, so eigvalsh is both
        # faster and numerically cleaner than eigvals here.
        eigs = np.linalg.eigvalsh(W @ W.T)
        n_eig[blk] = np.sum(eigs > 1.)
        weight_store[blk] = W

    out = summarise(accuracy, rt, choice, con_labels, rule_labels, tms_labels,
                    n_eig, seizure_frac, sat_frac, osc_index, sync_index,
                    rel_index, built, task, num_rule_epochs)

    out['weights'] = weight_store
    out['accuracy_trials'] = accuracy[:, num_rule_epochs:]
    out['rt_trials'] = rt[:, num_rule_epochs:]
    out['stim_labels'] = stim_labels[:, num_rule_epochs:, :]
    out['congruent'] = con_labels[:, num_rule_epochs:]
    out['rule_labels'] = rule_labels[:, num_rule_epochs:]
    out['tms_labels'] = tms_labels[:, num_rule_epochs:]
    out['n_eig'] = n_eig
    out['tms_dose'] = tms_dose
    out['seed'] = rnd_seed

    if return_activity:
        out['conj'] = conj_store
        out['feat'] = feat_store

    return out


# %% summary measures

def summarise(accuracy, rt, choice, con_labels, rule_labels, tms_labels,
              n_eig, seizure_frac, sat_frac, osc_index, sync_index,
              rel_index, built, task, num_rule_epochs):
    """
    Collapse a run into the scalar outcome measures used in the sweeps:
    accuracy, RT, the congruency effect, the prioritisation index, the mean
    number of amplifying eigenvalues, and the seizure index.

    RT measures are scored on CORRECT TRIALS ONLY, which is the convention in
    the human RT literature and matches how the empirical effect this model is
    compared against was computed.
    """

    # drop the rule epochs
    acc = accuracy[:, num_rule_epochs:].flatten()
    rts = rt[:, num_rule_epochs:].flatten()
    con = con_labels[:, num_rule_epochs:].flatten()
    tms = tms_labels[:, num_rule_epochs:].flatten()
    rel = rel_index[:, num_rule_epochs:].flatten()

    # when TMS is interleaved, behaviour is scored on TMS trials only
    if np.any(tms):
        keep = tms
    else:
        keep = np.ones_like(acc, dtype=bool)

    correct = np.logical_and(acc == 1, keep)

    out = {}
    out['accuracy'] = np.mean(acc[keep]) if np.any(keep) else np.nan
    out['rt'] = np.nanmean(rts[correct]) if np.any(correct) else np.nan

    con_correct = np.logical_and(correct, con)
    incon_correct = np.logical_and(correct, ~con)
    out['rt_congruent'] = np.nanmean(rts[con_correct]) \
        if np.any(con_correct) else np.nan
    out['rt_incongruent'] = np.nanmean(rts[incon_correct]) \
        if np.any(incon_correct) else np.nan
    out['congruency_effect'] = out['rt_incongruent'] - out['rt_congruent']

    out['acc_congruent'] = np.mean(acc[np.logical_and(keep, con)]) \
        if np.any(np.logical_and(keep, con)) else np.nan
    out['acc_incongruent'] = np.mean(acc[np.logical_and(keep, ~con)]) \
        if np.any(np.logical_and(keep, ~con)) else np.nan

    out['prioritisation'] = np.nanmean(rel[correct]) \
        if np.any(correct) else np.nan

    out['mean_eig'] = np.mean(n_eig)
    # proportion of blocks retaining enough amplifying eigenvalues to support
    # one attractor per stimulus-response mapping
    out['prop_blocks_full_eig'] = np.mean(n_eig >= built['num_rule_epochs'])

    # seizure index: proportion of stimulus trials showing synchronous
    # period-2 oscillation of the conjunction population. osc_index and
    # sync_index are the two underlying quantities, reported separately so
    # that the criterion can be re-derived from them if the threshold is
    # questioned. sat_index tracks runaway saturation, a distinct failure mode.
    out['seizure_index'] = np.mean(seizure_frac[:, num_rule_epochs:])
    out['seizure_flag'] = float(out['seizure_index'] > task['seizure_thresh'])
    out['sat_index'] = np.mean(sat_frac[:, num_rule_epochs:])
    out['osc_index'] = np.mean(osc_index[:, num_rule_epochs:])
    out['sync_index'] = np.mean(sync_index[:, num_rule_epochs:])

    return out


# %% legacy interface

def plasticattractor_sim_legacy(num_blocks, num_trials, rnd_seed, TMS_sim,
                                TMS_start, params=None, task=None):
    """
    Thin wrapper returning the same 9-tuple as the original
    plasticattractor_sim, so that plasticattractor_behaviouralanalysis.py and
    plasticattractor_decoding.py can be run unchanged against this module.
    Only valid for the default two dimensional task.
    """

    if task is None:
        task = default_task()

    assert task['num_dims'] == 2, 'legacy interface assumes two stimulus dimensions'

    tms_dose = (task['stim_off'] - TMS_start) if TMS_sim else 0

    out = plasticattractor_sim(num_blocks, num_trials, rnd_seed,
                               tms_dose=tms_dose, params=params, task=task,
                               return_activity=True)

    built = build_task(task)
    num_rule_epochs = built['num_rule_epochs']

    conj_dict = {}; feat_dict = {}; choice_dict = {}; acc_dict = {}
    rt_dict = {}; c_label_dict = {}; s_label_dict = {}; r_label_dict = {}
    weight_dict = {}

    for blk in range(num_blocks):
        conj_dict[blk] = out['conj'][blk]
        feat_dict[blk] = out['feat'][blk]
        acc_dict[blk] = np.concatenate([np.zeros(num_rule_epochs),
                                        out['accuracy_trials'][blk]])
        rt_dict[blk] = np.concatenate([np.zeros(num_rule_epochs),
                                       out['rt_trials'][blk]])
        choice_dict[blk] = np.zeros(num_rule_epochs + out['accuracy_trials'].shape[1])
        # labels are 1-indexed in the original analysis scripts
        # stim_labels is now (trials, dims) holding the feature index chosen
        # on each dimension, so the colour and shape labels read straight off
        # its two columns (1-indexed, as the original analysis expects)
        stim = out['stim_labels'][blk]
        c_label_dict[blk] = {i: int(stim[i, 0]) + 1
                             for i in range(stim.shape[0])}
        s_label_dict[blk] = {i: int(stim[i, 1]) + 1
                             for i in range(stim.shape[0])}
        r_label_dict[blk] = {i: int(out['rule_labels'][blk][i])
                             for i in range(len(stim))}
        weight_dict[blk] = out['weights'][blk]

    return conj_dict, feat_dict, choice_dict, acc_dict, rt_dict, \
        c_label_dict, s_label_dict, r_label_dict, weight_dict
