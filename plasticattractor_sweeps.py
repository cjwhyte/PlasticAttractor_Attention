"""
Sweep harness for the plastic attractor network

Drives plasticattractor_networksims_general.py to produce the analyses
requested by Reviewer 2:

    comment 3, parameter sensitivity
        sweep_marginal  -- each free parameter varied alone over +/- a relative
                           range, all others held at their default
        sweep_grid      -- the alpha1 x alpha5 plane, i.e. the two parameters
                           we adjusted by hand relative to Manohar et al
                           (2019). This is also the joint perturbation
                           analysis: two parameters varied at once, and the
                           two whose values were set by hand. Broader joint
                           coverage is cited to the sweep in Manohar et al.

    comment 4, scalability
        sweep_dimensions -- number of stimulus dimensions, at a FIXED network
                            size (4 conjunction units, two features per
                            dimension, two responses), with the drive to
                            irrelevant features reduced from 0.5 to 0.25. The
                            network is deliberately not rescaled: it is
                            non-linear, so its parameters do not scale
                            linearly with size, and 4 units is in any case the
                            largest population the published parameters admit
        sweep_noise      -- noise scale

    supporting
        sweep_dose      -- TMS dose response; regression test, and fixes the
                           operating dose used by the sensitivity sweeps

An important distinction runs through all of the sensitivity analyses: the
parameter range over which the model performs the TASK is not the same as the
range over which it RECOVERS FROM PERTURBATION. Only the latter was hand tuned
(alpha1 from -0.5 to -0.45, alpha5 from 0.7 to 0.73). Every sweep therefore
takes a tms_dose argument and should be run at least twice, once at dose 0 and
once at the operating dose of 11.

This module contains definitions only. The sweeps are called from
plasticattractor_sweeps_centralscript.py, which is where the settings for each
run live.


ANATOMY OF A SWEEP
------------------
Every sweep_* function below does the same four things in the same order. Once
this pattern is clear, each individual sweep is just a different nested loop in
step 1.

    1. BUILD A FLAT JOB LIST.
       A "setting" is one combination of parameters and/or task structure. Each
       setting is run over several random seeds. Rather than looping settings
       in the outer scope and seeds in the inner scope at run time, every
       (setting, seed) pair is flattened into a single list of argument tuples:

           jobs = [(params, task, tms_dose, seed, num_blocks, num_trials), ...]

       Flattening matters because the grids are ragged -- a k = 6 capacity cell
       runs 108 trials per block while a k = 2 cell runs 12 -- so dispatching
       settings one at a time would leave most cores idle waiting for the slow
       cells. One flat list keeps every worker busy.

       Crucially, SEEDS VARY FASTEST in this list. Setting i occupies jobs
       [i*n_seeds : (i+1)*n_seeds]. Step 3 relies on that ordering.

       Alongside jobs, most sweeps also build a `settings` list recording what
       each setting actually was. Its columns differ per sweep and are
       documented in each function.

    2. DISPATCH.
       _dispatch runs the job list through joblib, checkpointing to disk as it
       goes. It returns a flat list of dicts, one per job, in job order.

    3. RESHAPE.
       _reshape folds the flat list back into (n_settings, n_seeds), then takes
       the mean and standard error across seeds. It returns a dict `summ` whose
       keys are:

           summ['accuracy']        -- array, one value per setting
           summ['accuracy_sem']    -- its standard error over seeds
           ... and the same pair for every entry of METRICS ...
           summ['n_valid']         -- how many seeds gave a usable value

    4. SAVE AND RETURN.
       _save writes summ plus the setting descriptors to
       sweep_results/<name>.npz, _clear_cache removes the checkpoint, and the
       function returns (setting descriptors, summ) for plotting.

The plot_* functions at the bottom all take a `summ` dict of this shape.

@author: Christopher Whyte
"""

import os
import hashlib
import warnings
import numpy as np
from joblib import Parallel, delayed

from plasticattractor_networksims_general import (
    plasticattractor_sim, default_params, default_task, build_task)


# %% configuration

# Outcome measures carried through every sweep. These are keys of the dict
# returned by plasticattractor_sim, and every one of them gets a mean and a
# standard error in the summary dicts.
#
#   accuracy             proportion correct
#   rt                   mean RT on correct trials
#   congruency_effect    RT(incongruent) - RT(congruent), on correct trials
#   prioritisation       relevant minus irrelevant firing rate separation
#   mean_eig             mean number of eigenvalues of W W' above 1
#   prop_blocks_full_eig proportion of blocks with one amplifying eigenvalue
#                        per stimulus-response mapping
#   seizure_index        proportion of trials showing synchronous period-2
#                        oscillation of the conjunction population
#   seizure_flag         whether seizure_index exceeded task['seizure_thresh']
#   osc_index            lag-1 alternation of the population mean, and
#   sync_index           mean pairwise correlation between conjunction units.
#                        These two are the raw quantities behind the seizure
#                        criterion, kept so the threshold can be re-derived
#                        without re-running anything
#   sat_index            fraction of timesteps with 2+ saturated units. A
#                        different failure mode (runaway excitation) which in
#                        practice stays at zero over the ranges swept here
METRICS = ['accuracy', 'rt', 'congruency_effect', 'prioritisation',
           'mean_eig', 'prop_blocks_full_eig', 'seizure_index',
           'seizure_flag', 'osc_index', 'sync_index', 'sat_index']

# Free parameters that the sensitivity analyses vary. Names follow the code:
# alpha1, alpha4 lateral inhibition; alpha2, alpha5 self excitation;
# alpha3, alpha6 synaptic gain.
#
# Two deliberate exclusions:
#   - the learning rates gamma_short and gamma_long
#   - epsilon, the noise scale. A +/- 50% marginal sweep spans 0.0025 to
#     0.0075, which is far too narrow to move any outcome measure, and the
#     question it would answer is covered better by sweep_noise, which spans
#     0.005 to 0.2 and crosses it with the number of conjunction units.
# Add any of them back to this list to include them.
SWEEP_PARAMS = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6',
                'beta']

# Defaults for sweeps, deliberately lighter than the main simulations (20
# blocks, 20 seeds). Ample for the summary measures and keeps the grids
# tractable. The central script overrides these per call.
SWEEP_BLOCKS = 10
SWEEP_TRIALS = 12
SWEEP_SEEDS = 10

# where .npz results and .png figures are written
OUTDIR = 'sweep_results'


# %% parameter handling
#
# The parameter dict from default_params() holds the six alphas in a single
# numpy array, but the sweeps need to address them individually by name. These
# two helpers translate between the two representations, so that 'alpha1'
# means params['alpha'][0] and 'beta' means params['beta'].

def get_param(params, name):
    """
    Read a swept parameter by name.

    'alpha1'..'alpha6' index into the alpha array (1-indexed to match the
    manuscript); any other name is a top level key of the params dict.
    """
    if name.startswith('alpha'):
        return params['alpha'][int(name[5:]) - 1]
    return params[name]


def set_param(params, name, value):
    """
    Return a COPY of params with one named parameter set.

    The copy matters: the alpha entry is a numpy array, so a shallow copy would
    leave every job in the sweep sharing (and overwriting) the same array. The
    dict comprehension below copies any array values explicitly.
    """
    new = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in params.items()}
    if name.startswith('alpha'):
        new['alpha'][int(name[5:]) - 1] = value
    else:
        new[name] = value
    return new


# %% running and caching

def run_one(params, task, tms_dose, seed, num_blocks, num_trials):
    """
    One simulation: one parameter/task setting at one seed.

    This is the unit of work joblib dispatches, so it is defined at module
    level and takes only picklable arguments (dicts, ints) rather than closing
    over anything.

    return_activity is False: the sweeps only need the summary measures, and
    holding the full conjunction and feature arrays for thousands of runs would
    exhaust memory for no benefit.
    """
    out = plasticattractor_sim(num_blocks, num_trials, seed,
                               tms_dose=tms_dose, params=params, task=task,
                               return_activity=False)
    # keep only the summary measures, discard everything else
    return {m: out[m] for m in METRICS}


# Measures returned by the legacy dose-response path. Fewer than METRICS,
# because they come from the ORIGINAL analysis function rather than from
# summarise(): plasticattractor_beh returns behaviour only, and the
# eigenvalue count is computed here from the weights it hands back.
LEGACY_METRICS = ['accuracy', 'rt', 'rt_congruent', 'rt_incongruent',
                  'congruency_effect', 'acc_congruent', 'acc_incongruent',
                  'mean_eig']


def run_one_legacy(params, task, tms_dose, seed, num_blocks, num_trials):
    """
    One simulation through the LEGACY interface, scored with the ORIGINAL
    analysis code.

    This is the regression test for the whole refactor. Rather than calling
    plasticattractor_sim and summarise(), it calls
    plasticattractor_sim_legacy -- which returns the same nine-tuple of dicts
    the original plasticattractor_networksims.py did -- and passes that
    straight into plasticattractor_beh from
    plasticattractor_behaviouralanalysis.py, unmodified. If the refactored
    simulation still reproduces the published dose response through the
    original analysis chain, both ends of the pipeline are intact.

    plasticattractor_behaviouralanalysis.py therefore has to be importable,
    i.e. sitting alongside these scripts.
    """
    from plasticattractor_networksims_general import (
        plasticattractor_sim_legacy)
    from plasticattractor_behaviouralanalysis import plasticattractor_beh

    # the legacy interface takes a TMS on/off flag and a start timestep
    # rather than a dose; dose is stim_off - TMS_start
    tms_on = tms_dose > 0
    tms_start = task['stim_off'] - tms_dose

    (conj_dict, feat_dict, choice_dict, acc_dict, rt_dict, c_label_dict,
     s_label_dict, r_label_dict, weight_dict) = plasticattractor_sim_legacy(
        num_blocks, num_trials, seed, tms_on, tms_start,
        params=params, task=task)

    (accuracy_overall, correct_rt, congruent_rt, congruent_accuracy,
     incongruent_rt, incongruent_accuracy, *_) = plasticattractor_beh(
        num_trials, num_blocks, acc_dict, rt_dict, c_label_dict,
        s_label_dict, r_label_dict, conj_dict, feat_dict)

    # amplifying eigenvalues at the end of each block, from the weights the
    # legacy interface returns
    n_eig = []
    for blk in range(num_blocks):
        W = weight_dict[blk]
        n_eig.append(np.sum(np.linalg.eigvalsh(W @ W.T) > 1.))

    return {'accuracy': accuracy_overall,
            'rt': correct_rt,
            'rt_congruent': congruent_rt,
            'rt_incongruent': incongruent_rt,
            'congruency_effect': incongruent_rt - congruent_rt,
            'acc_congruent': congruent_accuracy,
            'acc_incongruent': incongruent_accuracy,
            'mean_eig': float(np.mean(n_eig))}


def _dispatch(jobs, n_jobs, verbose=5, cache_name=None, chunk_size=200,
              runner=None, metrics=None):
    """
    Run a flat list of job tuples and return a flat list of result dicts, in
    the same order.

    Each job is (params, task, tms_dose, seed, num_blocks, num_trials), which
    is exactly run_one's signature, so the jobs are splatted straight into it.

    CHECKPOINTING. If cache_name is given, the job list is run in chunks and
    results are written to sweep_results/_cache_<name>.npz after every chunk. A
    re-run with the same cache_name loads what is already there and starts from
    where it stopped. The full sweeps take hours, so an interrupted session
    costs one chunk rather than the whole run. The cache is deleted
    automatically once the sweep saves its results; delete it by hand to force
    a clean re-run.

    A cache is only reused if its SIGNATURE matches: the job list it was
    computed from, and the metric names its columns stand for. Rows are
    positional in two independent senses -- the row index encodes
    (setting, seed) through the ordering _reshape relies on, and the column
    index encodes the metric -- so a cache written under different settings is
    not merely stale, it is silently MISLABELLED. A cache written at 20 seeds
    and resumed at 40 has its rows re-attributed to the wrong settings by
    _reshape's (n_settings, n_seeds) fold; one written with a different metric
    list and resumed at exactly len(jobs) rows skips the loop entirely, and
    dict(zip(metrics, row)) then shifts every label past the point where the
    lists diverge. The old guard (len(done) > len(jobs)) caught neither,
    because it only fired when the sweep SHRANK.
    """

    # simple path: no checkpointing, run everything in one go
    runner = runner if runner is not None else run_one
    metrics = metrics if metrics is not None else METRICS

    if cache_name is None:
        return Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(runner)(*job) for job in jobs)

    os.makedirs(OUTDIR, exist_ok=True)
    cache_path = os.path.join(OUTDIR, '_cache_' + cache_name + '.npz')
    signature = _job_signature(jobs)

    # `done` holds completed results as a (n_completed, n_metrics) array, which
    # is why METRICS order is fixed: the columns are positional
    done = np.zeros([0, len(metrics)])

    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            reason = _cache_mismatch(cached, jobs, signature, metrics)
            if reason is not None:
                print('ignoring %s: %s' % (cache_path, reason))
            elif len(cached['results']):
                done = cached['results']
                print('resuming from %d/%d completed runs'
                      % (len(done), len(jobs)))

    # resume from the first job not already in the cache
    start = len(done)
    for lo in range(start, len(jobs), chunk_size):
        hi = min(lo + chunk_size, len(jobs))
        chunk = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(runner)(*job) for job in jobs[lo:hi])
        # dicts -> rows, in metric order
        chunk = np.array([[r[m] for m in metrics] for r in chunk], dtype=float)
        done = np.vstack([done, chunk])
        np.savez(cache_path, results=done, signature=signature,
                 metrics=np.array(metrics))
        print('  cached %d/%d' % (len(done), len(jobs)))

    # rows -> dicts, so the caller sees the same thing either path
    return [dict(zip(metrics, row)) for row in done]


def _job_signature(jobs):
    """
    A digest of the full job list, used to decide whether a checkpoint may be
    resumed.

    Each job is (params, task, tms_dose, seed, num_blocks, num_trials), so
    hashing the repr covers everything that could change what a row means: the
    parameter values, the task dict, the dose, the seed, and the block and
    trial counts. Dict repr is insertion ordered and the config builders
    construct them the same way every time, so this is deterministic within a
    code version. If it ever is not, the failure mode is a spurious cache
    miss -- a wasted re-run, never a mislabelled result.
    """
    return hashlib.sha256(repr(jobs).encode()).hexdigest()


def _cache_mismatch(cached, jobs, signature, metrics):
    """
    Return None if `cached` may be resumed, else a short reason why not.

    Caches written before signatures were added carry no 'signature' key, and
    are refused rather than guessed at: they are exactly the ones whose
    settings cannot be established.
    """
    if 'signature' not in cached:
        return 'no signature, so the settings it used cannot be verified'
    if str(cached['signature']) != signature:
        return 'job list differs from the one that wrote it'
    if list(cached['metrics']) != list(metrics):
        return 'metric list differs from the one that wrote it'
    if len(cached['results']) > len(jobs):
        return 'more cached rows than jobs'
    return None


def _clear_cache(cache_name):
    """Remove a completed sweep's checkpoint file."""
    path = os.path.join(OUTDIR, '_cache_' + cache_name + '.npz')
    if os.path.exists(path):
        os.remove(path)


def _reshape(results, n_settings, n_seeds, metrics=None):
    """
    Fold the flat result list back into settings x seeds and average.

    This depends on the ordering established when the job list was built:
    seeds vary fastest, so setting i occupies results[i*n_seeds:(i+1)*n_seeds]
    and a straight reshape to (n_settings, n_seeds) recovers the grid. If a
    sweep ever builds its job list with the loops the other way round, this
    silently mislabels everything.

    Returns a dict with, for each measure m in METRICS:
        out[m]              mean over seeds, one value per setting
        out[m + '_sem']     standard error over the seeds that contributed
        out[m + '_n_valid'] how many seeds returned a usable (non-NaN) value
    plus:
        out['n_valid']      alias for out['accuracy_n_valid']

    n_valid matters because degenerate settings produce NaN -- for instance a
    network with no correct trials has no RT to average. Without it, a cell
    where 15 of 20 seeds failed looks identical to one where none did. Check it
    before interpreting any cell near the edge of a sweep.

    It is counted PER METRIC. It used to be taken once from accuracy "since the
    NaN pattern is shared", but it is not: accuracy is never NaN, while
    congruency_effect is NaN for any seed with no correct trials in one of the
    congruency cells. Reading the accuracy count would have said 20/20 seeds
    for a congruency effect that only 5 seeds contributed to.

    THE SEM DIVIDES BY n_valid, NOT n_seeds. Dividing by the nominal seed count
    made the error bar SMALLER the more seeds dropped out -- a cell where 15 of
    20 seeds failed got a bar 2x too narrow, in exactly the degenerate regime
    where it should have been widest. ddof=1 for the same reason: the sample SD
    is what a standard error wants, and the bias only matters once the
    effective n is small, which is the same regime.
    """
    metrics = metrics if metrics is not None else METRICS
    out = {}
    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=float)
        vals = vals.reshape(n_settings, n_seeds)
        n_valid = np.sum(~np.isnan(vals), axis=1)

        # nanmean/nanstd over an all-NaN row warn rather than failing; suppress
        # that, since n_valid records the failure explicitly. Note errstate
        # alone does not do it -- "Mean of empty slice" and "Degrees of freedom
        # <= 0" come through the warnings machinery, not the floating point
        # error state.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[m] = np.nanmean(vals, axis=1)
                sd = np.nanstd(vals, axis=1, ddof=1)
                sem = sd / np.sqrt(n_valid)

        # a standard error needs at least two contributing seeds
        out[m + '_sem'] = np.where(n_valid >= 2, sem, np.nan)
        out[m + '_n_valid'] = n_valid
        if m == 'accuracy':
            # the name every existing caller uses
            out['n_valid'] = n_valid
    return out


def _save(name, **arrays):
    """
    Write a sweep to sweep_results/<name>.npz.

    Called with the summary dict splatted in (**summ) alongside the setting
    descriptors, so the .npz holds everything needed to re-plot without
    re-running: e.g. np.load('sweep_results/dose_response.npz')['accuracy'].
    """
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + '.npz')
    np.savez(path, **arrays)
    print('saved ' + path)
    return path


# %% configurations
#
# A "config" is the full description of one setting: the parameter dict, the
# task dict, the TMS dose, and how many blocks and trials to run. Each sweep
# builds its list of configs first, and then turns them into jobs.
#
# Splitting it out this way means the example-block figures can be generated
# from the same list the sweep itself ran, rather than from a second copy of
# the loop that could drift out of step with it. Call the builder, pass the
# result to plot_example_blocks, and you are looking at exactly the settings
# the sweep used.
#
# Every config carries a 'label' string naming the setting, which is what
# appears in the example-block titles.

def _configs_to_jobs(configs, seeds):
    """
    Flatten a list of configs into the job list _dispatch expects.

    Config order is preserved and seeds vary fastest, which is the ordering
    _reshape depends on.
    """
    jobs = []
    for c in configs:
        for s in seeds:
            jobs.append((c['params'], c['task'], c['tms_dose'], int(s),
                         c['num_blocks'], c['num_trials']))
    return jobs


def configs_dose(dose_list=None, num_blocks=SWEEP_BLOCKS,
                 num_trials=SWEEP_TRIALS, params=None, task=None):
    """One config per TMS dose; parameters and task held at their defaults."""

    if dose_list is None:
        dose_list = list(range(0, 20))

    base_params = params if params is not None else default_params()
    base_task = task if task is not None else default_task()

    configs = []
    for dose in dose_list:
        configs.append({'label': 'TMS dose = %d' % dose,
                        'params': base_params,
                        'task': base_task,
                        'tms_dose': int(dose),
                        'num_blocks': num_blocks,
                        'num_trials': num_trials,
                        'dose': dose})
    return configs


def configs_marginal(param_names=None, rel_range=.25, n_points=11,
                     tms_dose=0, num_blocks=SWEEP_BLOCKS,
                     num_trials=SWEEP_TRIALS):
    """
    One config per (parameter, relative factor), ordered parameter-major.

    Also returns the factor array, which is shared across parameters and is
    the natural x axis for the marginal plots.
    """

    if param_names is None:
        param_names = SWEEP_PARAMS

    base_params = default_params()
    task = default_task()
    factors = np.linspace(1 - rel_range, 1 + rel_range, n_points)

    configs = []
    for pname in param_names:
        base_val = get_param(base_params, pname)
        for fac in factors:
            val = base_val * fac
            # set_param returns a copy, so each config gets its own params
            configs.append({'label': '%s = %.4f (%.2fx)' % (pname, val, fac),
                            'params': set_param(base_params, pname, val),
                            'task': task,
                            'tms_dose': tms_dose,
                            'num_blocks': num_blocks,
                            'num_trials': num_trials,
                            'param': pname,
                            'value': val,
                            'factor': fac})
    return configs, factors


def configs_grid(param_a='alpha1', param_b='alpha5', rel_range=.25,
                 n_points=13, tms_dose=11, num_blocks=SWEEP_BLOCKS,
                 num_trials=SWEEP_TRIALS):
    """
    One config per cell of the two parameter plane, a-major then b, which is
    the row-major order the grid reshape assumes.
    """

    base_params = default_params()
    task = default_task()

    a_vals = get_param(base_params, param_a) * \
        np.linspace(1 - rel_range, 1 + rel_range, n_points)
    b_vals = get_param(base_params, param_b) * \
        np.linspace(1 - rel_range, 1 + rel_range, n_points)

    configs = []
    for a in a_vals:
        for b in b_vals:
            # nested set_param: copy with a set, then copy again with b set
            p = set_param(set_param(base_params, param_a, a), param_b, b)
            configs.append({'label': '%s = %.4f, %s = %.4f'
                            % (param_a, a, param_b, b),
                            'params': p,
                            'task': task,
                            'tms_dose': tms_dose,
                            'num_blocks': num_blocks,
                            'num_trials': num_trials,
                            'a': a, 'b': b})
    return configs, a_vals, b_vals


def configs_dimensions(dims_list=(2, 3, 4), irrel_drive=.25,
                       num_conjunction=4, tms_dose=0, num_blocks=24,
                       num_trials=SWEEP_TRIALS):
    """
    One config per number of stimulus dimensions, at a FIXED network size.

    The network is not rescaled: num_conjunction stays at the published 4,
    which is the only stable population size (the conjunction layer is flip
    unstable beyond N = (1 + alpha2)/|alpha1| = 4.44). Features per dimension
    and responses stay at two, so a rule always contains two
    stimulus-response mappings and chance stays at 0.50 whatever D is. The
    only thing that grows is the number of IRRELEVANT dimensions to ignore:
    one at D = 2, three at D = 4.

    THE DRIVE TO IRRELEVANT FEATURES IS REDUCED FROM 0.5 TO 0.25. This is the
    substantive change. During rule presentation the features of irrelevant
    dimensions are driven to irrel_drive, instructing the network that they
    are present but not decisive. The Hebbian update then binds the winning
    conjunction unit to every above-baseline feature, so that drive creates
    diffuse weight alongside the targeted weight from the +1 relevant feature
    and response. At D = 2 that is two diffuse features against two targeted
    ones and the targeted component wins; each extra dimension adds two more
    diffuse features while the targeted count stays at two, so by D = 3 the
    ratio has inverted and the weight matrix never becomes specific enough to
    form two amplifying eigenvalues.

    Halving the drive restores this. At D = 3 accuracy goes from 0.51 (chance)
    to 0.85 and the eigenvalue count from 1.79 back to 2.02. It costs nothing
    at D = 2, where accuracy is unchanged at 0.97 and the congruency effect
    improves, and the full TMS dose response is preserved.

    Note this is a property of how the RULE IS PRESENTED, not of the network's
    dynamics -- the analogue of how salient the irrelevant dimension is in the
    instruction, rather than a re-tuning of the model. The published 0.5 is
    left as the default in default_task(); 0.25 is set explicitly here.

    Note also that it is not a 1/(D-1) normalisation: a flat 0.25 works across
    D = 2 to 5, and 0.167 was worse than 0.25 at D = 4. The accurate statement
    is that 0.5 carries little margin even at D = 2 and none by D = 3.

    num_blocks defaults to 24 so that the D rules (one per dimension) are
    cycled an equal number of times at D = 2, 3 and 4.

    Trials per block stay at the published twelve for every D. Stimuli are
    sampled rather than enumerated (see stim_generator), so the trial count
    does not have to track the 2 ** D size of the stimulus space, and holding
    it constant removes a confound.
    """

    base_params = default_params()

    configs = []
    for nd in dims_list:
        task = default_task()
        task['num_dims'] = nd
        task['num_features_per_dim'] = 2
        task['num_responses'] = 2
        task['num_conjunction'] = num_conjunction
        task['irrel_drive'] = irrel_drive

        configs.append({'label': '%d dimensions (%d irrelevant), '
                        'irrel_drive %.2f' % (nd, nd - 1, irrel_drive),
                        'params': base_params,
                        'task': task,
                        'tms_dose': tms_dose,
                        'num_blocks': num_blocks,
                        'num_trials': num_trials,
                        'num_dims': nd})
    return configs


def configs_noise(eps_list=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2), tms_dose=0,
                  num_blocks=SWEEP_BLOCKS, num_trials=SWEEP_TRIALS):
    """One config per noise scale, at the published network size."""

    base_params = default_params()
    task = default_task()

    configs = []
    for eps in eps_list:
        configs.append({'label': 'epsilon = %.3f' % eps,
                        'params': set_param(base_params, 'epsilon', eps),
                        'task': task,
                        'tms_dose': tms_dose,
                        'num_blocks': num_blocks,
                        'num_trials': num_trials,
                        'epsilon': eps})
    return configs


# %% ==================== SWEEP: TMS dose response ====================

def sweep_dose(dose_list=None, num_blocks=SWEEP_BLOCKS,
               num_trials=SWEEP_TRIALS, n_seeds=SWEEP_SEEDS, params=None,
               task=None, n_jobs=-1, name=None):
    """
    TMS dose response, run through the LEGACY INTERFACE and scored with the
    ORIGINAL analysis code.

    This is the regression test for the refactor. plasticattractor_sim_legacy
    returns the same nine-tuple of dicts the original
    plasticattractor_networksims.py did, and that goes straight into
    plasticattractor_beh from plasticattractor_behaviouralanalysis.py,
    unmodified. Reproducing the published dose response through that chain
    checks both ends of the pipeline at once, which running the new
    summarise() path would not. plasticattractor_behaviouralanalysis.py has to
    be importable, i.e. sitting alongside these scripts.

    Note the measures here are LEGACY_METRICS, not METRICS: the original
    analysis returns behaviour only, so there is no prioritisation or seizure
    index. The eigenvalue count is computed from the weights the legacy
    interface hands back.

    Reproduces Figures 4 and 5, and serves two purposes:

        1. As a validation test. Any change to the simulation code should
           leave the published pattern intact: accuracy at ceiling out to a
           dose of about 11, collapse over 12-14, eigenvalues falling from 2
           to 1 in step with accuracy, and the congruency effect shrinking.

        2. To locate the operating dose. The sensitivity sweeps are run at
           dose 11 because that is where the empirical constellation of
           effects is reproduced. If the task or parameters change, the
           operating dose has to be re-derived rather than assumed.

    Only two purposes here, not three: the seizure index is not available
    through this path, since the original analysis function does not compute
    it. The quantified seizure boundary comes from the marginal sweep
    instead (plot_seizure_boundary), which runs the current pipeline.

    Returns
        dose_list : array of doses, one entry per setting
        summ      : summary dict (see ANATOMY OF A SWEEP in the module
                    docstring)
    """

    if dose_list is None:
        dose_list = list(range(0, 20))

    seeds = np.arange(n_seeds)
    name = name or 'dose_response'

    # STEP 1: one config per dose, flattened into jobs with seeds fastest
    configs = configs_dose(dose_list=dose_list, num_blocks=num_blocks,
                           num_trials=num_trials, params=params, task=task)
    jobs = _configs_to_jobs(configs, seeds)

    # STEPS 2 and 3: run through the LEGACY interface and the ORIGINAL
    # analysis code, then average over seeds
    results = _dispatch(jobs, n_jobs, cache_name=name,
                        runner=run_one_legacy, metrics=LEGACY_METRICS)
    summ = _reshape(results, len(configs), n_seeds, metrics=LEGACY_METRICS)

    # STEP 4: save the dose alongside every measure, then drop the checkpoint
    _save(name, dose=np.array(dose_list), **summ)
    _clear_cache(name)
    return np.array(dose_list), summ


# %% =========== SWEEP: marginal parameter sensitivity (R2.3) ===========

def sweep_marginal(param_names=None, rel_range=.25, n_points=11, tms_dose=0,
                   num_blocks=SWEEP_BLOCKS, num_trials=SWEEP_TRIALS,
                   n_seeds=SWEEP_SEEDS, n_jobs=-1, name=None):
    """
    Vary each parameter alone over default * (1 -/+ rel_range), holding all
    the others at their published values.

    Settings are ordered parameter-major: all n_points values of the first
    parameter, then all n_points of the second, and so on. `labels` records
    which parameter each setting belongs to, which is how the plot functions
    split the results back into panels.

    NOTE ON SIGN: the lateral inhibition parameters are negative, so scaling by
    a factor above 1 moves them further from zero. That is the intended
    reading, i.e. a factor of 1.25 means "25% more lateral inhibition".

    rel_range defaults to 0.25 rather than something wider so that the grid is
    fine enough to resolve where behaviour changes. At +/- 25% with 11 points
    the alpha1 step is 0.0225, which still brackets the flip boundary at
    alpha1 = -(1 + alpha2)/N = -0.5 (a factor of 1.111), so the seizure
    transition is captured with roughly twice the resolution a +/- 50% sweep
    would give it.

    Returns
        labels  : list, parameter name per setting (length n_params*n_points)
        values  : list, the absolute parameter value used per setting
        factors : array of n_points relative factors (shared by all parameters,
                  and the natural x axis for the marginal plots)
        summ    : summary dict
    """

    if param_names is None:
        param_names = SWEEP_PARAMS

    seeds = np.arange(n_seeds)
    name = name or 'marginal_tms{:d}'.format(tms_dose)

    # STEP 1: one config per (parameter, factor), then flatten into jobs
    configs, factors = configs_marginal(
        param_names=param_names, rel_range=rel_range, n_points=n_points,
        tms_dose=tms_dose, num_blocks=num_blocks, num_trials=num_trials)
    jobs = _configs_to_jobs(configs, seeds)

    # which parameter and which value each setting corresponds to
    labels = [c['param'] for c in configs]
    values = [c['value'] for c in configs]

    # STEPS 2 and 3
    results = _dispatch(jobs, n_jobs, cache_name=name)
    summ = _reshape(results, len(configs), n_seeds)

    # STEP 4. factor is tiled so it lines up with labels/values row for row
    _save(name, param=np.array(labels), value=np.array(values),
          factor=np.tile(factors, len(param_names)), tms_dose=tms_dose,
          **summ)
    _clear_cache(name)
    return labels, values, factors, summ


# %% =============== SWEEP: alpha1 x alpha5 plane (R2.3) ===============

def sweep_grid(param_a='alpha1', param_b='alpha5', rel_range=.25,
               n_points=13,
               tms_dose=11, num_blocks=SWEEP_BLOCKS, num_trials=SWEEP_TRIALS,
               n_seeds=SWEEP_SEEDS, n_jobs=-1, name=None):
    """
    Two parameter grid. Defaults to the plane we actually tuned by hand:
    alpha1 (conjunction lateral inhibition, moved from -0.5 to -0.45 so that
    simulated TMS did not produce seizure-like dynamics) against alpha5
    (feature self excitation, moved from 0.7 to 0.73 so that the network could
    recover after a pulse). Run at the operating TMS dose, since that is the
    regime the tuning was for.

    This is also the joint perturbation analysis for the revision: two
    parameters varied simultaneously, and the two whose values we set by hand.

    UNLIKE THE OTHER SWEEPS, the arrays in summ are 2-D here, of shape
    (n_points, n_points), indexed [index into a_vals, index into b_vals]. The
    reshape at the end does that conversion.

    Returns
        a_vals : the n_points values of param_a
        b_vals : the n_points values of param_b
        summ   : summary dict, each entry an (n_points, n_points) array
    """

    seeds = np.arange(n_seeds)
    name = name or 'grid_{}_{}_tms{:d}'.format(param_a, param_b, tms_dose)

    # STEP 1: one config per cell, a-major then b, which is the row-major
    # order the reshape below assumes
    configs, a_vals, b_vals = configs_grid(
        param_a=param_a, param_b=param_b, rel_range=rel_range,
        n_points=n_points, tms_dose=tms_dose, num_blocks=num_blocks,
        num_trials=num_trials)
    jobs = _configs_to_jobs(configs, seeds)

    # STEPS 2 and 3
    results = _dispatch(jobs, n_jobs, cache_name=name)
    summ = _reshape(results, n_points * n_points, n_seeds)

    # fold the flat list of settings onto the 2-D grid
    summ = {k: v.reshape(n_points, n_points) for k, v in summ.items()}

    # STEP 4
    _save(name, a_vals=a_vals, b_vals=b_vals, param_a=param_a,
          param_b=param_b, tms_dose=tms_dose, **summ)
    _clear_cache(name)
    return a_vals, b_vals, summ


# %% ======== SWEEP: stimulus dimensionality at fixed size (R2.4) ========

def sweep_dimensions(dims_list=(2, 3, 4), irrel_drive=.25, num_conjunction=4,
                     tms_dose=0, num_blocks=24, num_trials=SWEEP_TRIALS,
                     n_seeds=SWEEP_SEEDS, n_jobs=-1, name=None):
    """
    How does the model scale to higher dimensional stimulus spaces?

    The network is held at its published size -- 4 conjunction units, two
    features per dimension, two responses -- and only the number of stimulus
    dimensions grows. One dimension is relevant per rule and the rest are
    ignored, so a rule always contains two mappings and chance stays at 0.50
    throughout, so accuracy is directly comparable across D.

    The drive to irrelevant features during rule presentation is reduced from
    the published 0.5 to 0.25, which is what makes D > 2 work at all. See
    configs_dimensions for the mechanism and the evidence that this costs
    nothing at D = 2 or under TMS.

    This answers the "higher dimensional stimulus spaces" half of Reviewer 2's
    comment 4. The other half, more complex rule sets, is a separate and
    structurally different limit: k mappings need k amplifying eigenvalues from
    a matrix of rank N, so k <= N = 4, and reducing the drive does not help
    there (it removes most of the conjunction unit reuse but leaves accuracy
    where it was).

    PRIORITISATION IS THE MEASURE OF INTEREST alongside accuracy: the
    prediction is about whether the targeted binding survives a growing
    diffuse component.

    Returns
        dims : array of dimension counts, one entry per setting
        summ : summary dict
    """

    seeds = np.arange(n_seeds)
    name = name or 'dimensions_tms{:d}'.format(tms_dose)

    # STEP 1: one config per dimensionality; seeds vary fastest
    configs = configs_dimensions(dims_list=dims_list,
                                 irrel_drive=irrel_drive,
                                 num_conjunction=num_conjunction,
                                 tms_dose=tms_dose, num_blocks=num_blocks,
                                 num_trials=num_trials)
    jobs = _configs_to_jobs(configs, seeds)

    # STEPS 2 and 3
    results = _dispatch(jobs, n_jobs, cache_name=name)
    summ = _reshape(results, len(configs), n_seeds)

    # STEP 4
    dims = np.array([c['num_dims'] for c in configs])
    _save(name, num_dims=dims, irrel_drive=irrel_drive,
          num_conjunction=num_conjunction, num_trials=num_trials,
          tms_dose=tms_dose, **summ)
    _clear_cache(name)
    return dims, summ


# %% =============== SWEEP: noise sensitivity (R2.4) ===============

def sweep_noise(eps_list=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2), tms_dose=0,
                num_blocks=SWEEP_BLOCKS, num_trials=SWEEP_TRIALS,
                n_seeds=SWEEP_SEEDS, n_jobs=-1, name=None):
    """
    Noise scale, at the published network size.

    Answers the second half of Reviewer 2's comment 4 ("capacity limits and
    susceptibility to noise"). The published value is epsilon = 0.005,
    following Manohar et al. (2019), and the list spans a 40-fold range above
    it -- far wider than the marginal sensitivity sweep, which is why epsilon
    is excluded from SWEEP_PARAMS.

    The number of conjunction units is NOT varied here. Crossing noise with
    population size would fold a capacity manipulation into what should be a
    single-factor result, and task size is already covered by
    sweep_mappings.

    Returns
        eps_vals : array of noise scales, one entry per setting
        summ     : summary dict
    """

    seeds = np.arange(n_seeds)
    name = name or 'noise_tms{:d}'.format(tms_dose)

    # STEP 1: one config per noise scale; seeds vary fastest
    configs = configs_noise(eps_list=eps_list, tms_dose=tms_dose,
                            num_blocks=num_blocks, num_trials=num_trials)
    jobs = _configs_to_jobs(configs, seeds)

    # STEPS 2 and 3
    results = _dispatch(jobs, n_jobs, cache_name=name)
    summ = _reshape(results, len(configs), n_seeds)

    # STEP 4
    eps_vals = np.array(eps_list)
    _save(name, epsilon=eps_vals, tms_dose=tms_dose, **summ)
    _clear_cache(name)
    return eps_vals, summ


# %% ========================= figures =========================
#
# Every plot function takes a `summ` dict as produced by a sweep, plus whatever
# setting descriptors that sweep returned, and follows the same convention:
#
#   measure  which key of summ to plot; the matching '_sem' key supplies the
#            error bars
#   fname    where to write the .png, or None to skip writing
#   show     display the figure as well as writing it. The central script
#            wires this to its show_figures flag. When not shown the figure is
#            closed, so a long run producing many figures does not accumulate
#            them in memory.
#   title    optional suptitle. The central script passes a string recording
#            the configuration the sweep ran at (TMS dose, blocks, seeds, and
#            anything held fixed), so a .png sitting in sweep_results is
#            self-describing rather than needing the script to interpret it.
#
# pyplot is imported inside each function rather than at module level so that
# the backend can still be chosen by the caller before the first import.

def _suptitle(fig, title):
    """
    Render a figure title, wrapped to the figure width and with room left for
    it. Titles carry the full configuration a sweep ran at, so they are long,
    and an unwrapped suptitle is silently clipped at the figure edge.
    """
    import textwrap
    if not title:
        return
    # roughly 12 characters per inch at fontsize 10
    width = max(30, int(fig.get_size_inches()[0] * 12))
    wrapped = textwrap.fill(title, width)
    n_lines = wrapped.count('\n') + 1
    fig.suptitle(wrapped, fontsize=9)
    # tight_layout does not reliably reserve space for a multi-line suptitle
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.045 * n_lines])


def _show_or_close(fig, show):
    """Display the figure, or close it to free the memory."""
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dose(dose_list, summ, fname=None, show=False, title=None):
    """
    Four panels against TMS dose: accuracy, RT, congruency effect, and the
    number of amplifying eigenvalues. Reproduces Figure 5. Takes the summary
    dict from sweep_dose, whose measures are LEGACY_METRICS. The first dose is
    dropped, so the panels start from the second element of dose_list.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    # (summ key, y axis label) for each panel
    panels = [('accuracy', 'accuracy'), ('rt', 'RT (timesteps)'),
              ('congruency_effect', 'congruency effect'),
              ('mean_eig', 'eigenvalues > 1')]

    doses = list(dose_list)[1:]

    for ax, (m, label) in zip(axes, panels):
        ax.errorbar(doses, summ[m][1:], yerr=summ[m + '_sem'][1:],
                    color='k', marker='o', ms=2.5, lw=1, capsize=2.5,
                    elinewidth=.8)
        ax.set_xlabel('TMS dose (train length)')
        ax.set_ylabel(label)

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


def plot_marginal_compare(summ_task, summ_tms, labels, factors,
                          measure='accuracy', fname=None, show=False,
                          title=None):
    """
    Overlay the dose 0 and operating-dose marginal sweeps on the same axes.
    Takes two summary dicts from two calls to sweep_marginal, which must have
    used the same param_names, rel_range and n_points.

    This is the figure that carries the argument for Reviewer 2, comment 3.
    The claim is not "the model is robust" flatly, but that two different
    ranges are being conflated: the range over which the model performs the
    TASK (dose 0, expected to be broad) and the range over which it RECOVERS
    FROM PERTURBATION (the operating dose, narrower, and the only one we ever
    hand tuned). Plotting them separately makes that distinction visible;
    plotting only one invites the reviewer to keep reading them as the same
    thing.
    """

    import matplotlib.pyplot as plt

    labels = np.array(labels)
    param_names = list(dict.fromkeys(labels))
    n = len(param_names)
    ncol = 4
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, pname in enumerate(param_names):
        idx = labels == pname
        ax = axes[i]
        # black: does the model do the task at all
        ax.errorbar(factors, summ_task[measure][idx],
                    yerr=summ_task[measure + '_sem'][idx],
                    color='k', marker='o', ms=2.5, lw=1, capsize=2.5,
                    elinewidth=.8, label='no TMS')
        # blue: does it recover from perturbation
        ax.errorbar(factors, summ_tms[measure][idx],
                    yerr=summ_tms[measure + '_sem'][idx],
                    color='tab:blue', marker='s', ms=2.5, lw=1, capsize=2.5,
                    elinewidth=.8, label='TMS, operating dose')
        # factor 1.0 is the published value
        ax.axvline(1., color='r', ls='--', lw=1)
        ax.set_title(pname)
        ax.set_xlabel('relative value')
        if i % ncol == 0:
            ax.set_ylabel(measure)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


def plot_seizure_boundary(labels, values, factors, summ, param='alpha1',
                          fname=None, show=False, title=None):
    """
    Seizure index against one parameter, with the criterion threshold and the
    published operating point marked. Takes the output of sweep_marginal, run
    at the operating dose.

    The Methods currently justify moving alpha1 from -0.5 to -0.45 by saying
    that the original value produced seizure-like behaviour under simulated
    TMS. As written that is an unquantified observation, and it is the single
    sentence that most invites Reviewer 2's "narrow stability regime" reading.
    This panel replaces it with a measured boundary.

    The seizure index is the proportion of trials on which the conjunction
    population falls into a synchronous period-2 oscillation: every unit
    alternating in phase between a positive value and zero on successive
    timesteps, with selectivity destroyed. See the trial loop in
    plasticattractor_networksims_general.py for the criterion, which is a
    strongly negative lag-1 autocorrelation of the population mean combined
    with a high mean pairwise correlation between units.

    """

    import matplotlib.pyplot as plt

    labels = np.array(labels)
    idx = labels == param              # settings for this parameter only
    base = get_param(default_params(), param)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    # x axis is the absolute parameter value here
    ax.errorbar(np.array(values)[idx], summ['seizure_index'][idx],
                yerr=summ['seizure_index_sem'][idx],
                color='k', marker='o', ms=3, lw=1, capsize=3, elinewidth=.9)
    ax.axhline(default_task()['seizure_thresh'], color='r', ls=':', lw=1,
               label='seizure criterion')
    ax.axvline(base, color='r', ls='--', lw=1, label='publication value')
    ax.set_xlabel(param)
    ax.set_ylabel('seizure index')
    ax.legend(fontsize=8)

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


def plot_grid(a_vals, b_vals, summ, measure='accuracy', param_a='alpha1',
              param_b='alpha5', base=None, fname=None, show=False, title=None):
    """
    Heatmap of one measure over the two parameter plane, with the published
    operating point marked and the seizure criterion overlaid as a contour.

    Takes the 2-D summary dict from sweep_grid, whose arrays are indexed
    [a index, b index].
    """

    import matplotlib.pyplot as plt

    if base is None:
        base = default_params()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # imshow wants [row, col] = [y, x], and summ is [a, b] with a on the x
    # axis, so transpose. extent puts the axes in parameter units.
    im = ax.imshow(summ[measure].T, origin='lower', aspect='auto',
                   extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                   cmap='viridis')
    fig.colorbar(im, ax=ax, label=measure)

    # seizure boundary. A and B are built with 'ij' indexing so they match the
    # (a, b) ordering of the summary arrays directly, no transpose needed.
    A, B = np.meshgrid(a_vals, b_vals, indexing='ij')
    ax.contour(A, B, summ['seizure_index'],
               levels=[default_task()['seizure_thresh']],
               colors='w', linewidths=1.5)

    ax.plot(get_param(base, param_a), get_param(base, param_b),
            marker='*', color='r', ms=16, ls='none', label='publication value')
    ax.set_xlabel(param_a)
    ax.set_ylabel(param_b)
    ax.legend(loc='lower right')

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


def plot_dimensions(dims, summ, measure='accuracy', fname=None, show=False,
                    title=None):
    """
    One measure against the number of stimulus dimensions, at fixed network
    size.

    Responses stay at two whatever D is, so chance is a constant 0.50 and is
    drawn as a flat reference line when plotting accuracy. For any other
    measure the chance line is omitted.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.errorbar(dims, summ[measure], yerr=summ[measure + '_sem'],
                color='k', marker='o', ms=3, lw=1.2, capsize=3,
                elinewidth=.9, label=measure)

    if measure == 'accuracy':
        # two responses at every D, so chance is flat
        ax.axhline(.5, color='r', ls=':', lw=1, label='chance')
        ax.set_ylim(0, 1.02)

    ax.set_xticks(dims)
    ax.set_xlabel('stimulus dimensions (one relevant, the rest ignored)')
    ax.set_ylabel(measure)
    ax.legend(fontsize=8)

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


def plot_noise(eps_vals, summ, measure='accuracy', fname=None, show=False,
               title=None):
    """
    One measure against the noise scale, on a log x axis since eps_list spans
    more than an order of magnitude. The publication value (0.005) is marked.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.errorbar(eps_vals, summ[measure], yerr=summ[measure + '_sem'],
                color='k', marker='o', ms=3, lw=1, capsize=3, elinewidth=.9)
    ax.axvline(default_params()['epsilon'], color='r', ls='--', lw=1,
               label='publication value')
    ax.set_xscale('log')
    ax.set_xlabel('noise scale (epsilon)')
    ax.set_ylabel(measure)
    ax.legend(fontsize=8)

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=300)
    _show_or_close(fig, show)
    return fig


# %% ===================== example activity blocks =====================

def plot_example_blocks(configs, seed=0, num_blocks=1, blk=0, max_configs=6,
                        n_trials_shown=6, show_rule=True, fname=None,
                        show=False, title=None):
    """
    Run one simulation per config and plot an example block of activity, so
    that the numbers coming out of a sweep can be checked against what the
    network is actually doing at each setting.

    Each config gets three stacked panels, laid out as in
    plasticattracto_centralscript.py:

        top     motor (response) feature units, as line plots
        middle  conjunction units, as line plots, with the lateral inhibition
                threshold beta marked
        bottom  sensory feature units, as a seaborn heatmap

    Separating the motor units from the sensory ones matters for reading the
    figure: they are the same variety of unit in the model, but the motor
    units carry the decision and the sensory units carry the stimulus
    representation, and overlaying them hides both.

    The rule presentation epochs are drawn first, shaded, and separated from
    the stimulus trials by a heavier line. There is one rule epoch per
    stimulus-response mapping. Including them matters because that is where
    the attractors are formed: a setting that performs badly because the rule
    never imprinted on the weight matrix looks quite different from one where
    the rule imprinted and the response competition then failed, and the
    summary measures do not distinguish those.

    Only the first n_trials_shown stimulus trials are drawn, since a full
    block at 400 timesteps per trial compresses to an unreadable smear. Set
    show_rule=False to omit the rule epochs and show trials only.

    Arguments
        configs        list from any configs_* builder. Pass the same call the
                       sweep used and the panels correspond one for one with
                       the points in the sweep figures.
        seed           which seed to run. One seed only: this is a qualitative
                       check on the dynamics, not a measurement, and the
                       accuracy printed in each panel title is that single
                       run rather than the sweep mean.
        num_blocks     blocks to simulate; only block `blk` is plotted, but
                       running more lets you look at a later block once the
                       long term weights have settled.
        max_configs    if there are more configs than this, an evenly spaced
                       subset is drawn and a note is printed. Three panels per
                       config makes tall figures, so this is deliberately low.
        n_trials_shown trials to draw, counting from the first stimulus trial
                       after the rule epochs.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # subsample if the sweep has more settings than will fit on a page
    if len(configs) > max_configs:
        idx = np.linspace(0, len(configs) - 1, max_configs).astype(int)
        print('plot_example_blocks: showing %d of %d configs'
              % (max_configs, len(configs)))
        configs = [configs[i] for i in idx]

    n = len(configs)

    # three rows per config: motor, conjunction, sensory heatmap
    fig, axes = plt.subplots(3 * n, 1, figsize=(11, 4.2 * n), squeeze=False,
                             gridspec_kw={'height_ratios': [1, 1, 1.1] * n})
    axes = axes[:, 0]

    for i, c in enumerate(configs):

        # run this one setting, keeping the activity this time
        out = plasticattractor_sim(num_blocks, c['num_trials'], seed,
                                   tms_dose=c['tms_dose'],
                                   params=c['params'], task=c['task'],
                                   return_activity=True)

        conj = out['conj'][blk]        # units x time x trials
        feat = out['feat'][blk]

        built = build_task(c['task'])
        resp_slice = built['resp_slice']
        n_sensory = built['num_stim_features']

        # The rule epochs sit at the front of the trial axis, one per
        # stimulus-response mapping. Start from trial 0 to include them, or
        # from the first stimulus trial if show_rule is off.
        n_rule = conj.shape[2] - c['num_trials']
        first = 0 if show_rule else n_rule
        stop = min(n_rule + n_trials_shown, conj.shape[2])

        # concatenate the chosen trials along time into one continuous trace
        conj_blk = np.concatenate(
            [conj[:, :, t] for t in range(first, stop)], axis=1)
        feat_blk = np.concatenate(
            [feat[:, :, t] for t in range(first, stop)], axis=1)

        period = c['task']['trial_period']
        rule_end = (n_rule - first) * period
        n_time = conj_blk.shape[1]
        beta = c['params']['beta']

        ax_motor = axes[3 * i]
        ax_conj = axes[3 * i + 1]
        ax_feat = axes[3 * i + 2]

        # ---- top: motor units ----------------------------------------
        # the palette of plasticattracto_centralscript.py where the unit
        # counts match, and a colormap otherwise
        motor = feat_blk[resp_slice, :]
        if motor.shape[0] == 2:
            motor_cols = ['darkviolet', 'orchid']
        else:
            motor_cols = plt.cm.plasma(
                np.linspace(.1, .75, motor.shape[0]))
        for u in range(motor.shape[0]):
            ax_motor.plot(motor[u], lw=1, color=motor_cols[u],
                          label='response %d' % (u + 1))
        ax_motor.set_ylabel('motor')
        ax_motor.legend(fontsize=6, ncol=motor.shape[0], loc='upper right')
        ax_motor.set_title('%s  (accuracy %.2f, eig %.2f)'
                           % (c['label'], out['accuracy'], out['mean_eig']), fontsize=9)

        # ---- middle: conjunction units -------------------------------
        if conj_blk.shape[0] == 4:
            conj_cols = ['lightseagreen', 'teal', 'darkturquoise',
                         'dodgerblue']
        else:
            conj_cols = plt.cm.winter(
                np.linspace(0, 1, conj_blk.shape[0]))
        for u in range(conj_blk.shape[0]):
            ax_conj.plot(conj_blk[u], lw=1, color=conj_cols[u])
        ax_conj.axhline(beta, color='k', ls='--', lw=.8)
        ax_conj.set_ylabel('conjunction')

        for ax in (ax_motor, ax_conj):
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, n_time)
            ax.margins(x=0)
            # trial boundaries
            for b in range(1, stop - first):
                ax.axvline(b * period, color='.75', lw=.5)
            if rule_end > 0:
                ax.axvspan(0, rule_end, color='.92', zorder=0)
                ax.axvline(rule_end, color='k', lw=1.2)
        if rule_end > 0:
            ax_motor.text(rule_end / 2, 1.0,
                          'rule (%d epoch%s)'
                          % (n_rule, '' if n_rule == 1 else 's'),
                          ha='center', va='top', fontsize=7, color='.3')

        # ---- bottom: sensory feature units as a heatmap ---------------
        # labelled by dimension and feature, e.g. d1f2 is the second feature
        # of the first stimulus dimension
        ylabels = ['d%df%d' % (d + 1, f + 1)
                   for d in range(c['task']['num_dims'])
                   for f in range(c['task']['num_features_per_dim'])]
        sns.heatmap(feat_blk[:n_sensory, :], cmap='Greys', cbar=False,
                    vmin=0, vmax=1, xticklabels=False, yticklabels=ylabels,
                    ax=ax_feat)
        ax_feat.set_ylabel('sensory')
        ax_feat.tick_params(axis='y', labelsize=6, rotation=0)
        if rule_end > 0:
            ax_feat.axvline(rule_end, color='r', lw=1.2)
        if i == n - 1:
            ax_feat.set_xlabel('timestep')

    fig.tight_layout()
    _suptitle(fig, title)
    if fname:
        fig.savefig(fname, dpi=200)
    _show_or_close(fig, show)
    return fig
