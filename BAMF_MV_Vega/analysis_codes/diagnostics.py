import pystan
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger("pystan")


def init_nothing():
    return {}


def check_n_eff(fit, pars=None, verbose=True):

    verbosity = int(verbose)

    n_iter = sum(fit.sim["n_save"]) - sum(fit.sim["warmup2"])

    if pars is None:
        pars = fit.sim["fnames_oi"]
    else:
        if isinstance(pars, string_types):
            pars = [pars]
        pars = _remove_empty_pars(pars, fit.sim["pars_oi"], fit.sim["dims_oi"])
        allpars = fit.sim["pars_oi"] + fit.sim["fnames_oi"]
        _check_pars(allpars, pars)
        packed_pars = set(pars) - set(fit.sim["fnames_oi"])
        if packed_pars:
            unpack_dict = {}
            for par_unpacked in fit.sim["fnames_oi"]:
                par_packed = par_unpacked.split("[")[0]
                if par_packed not in unpack_dict:
                    unpack_dict[par_packed] = []
                unpack_dict[par_packed].append(par_unpacked)
            pars_unpacked = []
            for par in pars:
                if par in packed_pars:
                    pars_unpacked.extend(unpack_dict[par])
                else:
                    pars_unpacked.append(par)
            pars = pars_unpacked

    par_n_dict = {}
    for n, par in enumerate(fit.sim["fnames_oi"]):
        par_n_dict[par] = n

    no_warning = True
    ratio_ls = {}
    for name in pars:
        n = par_n_dict[name]
        n_eff = pystan.chains.ess(fit.sim, n)
        ratio = n_eff / n_iter

        if (ratio < 0.001) or np.isnan(ratio) or np.isinf(ratio):
            if verbosity > 1:
                logger.warning(
                    "n_eff / iter for parameter {} is {:.3g}!".format(name, ratio)
                )

            no_warning = False
            if verbosity <= 1:
                break
        ratio_ls[name] = ratio
    return ratio_ls


def check_rhat(fit, pars=None, verbose=True):
    """Checks the potential scale reduction factors, i.e., Rhat values

    Parameters
    ----------
    fit : StanFit4Model object
    pars : {str, sequence of str}, optional
        Parameter (or quantile) name(s). Test only specific parameters.
        Raises an exception if parameter is not valid.
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there are Rhat values
        too far from 1. If ``verbose`` is an integer greater than 1,
        parameter (quantile) diagnostics are printed. If ``verbose``
        is an integer greater than 2, then extra diagnostic messages are printed.


    Returns
    -------
    bool
        ``True`` if there are no problems with with Rhat and ``False``
        otherwise.

    """

    verbosity = int(verbose)

    if pars is None:
        pars = fit.sim["fnames_oi"]
    else:
        if isinstance(pars, string_types):
            pars = [pars]
        pars = _remove_empty_pars(pars, fit.sim["pars_oi"], fit.sim["dims_oi"])
        allpars = fit.sim["pars_oi"] + fit.sim["fnames_oi"]
        _check_pars(allpars, pars)
        packed_pars = set(pars) - set(fit.sim["fnames_oi"])
        if packed_pars:
            unpack_dict = {}
            for par_unpacked in fit.sim["fnames_oi"]:
                par_packed = par_unpacked.split("[")[0]
                if par_packed not in unpack_dict:
                    unpack_dict[par_packed] = []
                unpack_dict[par_packed].append(par_unpacked)
            pars_unpacked = []
            for par in pars:
                if par in packed_pars:
                    pars_unpacked.extend(unpack_dict[par])
                else:
                    pars_unpacked.append(par)
            pars = pars_unpacked

    par_n_dict = {}
    for n, par in enumerate(fit.sim["fnames_oi"]):
        par_n_dict[par] = n

    no_warning = True
    rhat_ls = {}
    for name in pars:
        n = par_n_dict[name]
        # rhat = pystan.chains.splitrhat(fit.sim, n)
        rhat = splitrhat(fit.sim, n)
        rhat_ls[name] = rhat
        if np.isnan(rhat) or np.isinf(rhat) or (rhat > 1.1) or (rhat < 0.9):

            if verbosity > 1:
                logger.warning("Rhat for parameter {} is {:.3g}!".format(name, rhat))

            no_warning = False
            if verbosity <= 1:
                break

    if no_warning:
        if verbosity > 2:
            logger.info("Rhat looks reasonable for all parameters")

    else:
        if verbosity > 0:
            logger.warning(
                "Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed"
            )

    return rhat_ls


def splitrhat(sim, idx):
    """Calculate rhat

    Parameters
    ----------
    sim : chains
    n : int
        Parameter index starting from 0
    """
    try:
        rhat = split_potential_scale_reduction(sim, idx)
    except (ValueError, ZeroDivisionError):
        rhat = np.nan
    return rhat


def get_kept_samples(sim, k, n):
    """

    Parameters
    ----------
    k : unsigned int
        Chain index
    n : unsigned int
        Parameter index
    """

    warmup2 = np.array(sim["warmup2"])

    slst = sim["samples"][k]["chains"]  # chain k, an OrderedDict
    param_names = list(slst.keys())  # e.g., 'beta[1]', 'beta[2]', ...

    nv = slst[param_names[n]]  # parameter n
    samples = []
    for i in range(nv.shape[0] - warmup2[k]):
        samples.append(nv[warmup2[k] + i])
    return samples


def split_potential_scale_reduction(sim, idx):
    """
    Return the split potential scale reduction (split R hat) for the
    specified parameter.

    Current implementation takes the minimum number of samples
    across chains as the number of samples per chain.

    Parameters
    ----------
    n : unsigned int
        Parameter index

    Returns
    -------
    rhat : float
        Split R hat

    """
    n_chains = sim["chains"]
    ns_kept = [s - w for s, w in zip(sim["n_save"], sim["warmup2"])]
    n_samples = int(min(ns_kept))

    if n_samples % 2 == 1:
        n_samples = n_samples - 1

    split_chain_mean = []
    split_chain_var = []
    for chain in range(n_chains):
        samples = get_kept_samples(sim, chain, idx)
        split_chain = []
        for i in range(int(n_samples / 2)):
            split_chain.append(samples[i])
        split_chain_mean.append(np.mean(split_chain))
        split_chain_var.append(np.var(split_chain, ddof=1))

        split_chain = []
        for i in range(int(n_samples / 2), n_samples):
            split_chain.append(samples[i])
        split_chain_mean.append(np.mean(split_chain))
        split_chain_var.append(np.var(split_chain, ddof=1))

    var_between = n_samples / 2 * np.var(split_chain_mean, ddof=1)
    var_within = np.mean(split_chain_var)

    srhat = np.sqrt((var_between / var_within + n_samples / 2 - 1) / (n_samples / 2))
    return srhat


def get_kept_samples_sorted(data, chain, *idx):
    """

    Parameters
    ----------
    k : unsigned int
        Chain index
    n : unsigned int
        Parameter index
    """

    extract = data[:, chain]

    i = 0
    while i < len(idx) and idx:
        extract = extract[:, idx[i]]
        i = i + 1

    return extract


def split_potential_scale_reduction_sorted(n_samples, n_chains, data, *idx):
    """
    Return the split potential scale reduction (split R hat) for the
    specified parameter.

    Current implementation takes the minimum number of samples
    across chains as the number of samples per chain.

    Parameters
    ----------
    n : unsigned int
        Parameter index
    Returns
    -------
    rhat : float
        Split R hat

    """

    if n_samples % 2 == 1:
        n_samples = n_samples - 1

    split_chain_mean = []
    split_chain_var = []

    for chain in range(n_chains):

        samples = get_kept_samples_sorted(data, chain, *idx)
        split_chain = []
        for i in range(int(n_samples / 2)):
            split_chain.append(samples[i])
        split_chain_mean.append(np.mean(split_chain))
        split_chain_var.append(np.var(split_chain, ddof=1))

        split_chain = []
        for i in range(int(n_samples / 2), n_samples):
            split_chain.append(samples[i])
        split_chain_mean.append(np.mean(split_chain))
        split_chain_var.append(np.var(split_chain, ddof=1))

    var_between = n_samples / 2 * np.var(split_chain_mean, ddof=1)
    var_within = np.mean(split_chain_var)

    srhat = np.sqrt((var_between / var_within + n_samples / 2 - 1) / (n_samples / 2))
    return srhat
