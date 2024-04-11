from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import binom_test, kruskal, ks_2samp, pearsonr, spearmanr
from time import ctime
from tqdm import tqdm


from . import plotting, itx_helper
from .helpers import (
    Collated_output,
    calculate_Z_contribution,
    distance_label_batch,
    distance_label_batch_by_Z,
    dump_hdf5,
    get_plot_filename,
    normalize,
    normalize_chains,
    sort_by_batch_label,
)


def get_generic_data(filename, labels=["F", "G", "X_orig", "error"]):
    """extract the matrixes from filename.

    Args:
        filename (_type_): file position
        labels (list, optional): extracted matrixes name. Defaults to ["F", "G", "X_orig", "error"].

    Returns:
        _type_: extracted information, dict
    """
    result = {}

    with h5py.File(filename, "r") as result_file:
        for label in labels:
            if label == "selected_indexes":
                result[label] = result_file[label][()] - 1
            else:
                result[label] = result_file[label][()]

    return result


def get_axis(shape):
    """find the axis where the length equals 1.

    Args:
        shape (_type_): shape of array

    Returns:
        _type_: _description_
    """
    if len(shape) == 1:
        return 0
    for i, s in enumerate(shape):
        if s == 1:
            return i
    return 0


def get_generic_data_from_multiple_files(
    filenames,
    labels=["F", "G", "X_orig", "error", "alpha_n", "alpha_r", "alpha_s", "norm_const"],
    singles=["X_orig", "X", "n", "p", "m", "error", "sigma", "norm_const"],
):

    combined = {}

    results = [get_generic_data(f, labels) for f in filenames]
    for item in results:
        norm_and_scale(item)

    # if "F_non_selected" in results[0].keys():

    for key in results[0].keys():
        if key in singles:
            combined[key] = results[0][key]

            # test whether this key's value are the same among all the multiple results.
            compared = {}
            for idx, f in enumerate(filenames[1:], start=1):
                # changed:
                if idx == 0:
                    continue
                # Returns True if two arrays are element-wise equal
                compared[f] = np.allclose(results[0][key], results[idx][key])

            if np.any([not j for j in compared.items()]):
                print(key, compared, flush=True)
            else:
                print(key, "The files have matching variables.", flush=True)

            continue
        
        # combined the multiple results into one dictionary
        combined[key] = np.stack(
            [r[key].squeeze() for r in results], axis=get_axis(results[0][key].shape)
        )
        #TODO: same starting point
        # if combined[key].shape[0] == 1:
        #     print(1)
    if "F_non_selected" in labels:
        
        combined["F_selected"] = combined.pop("F")
        combined["F"] = combined.pop("F_non_selected")

    #     combined["X_full"] = combined.pop("X")
    #     combined["X"] = combined.pop("Xv")
    #     combined["sigma_full"] = combined.pop("sigma")
    #     combined["sigma"] = combined.pop("sigmav")
    return combined


def norm_and_scale(data):
    """normalize the input X, sigma and G matrix.

    Args:
        data (_type_): all the input data
    """

    if "X" in data.keys():
        data["X"] = data["X"] * data["norm_const"]
    else:
        data["X_orig"] = data["X_orig"] * data["norm_const"]

    if "sigma" in data.keys():
        data["sigma"] = data["sigma"] * data["norm_const"]
    else:
        data["error"] = data["error"] * data["norm_const"]

    data["G"] = data["G"] * data["norm_const"]


# ----------------------- changed--------------------
# def reshaper(x, n=1000):
#     return np.reshape(x[n:], [-1, x.shape[-2], x.shape[-1]])


def reshaper_1d(x, n=1000):

    return np.reshape(x[n:], [-1, x.shape[-1]])


def reshaper(x, n=1000):
    """This function reshape the sample of input dataframe into the shape of second
    axis as 1. The input data contains multiple runs results, we take samples of
    each runs and merge them into one run.

    Args:
        x (_type_): input multiple runs results
        n (int, optional): sampler size. Defaults to 1000.

    Returns:
        _type_: reshaped results as one run result.
    """
    return np.reshape(x[n:], (-1,) + x.shape[2:])


def merge_chains(data, n=1000, partial=False):
    """Extract the last n draws from each chain, merge them into one array.,
    sequence as draw 1 from chain 1, draw 1 from chain 2,.. , draw 1 from chain n;
    draw 2 from chain 1 to n.

    Args:
        data (_type_): multiple chain results
        n (int, optional): sample size extracted from each run. Defaults to 1000.
        partial (bool, optional): 
    """

    autocorrelation_parameters = [i for i in data.keys() if i.startswith("alpha")]

    # Sometimes all you have is one chain... 
    if data["F"].shape[1] == 1:
        data["F"] = data["F"][n:].squeeze()
        data["G"] = data["G"][n:].squeeze()
        for item in autocorrelation_parameters:
            data[item] = data[item][n:].squeeze()

    else:

        data["F"] = reshaper(data["F"], n)
        data["G"] = reshaper(data["G"], n)

        for item in autocorrelation_parameters:
            data[item] = reshaper(data[item], n)

        if partial:
            data["height"] = reshaper_1d(data["height"], n)

def merge_chains_vb(data, n=1000, partial=False):
    """Extract the last n draws from each chain, merge them into one array.,
    sequence as draw 1 from chain 1, draw 1 from chain 2,.. , draw 1 from chain n;
    draw 2 from chain 1 to n.

    Args:
        data (_type_): multiple chain results
        n (int, optional): sample size extracted from each run. Defaults to 1000.
        partial (bool, optional): 
    """

    autocorrelation_parameters = [i for i in data.keys() if i.startswith("alpha")]

    # Sometimes all you have is one chain... Q: why take the warmup + sampler
    if data["F"].shape[1] == 1:
        # data["F"] = data["F"].squeeze()
        # data["G"] = data["G"].squeeze()
        # for item in autocorrelation_parameters:
        #     data[item] = data[item].squeeze()

        # if partial:
        #     data["height"] = data["height"].squeeze()
        # data["F"] = data["F"][n:]
        # data["G"] = data["G"][n:]
        
        # for item in autocorrelation_parameters:
        #     data[item] = data[item][n:]

        data["F"] = data["F"][-1]
        data["G"] = data["G"][-1]
        
        for item in autocorrelation_parameters:
            data[item] = data[item][-1]

    else:
        size = min(data["F"].shape[0],1)
        data["F"] = reshaper(data["F"][-size:], n)
        data["G"] = reshaper(data["G"][-size:], n)

        for item in autocorrelation_parameters:
            # data[item] = reshaper(data[item][-size:], n)
            data[item] = np.reshape(data[item][-size:][n:], (-1,) + data[item][-size:].shape[2:])

        if partial:
            data["height"] = reshaper_1d(data["height"][-size:], n)




def spear(u, v):
    coeff, _ = spearmanr(u, v, axis=None)
    return coeff


def scale_times(times):
    """normalize the time by the median time

    Args:
        times (_type_): _description_

    Returns:
        _type_: _description_
    """

    timediff = times - times[0]

    """ Marta modifying this"""
    timediff = pd.to_timedelta(timediff) #Marta did that
    medtime = np.median(np.diff(timediff.total_seconds().values))

    scaletime = timediff.total_seconds() / medtime
    """medtime = np.median(np.diff(timediff.values)) 
    scaletime = timediff.values / medtime"""
    return scaletime, medtime


def sort_by_input_F(data, F_orig=None, metric="cosine"):

    if F_orig is None:
        indices = distance_label_batch(data["F"], data["F_orig"], metric=metric)
    else:
        indices = distance_label_batch(data["F"], F_orig, metric=metric)

    data["F"] = sort_by_batch_label(data["F"], indices)
    data["G"] = sort_by_batch_label(data["G"], indices, axis=3)
    data["G"] = np.swapaxes(data["G"], -1, -2)

    autocorrelation_parameters = ["alpha_n", "alpha_r", "alpha_s"]
    for item in autocorrelation_parameters:
        if item in data.keys():
            data[item] = sort_by_batch_label(data[item], indices, axis="1d")


def sort_by_Z_contribution(
    data, take=slice(None), use_exemplars=None, metric="correlation"
):
    """sort the G and F matrix by the Z distribution for each chain based on their last sample

    Args:
        data (_type_): draw samples
        take (_type_, optional): index for the wanted extracted samples. Defaults to slice(None).
        use_exemplars (_type_, optional): _description_. Defaults to None.
        metric (str, optional): _description_. Defaults to "correlation".

    Returns:
        _type_: _description_
    """

    # calculate the value to use as benchmark
    if use_exemplars is None:
        # each chain's last draw (F and G)
        subset_F = data["F"][take, :, :]
        subset_G = data["G"][take, :, :]

        print("Sorting sample sizes:", subset_F.shape, subset_G.shape, flush=True)
        n = subset_F.shape[0]
        exemplars = [
            calculate_Z_contribution(subset_F[i, :, :], subset_G[i, :, :])
            for i in (range(n))
        ]  # use the last sample for each chain as benchmark

    else:
        n = 1
        exemplars = use_exemplars

    # calculate the component order of each samples based on benchmark,
    # size: n_chain * (n_component* n_draws)

    orders = np.stack(
        [
            distance_label_batch_by_Z(
                data["F"], data["G"], exemplars[i], metric=metric
            ).flatten()
            for i in (range(n))
        ]
    )

    # select the chain with most occurence component sequence result
    ind, count = np.unique(orders, axis=0, return_counts=True)

    print("There are {} possible orderings".format(len(count)), flush=True)
    print("Each ordering has appeared:", count, flush=True)

    indices = ind[np.argmax(count)].reshape(data["F"].shape[0:2])

    print("Sorting starts...", flush=True)
    data_sorted = data
    data_sorted["F"] = sort_by_batch_label(data_sorted["F"], indices)
    data_sorted["G"] = sort_by_batch_label(data_sorted["G"], indices, axis=3)
    # TODO: change this two axeses due to numpy array because, after slicing python automatically
    # switch this two axese, don't know why
    data_sorted["G"] = np.swapaxes(data_sorted["G"], -1, -2)

    autocorrelation_parameters = [
        i for i in data_sorted.keys() if i.startswith("alpha")
    ]
    for item in autocorrelation_parameters:
        data_sorted[item] = sort_by_batch_label(data_sorted[item], indices, axis="1d")

    return data_sorted, indices


def z_scale(residuals):

    return (residuals - residuals.mean(axis=-2, keepdims=True)) / residuals.std(
        axis=-2, keepdims=True
    )


def reconstruct(F, G):
    return np.einsum("aij, ajk -> aik", G, F)


def match_labels(
    values,
    exemplar_frame,
    cutoff=0.1,
    distance_metric="cityblock",
    unknown_label="Unknown",
):
    """
    match the solved results to the benchmark and test the solved results is accepteable or not,
    assign label for each components.

    Args:
        values (_type_): solved results, draws
        exemplar_frame (_type_): benchmark result
        cutoff (float, optional): limit value to decide the acceptance of this solution . Defaults to 0.1.
        distance_metric (str, optional): the metric to calculate distance. Defaults to "cityblock".
        unknown_label (str, optional): lable to assign to unrecognized component. Defaults to "Unknown".

    Returns:
        _type_: _description_
    """

    distances = cdist(exemplar_frame, values, distance_metric)

    row_idx, col_idx = linear_sum_assignment(distances)

    accepted = distances[row_idx, col_idx] < cutoff
    selected_row = row_idx[accepted]
    selected_col = col_idx[accepted]
    rejected = col_idx[~accepted]

    # Match the draws label with the exemplars

    labels = exemplar_frame.index.values[selected_row]

    # Assign "Unknown" label to the rejected values
    selections = np.append(selected_col, rejected)

    labels = np.append(labels, np.repeat(unknown_label, len(rejected)))

    return selections, labels


def match_labels_old(
    values,
    exemplar_frame,
    cutoff=0.1,
    distance_metric="cityblock",
    unknown_label="Unknown",
):
    """
    match the solved results to the benchmark and test the solved results is accepteable or not,
    assign label for each components.

    Args:
        values (_type_): solved results, draws
        exemplar_frame (_type_): benchmark result
        cutoff (float, optional): limit value to decide the acceptance of this solution . Defaults to 0.1.
        distance_metric (str, optional): the metric to calculate distance. Defaults to "cityblock".
        unknown_label (str, optional): lable to assign to unrecognized component. Defaults to "Unknown".

    Returns:
        _type_: _description_
    """

    distances = cdist(exemplar_frame, values, distance_metric)

    row_idx, col_idx = linear_sum_assignment(distances)
    # print(row_idx, col_idx)
    accepted = distances[row_idx, col_idx] < cutoff
    # print(accepted)
    # print(distances[row_idx, col_idx], accepted)
    selected_row = row_idx[accepted]
    # print("selected row", selected_row)
    selected_col = col_idx[accepted]
    # print("selected col", selected_col)
    rejected = col_idx[~accepted]
    # print("rejected", rejected)

    # Match the draws label with the exemplars
    ordering = np.argsort(selected_row)
    # print("selected_row", selected_row, "ordering", ordering)
    selected_col = selected_col[ordering]
    # print("selected_col", selected_col)
    labels = exemplar_frame.index.values[selected_row[ordering]]

    # Assign "Unknown" label to the rejected values
    selections = np.append(selected_col, rejected)

    labels = np.append(labels, np.repeat(unknown_label, len(rejected)))

    return selections, labels


def sort_label_by_truth(match_file, data, cutoff, distance_metric):

    match_profiles = pd.read_hdf(match_file, "F")

    if len(data["F"].shape) == 2:
        median_F = data["F"]
    else:
        median_F = np.median(data["F"], axis=0)

    if "selected_indexes" in data and "F_selected" in data: # bamf anchor model
        match_profiles = match_profiles.drop(match_profiles.columns[data["selected_indexes"][0]],axis=1)

    elif "selected_indexes" in data and "F_selected" not in data: #bamf_subset model
        match_profiles = match_profiles.drop(match_profiles.columns[data["selected_indexes"][0]],axis=1)
        median_F = np.delete(median_F,data["selected_indexes"][0],axis=1)

    data["orig_labels"] = list(match_profiles.index.values)

    # sort the component index of extracted draws from each chain match to the benchmark based on F matrix


    matched_indices, labels = match_labels(
        median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric
    )

    all_indices = set(np.arange(match_profiles.shape[0]))

    # unmatched component index
    unresolved = list(set(all_indices) - set(matched_indices))

    indices = [*matched_indices, *unresolved]
    autocorr = [i for i in data.keys() if i.startswith("alpha")]

    if len(data["F"].shape) == 2:
        data["F"] = data["F"][indices, :]
        data["G"] = data["G"][:, indices]
        for item in autocorr:
            data[item] = data[item][indices]
    else:
        data["F"] = data["F"][:, indices, :]
        data["G"] = data["G"][:, :, indices]
        for item in autocorr:
            data[item] = data[item][:, indices]

    data["labels"] = labels  # component name

    # assign questionmark to unmatched component result
    if len(unresolved) > 0:
        for kk in range(len(unresolved)):
            data["labels"] = np.append(data["labels"], "?" + str(kk))

    data["reference_F"] = match_profiles
    return data


def sort_label_by_truth_map(match_file, data, cutoff, distance_metric):

    match_profiles = pd.read_hdf(match_file, "F")

    if "selected_indexes" in data or "F_selected" in data:
        
        match_profiles = match_profiles.drop(match_profiles.columns[data["selected_indexes"][0]],axis=1)

    data["orig_labels"] = list(match_profiles.index.values)

    # sort the component index of extracted draws from each chain match to the benchmark based on F matrix
    unresolved_ls = []
    for i in (range(data["F"].shape[0])):
        median_F = data["F"][i]
        matched_indices, labels = match_labels(
            median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric
        )
        
        all_indices = set(np.arange(match_profiles.shape[0]))
        # unmatched component index
        unresolved = list(set(all_indices) - set(matched_indices))

        indices = [*matched_indices, *unresolved]

        autocorr = [i for i in data.keys() if i.startswith("alpha")]


        data["F"][i] = data["F"][i, indices, :]
        data["G"][i] = np.swapaxes(data["G"][i, :, indices], -1, -2)

        for item in autocorr:
            data[item][i] = data[item][i, indices]

        data["labels"] = labels  # component name

        # assign questionmark to unmatched component result
        if "Unknown" in labels:
            unresolved_ls.append(i)
            # for kk in range(len(unresolved)):
            #     data["labels"] = np.append(data["labels"], "?" + str(kk))

        data["reference_F"] = match_profiles
    print("unresolved iterations:",len(unresolved_ls),"index:",unresolved_ls)
    data["F"] = np.delete(data["F"],unresolved_ls,axis=0)
    data["G"] = np.delete(data["G"],unresolved_ls,axis=0)
    for item in autocorr:
        data[item] = np.delete(data[item],unresolved_ls,axis=0)

    return data



def summarize_frame(data, benchmark=False):
    print("Stats calculating...")
    numbers = {}

    print("median solution",ctime(),flush=True)

    median_F = np.median(data["F"], axis=0)
    median_G = np.median(data["G"], axis=0)
    if "selected_indexes" in data:
        residuals =  np.delete(data["X"], data["selected_indexes"][0],axis=1) -data["Z"]
        respererr = residuals / np.delete(data["sigma"], data["selected_indexes"][0],axis=1)
    else:
        residuals = data["X"] - data["Z"]
        respererr = residuals / data["sigma"]

    print("descriptive",ctime(), flush=True)
    residuals_abs = np.abs(residuals)
    numbers["Mean(abs(residual))"] = np.mean(residuals_abs)
    numbers["Median(abs(residual))"] = np.median(residuals_abs)
    numbers["Max(abs(residual))"] = np.max(residuals_abs)

    numbers["Mean(abs(residual/error))"] = np.mean(np.abs(respererr))
    numbers["Median(abs(residual/error))"] = np.median(np.abs(respererr))
    numbers["Max(abs(residual/error))"] = np.max(np.abs(respererr))

    print("z-data",ctime(), flush=True)

    d_means = data["Z"].mean(axis=(0, 1))
    d_stds = data["Z"].std(axis=(0, 1))
    if "selected_indexes" in data:
        z_data = (np.delete(data["X"], data["selected_indexes"][0],axis=1)- d_means) / d_stds
    else:
        z_data = (data["X"] - d_means) / d_stds

    print("Q statistics",ctime(), flush=True)
    numbers["Qm"] = np.sum(z_data**2)
    numbers["Qexp"] = np.prod(data["X"].shape)  # - p*(np.sum(data["X_orig"].shape))

    print("Z statistics",ctime(), flush=True)
    z_mean = np.mean(z_data)
    z_std = np.std(z_data)
    numbers["Z scaled mean"] = z_mean
    numbers["Z scaled std"] = z_std

    print("Alpha statistics",ctime(), flush=True)
    alphas = [i for i in data.keys() if i.startswith("alpha")]
    for item in alphas:
        print(f"{item} mean", flush=True)
        means = np.mean(data[item], axis=0)
        print(f"{item} median", flush=True)
        medians = np.median(data[item], axis=0)
        print(f"{item} std", flush=True)
        stds = np.std(data[item], axis=0)
        for i in range(means.shape[0]):
            numbers[f"{item}{i} mean"] = means[i]
            numbers[f"{item}{i} median"] = medians[i]
            numbers[f"{item}{i} std"] = stds[i]

    if benchmark:
        # Only for matching labels to external reference

        ids = [(i, l) for i, l in enumerate(data["labels"]) if l in data["orig_labels"]]

        numbers["identified components"] = len(ids)

        for index, l in ids:

            print(f"{l} statistics",ctime(), flush=True)

            data_F = median_F[index, :]
            data_G = median_G[:, index]

            orig_index = data["orig_labels"].index(l)
            if "selected_indexes" in data and "F_selected" in data: # bamf anchor
                orig_F = np.array(data["F_orig"].drop(data["F_orig"].columns[data["selected_indexes"][0]],axis=1))[orig_index, :]
            elif "selected_indexes" in data and "F_selected" not in data: # subset
                orig_F = np.array(data["F_orig"].drop(data["F_orig"].columns[data["selected_indexes"][0]],axis=1))[orig_index, :]
                data_F = np.delete(data_F,data["selected_indexes"][0])
            else:
                orig_F = np.array(data["F_orig"])[orig_index, :]

            orig_G = np.array(data["G_orig"])[:, orig_index]

            # calculate G metrics
            mean_label = f"Mean $G/G_o$ {l}"
            median_label = f"Median $G-G_o$ {l}"
            maximum_label = f"Maximum $G-G_o$ {l}"
            spear_label = f"G $ρ_{{{l}}}$"
            pearson_label = f"G $r_{{{l}}}$"

            # Add pearson
            numbers[spear_label] = spear(data_G, orig_G)
            numbers[pearson_label], _ = pearsonr(data_G, orig_G)

            div = np.mean(data_G) / np.mean(orig_G)
            diff = np.abs(data_G - orig_G)
            # numbers[mean_label] = div
            numbers[mean_label] = np.mean(diff)
            numbers[median_label] = np.nanmedian(diff)
            numbers[maximum_label] = np.nanmax(diff)

            mean_label = f"Mean $F-F_o$ {l}"
            median_label = f"Median $F-F_o$ {l}"
            maximum_label = f"Maximum $F-F_o$ {l}"
            spear_label = f"F $ρ_{{{l}}}$"
            pearson_label = f"F $r_{{{l}}}$"

            numbers[spear_label] = spear(data_F, orig_F)
            numbers[pearson_label], _ = pearsonr(data_F, orig_F)

            div = np.abs(data_F - orig_F)
            numbers[mean_label] = np.mean(div)
            numbers[median_label] = np.nanmedian(div)
            numbers[maximum_label] = np.nanmax(div)

    return numbers


def summarize_diagnostic(di, name):

    result = {}

    n = int(di.shape[0] / 2)

    raveled = di[n:, ...].ravel()

    result[f"mean {name}"] = np.mean(raveled)
    result[f"median {name}"] = np.median(raveled)
    result[f"std {name}"] = np.std(raveled)

    return result


def chain_diagnostics(data, ks=25):
    """Compare the distribution of last 25 and first 25 samples for each chain.

    Args:
        data (_type_): the results of two runs
        ks (int, optional): the sample size. Defaults to 25.

    Returns:
        _type_: final compared results, containing the similarity.
    """
    # chain_diagnotics: data samples 's result distribution to give us the modeled result.
    results = {}
    residuals = []
    samples = data["samples"]
  
    for n in range(data["chains"]):
        Z = np.stack(
            [
                (data["G"][i, n, :, :] @ data["F"][i, n, :, :])
                for i in (range(data["G"].shape[0]))
            ]
        )
        if "F_non_selected" in data or "F_selected" in data:
            residuals = Z - np.delete(data["X"], data["selected_indexes"][0],axis=1)
        else:
            residuals = Z - data["X"]
        samp1 = residuals[samples : samples + ks, ...].ravel()
        samp2 = residuals[-ks:, ...].ravel()

        # take last 25 and first 25 samples and compare distributions, if the result is notably
        # different at the end, then chain has wandered onto another minima
        _, p = ks_2samp(samp1, samp2)

        results[f"Chain {n} residual ks, p"] = p

    # check chain median divergence from max 500 last samples
    x = -500
    if samples < 500:
        x = samples

    if n > 1:
        # Q: why are the two the same
        s, results["log prob kruskall chains p"] = kruskal(
            *[data["lp__"][x:, n].ravel() for n in range(data["chains"])]
        )

        s, results["residuals kruskall chains p"] = kruskal(
            *[data["lp__"][x:, n].ravel() for n in range(data["chains"])]
        )

    return results


def sign_diagnostic(di, name):

    result = {}

    n = int(di.shape[0] / 2)

    samples = di[n:, ...]

    signs = np.sign(samples[1:, :] - samples[:-1, :]).ravel()
    plus_minus = [np.count_nonzero(signs > 0), np.count_nonzero(signs < 0)]

    result[f"mean sign {name}"] = np.mean(signs)
    result[f"sign test {name} p"] = binom_test(plus_minus)

    return result


default_model_labels = {
    "bamf": [
        "F",
        "G",
        "X",
        "m",
        "n",
        "p",
        "lp__",
        "sigma",
        "norm_const",
        "alpha_a",
        "alpha_b",
    ],
    "bamf_anchor": [
        "F",
        "F_non_selected",
        'selected_indexes',
        "G",
        "X",
        "m",
        "n",
        "p",
        "lp__",
        "sigma",
        "norm_const",
        "alpha_a",
        "alpha_b",
    ],
    "bamf_subset": [
        "F",
        'selected_indexes',
        "G",
        "X",
        "m",
        "n",
        "p",
        "lp__",
        "sigma",
        "norm_const",
        "alpha_a",
        "alpha_b",
    ],
}


def analyze_frame(
    dataset,
    benchmark=False,
    match_file=None,
    distance_metric="cityblock",
    cutoff=0.1,
    mass_thres = 0.1,
    model_data=None,
    get_tags=None,
    basepath=Path("./plots"),
    save_plot=True,
    sorting=True,
    add_mz_mode=True,
):

    metrics = {}

    if model_data is not None:
        model_labels = model_data
    else:
        model_labels = default_model_labels

    if get_tags is None:
        model = Path(dataset.iloc[0, :]["model"]).stem
        if model not in model_labels.keys():
            if "anchor" in model:
                tags = model_labels["bamf_anchor"]
            elif "subset" in model:
                tags = model_labels["bamf_subset"]
            elif "bamf" in model:
                tags = model_labels["bamf"]
            
            else:
                raise ValueError(
                    f"Tags to extract not specified and model: {model} unknown."
                )
            
        else:
            tags = model_labels[model]
    else:
        tags = get_tags

    action = dataset.iloc[0, :]["action"]
    if action == "map" and "lp__" in tags:
        t = tags.copy()
        t.remove("lp__")
        tags = t

    data = get_generic_data_from_multiple_files(dataset["filename"], labels=tags)
    result_file = dataset.iloc[0, :]["filename"]  # results location
    data["chains"] = len(dataset.filename)

    datafile = dataset.iloc[0, :]["datafile"]  # input data file location

    # merge the results and if needed, sorting it.
    if action == "sampling":
        # Choose the most popular order of the last samples

        data["samples"] = int(data["G"].shape[0] / 2)  # sample half of the ressults

        metrics.update(chain_diagnostics(data))

        merge_chains(data, n=data["samples"])  # extract last n draws from each chain

        indices_chains = np.nan

        if sorting:
            data, indices_chains = sort_by_Z_contribution(
                data,
                take=slice(-len(dataset["filename"]), None),
                metric=distance_metric,
            )

    elif action == "map":

        data["samples"] = min(data["chains"], 100)

        
        if mass_thres != 0:
            unresolved_ls = []
            for i in range(data["G"].shape[0]):
                if np.mean(data["G"][i].sum(axis=1)/data["X"].sum(axis=1))>1 + mass_thres or np.mean(data["G"][i].sum(axis=1)/data["X"].sum(axis=1))< 1-mass_thres:

                    unresolved_ls.append(i)
            print("unresolved iterations:",len(unresolved_ls),"index:",unresolved_ls,flush=True)
            data["F"] = np.delete(data["F"],unresolved_ls,axis=0)
            data["G"] = np.delete(data["G"],unresolved_ls,axis=0)
            autocorr = [i for i in data.keys() if i.startswith("alpha")]
            for item in autocorr:
                data[item] = np.delete(data[item],unresolved_ls,axis=0)
            data["samples"] = min(data["samples"],data["G"].shape[0])

        indices_chains = np.nan
        if sorting:
            data, indices_chains = sort_by_Z_contribution(
                # data, take=slice(0, data["samples"]), metric=distance_metric
                 data,take=slice(-data["samples"], None), metric=distance_metric
                
            )
    elif action == "vb":

        # data["samples"] =   # sample half of the ressults
        data["samples"] = min(int(data["G"].shape[0]), 1)

        merge_chains_vb(data, n=0)  # extract last n draws from each chain

        if mass_thres != 0:
            unresolved_ls = []
            for i in range(data["G"].shape[0]):
                if np.mean(data["G"][i].sum(axis=1)/data["X"].sum(axis=1))>1 + mass_thres or np.mean(data["G"][i].sum(axis=1)/data["X"].sum(axis=1))< 1-mass_thres:

                    unresolved_ls.append(i)
            print("unresolved iterations:",len(unresolved_ls),"index:",unresolved_ls,flush=True)
            data["F"] = np.delete(data["F"],unresolved_ls,axis=0)
            data["G"] = np.delete(data["G"],unresolved_ls,axis=0)
            autocorr = [i for i in data.keys() if i.startswith("alpha")]
            for item in autocorr:
                data[item] = np.delete(data[item],unresolved_ls,axis=0)
            data["samples"] = min(data["samples"],data["G"].shape[0])

        indices_chains = np.nan

        if sorting:
            data, indices_chains = sort_by_Z_contribution(
                data,
                take=slice(-data["samples"], None),
                metric=distance_metric,
            )

    else:
        raise ValueError(f"Unknown action:{action} performed by model")


    if "selected_indexes" in data and "anchor" not in model:
        data["Z"] = np.stack(
        [data["G"][i, :, :] @ np.delete(data["F"],data["selected_indexes"][0],axis=2)[i, :, :] for i in (range(data["G"].shape[0]))]
        )
    else:
        data["Z"] = np.stack(
            [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
        )

    # Take metadata from result file (should match across files)
    with h5py.File(result_file, "r") as infile:
        time_str = infile["time_index"][()].astype("str")
        data["timestep"] = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S")
        try:
            data["variable"] = infile["columns"][()].astype("int")
        except ValueError:
            data["variable"] = infile["columns"][()].astype("float").astype("int")

    # read the G & F benchmark, sort the modeled results using benchmark if available.

    if match_file is not None:
        # read G & F, Assuming these are normalized correctly and the right size.
        if match_file == "datafile":
            orig_file = datafile
        else:
            orig_file = match_file

        data["F_orig"] = pd.read_hdf(orig_file, "F")
        data["G_orig"] = pd.read_hdf(orig_file, "G")

        data = sort_label_by_truth(orig_file, data, cutoff, distance_metric)

    else:
        data["labels"] = np.arange(data["p"]).astype(str)

    outfilename = get_plot_filename(basepath, result_file, sorting)
    outfilename.parent.mkdir(parents=True, exist_ok=True)
    sorted_filename = outfilename.with_suffix(".nc4")


    # Calculate metrics
    summary = summarize_frame(data, benchmark=benchmark)
    metrics.update(summary)
    if "lp__" in data:
        metrics.update(sign_diagnostic(data["lp__"], "log prob"))
    print("metrics calculation finished",ctime(),flush=True)


    # Save intermediary file containing sorted data
    if "selected_indexes" in data and "F_selected" not in data: #bamf_subset model
            
        data_xarray = xr.Dataset(
            {
                "F": (["samples", "labels", "variable"], data["F"]),
                "G": (["samples", "timestep", "labels"], data["G"]),
                "X": (["timestep", "variable"], data["X"]),
                "error": (["timestep", "variable"], data["sigma"]),
                "alpha_a": (["samples", "labels"], data["alpha_a"]),
                "alpha_b": (["samples", "labels"], data["alpha_b"]),
            },
            coords={
                "samples": np.arange(data["F"].shape[0]),
                "labels": data["labels"],
                "selected_variable":np.delete(data["variable"], data["selected_indexes"][0]),
                "variable": data["variable"],
                "timestep": data["timestep"],
            },
        )
    elif "selected_indexes" in data and "F_selected" in data: # bamf anchor model
        data_xarray = xr.Dataset(
            {
                "F": (["samples", "labels", "selected_variable"], data["F"]),
                "G": (["samples", "timestep", "labels"], data["G"]),
                "X": (["timestep", "variable"], data["X"]),
                "error": (["timestep", "variable"], data["sigma"]),
                "alpha_a": (["samples", "labels"], data["alpha_a"]),
                "alpha_b": (["samples", "labels"], data["alpha_b"]),
            },
            coords={
                "samples": np.arange(data["F"].shape[0]),
                "labels": data["labels"],
                "selected_variable":np.delete(data["variable"], data["selected_indexes"][0]),
                "variable": data["variable"],
                "timestep": data["timestep"],
            },
        )
    else:
        data_xarray = xr.Dataset(
            {
                "F": (["samples", "labels", "variable"], data["F"]),
                "G": (["samples", "timestep", "labels"], data["G"]),
                "X": (["timestep", "variable"], data["X"]),
                "error": (["timestep", "variable"], data["sigma"]),
                "alpha_a": (["samples", "labels"], data["alpha_a"]),
                "alpha_b": (["samples", "labels"], data["alpha_b"]),
            },
            coords={
                "samples": np.arange(data["F"].shape[0]),
                "labels": data["labels"],
                "variable": data["variable"],
                "timestep": data["timestep"],
            },
        )
    if "lp__" in data:
        data_xarray["lp__"] = xr.DataArray(
            data["lp__"],
            coords={
                "total": np.arange(data["lp__"].shape[0]),
                "chains": np.arange(data["lp__"].shape[1]),
            },
        )

    if benchmark and match_file is not None:
        orig_labels = data["F_orig"].index #original command
        print(orig_labels)
        print()
        data_xarray["G_orig"] = xr.DataArray(
            data["G_orig"],
            coords={
                "timestep": data["timestep"],
                "orig_labels": orig_labels, 
            },
        )
    
        data_xarray["F_orig"] = xr.DataArray(
            data["F_orig"],
            coords={
                "orig_labels": orig_labels,
                "variable": data["variable"],
            },
        )

    if  add_mz_mode:
        data_xarray = add_mz(data_xarray)
    
    if save_plot:
        data_xarray.to_netcdf(sorted_filename)
        print("Plotting starts...")
        if sorting == False:
            sorted = "_unsorted"
        else:
            sorted = ""

        with Collated_output(outfilename) as outfile:
            print("start plotting...", ctime(), flush=True)
            if "lp__" in data:
                f, p = plotting.plot_diagnostic(data["lp__"], "log prob")
                outfile.savefig(f, "log prob" + sorted)
                print("log prob saved", flush=True)
                plt.close()

                metrics["log prob p"] = p
                lp_m = summarize_diagnostic(data["lp__"], "log prob")
                metrics.update(lp_m)

            f = plotting.mass_conservation(data_xarray, sample="median")
            print("conservation from median G saved", flush=True)
            outfile.savefig(f, "conservation from median G" + sorted)
            plt.close()

            f = plotting.mass_conservation(data_xarray, sample=-1)
            print("conservation from last sample saved", flush=True)
            outfile.savefig(f, "conservation from last sample" + sorted)
            plt.close()

            # plot standard G,F plots, with truth if benchmark

            f,F_metrics = plotting.plot_F(data_xarray, benchmark=benchmark)
            print("F saved", flush=True)
            outfile.savefig(f, "F" + sorted)
            outfile.savetxt(F_metrics, "F" + sorted)
            plt.close()

            f,G_metrics = plotting.plot_G(data_xarray, benchmark=benchmark)
            print("G saved", flush=True)
            outfile.savefig(f, "G" + sorted)
            outfile.savetxt(G_metrics, "G" + sorted)
            plt.close()

            f,median_G_metrics = plotting.plot_median_G(data_xarray, benchmark=benchmark)
            print("median G saved", flush=True)
            outfile.savefig(f, "median_G" + sorted)
            outfile.savetxt(median_G_metrics, "median_G" + sorted)
            plt.close()

            if action == "map":
                for i in range(len(data_xarray["labels"])):
                    f = plotting.plot_individual_F(
                        data_xarray, i, benchmark=benchmark, sample_size=min(7, len(data_xarray["samples"]))
                    )
                    print(f"F_map_{i} " + "saved", flush=True)
                    outfile.savefig(f, f"F_map_{i}" + sorted)
                    plt.close()

                f = plotting.plot_individual_G(data_xarray, benchmark=benchmark, sample_size=min(7, len(data_xarray["samples"])))
                print("G_map saved", flush=True)
                outfile.savefig(f, "G_map" + sorted)
                plt.close()

                pass

    return data_xarray, metrics, indices_chains


def add_mz(data):
    print("adding few mzs...")
    mz_vars = data
    try:
        mz_vars["alpha_a"] = data["alpha_a"]
        mz_vars["alpha_b"] = data["alpha_b"]
    except:
        pass
    data["variable"] = data.variable.astype(int)

    mz44 = data["F"].loc[:, :, 44]
    index = [16, 17, 18, 28]
    add_F = np.zeros(shape=(len(data.samples), len(data.labels), len(index)))
    add_F = xr.DataArray(
        add_F,
        coords={"samples": data.samples, "labels": data.labels, "variable": index},
        name="F",
    )
    add_F.loc[:, :, 16] = 0.225 * 0.04 * mz44
    add_F.loc[:, :, 17] = 0.225 * 0.25 * mz44
    add_F.loc[:, :, 18] = 0.225 * mz44
    add_F.loc[:, :, 28] = mz44

    F_combined = xr.concat((data.F, add_F), dim="variable").sortby("variable")

    # renormalize F and G
    norm_F = F_combined / F_combined.sum(dim="variable")
    norm_G = data.G * F_combined.sum(dim="variable")

    mz_vars["F"] = norm_F
    mz_vars["G"] = norm_G

    add_data = np.zeros(shape=(len(data.timestep), len(index)))
    add_data = xr.DataArray(
        add_data, coords={"timestep": data.timestep, "variable": index}, name="X"
    )
    mz44 = data["X"].loc[:, 44]
    add_data.loc[:, 16] = 0.225 * 0.04 * mz44
    add_data.loc[:, 17] = 0.225 * 0.25 * mz44
    add_data.loc[:, 18] = 0.225 * mz44
    add_data.loc[:, 28] = mz44

    mz_vars["X"] = xr.concat((data.X, add_data), dim="variable").sortby("variable")

    add_error = np.zeros(shape=(len(data.timestep), len(index)))
    add_error = xr.DataArray(
        add_error, coords={"timestep": data.timestep, "variable": index}, name="sigma"
    )
    mz44 = data["error"].loc[:, 44]
    add_error.loc[:, 16] = 0.225 * 0.04 * mz44
    add_error.loc[:, 17] = 0.225 * 0.25 * mz44
    add_error.loc[:, 18] = 0.225 * mz44
    add_error.loc[:, 28] = mz44

    mz_vars["error"] = xr.concat((data.error, add_error), dim="variable").sortby(
        "variable"
    )
    # mz_vars["error"] = t.error.combine_first(add_error)

    if "F_orig" in data:

        add_F = np.zeros(shape=(len(data.labels), len(index)))
        add_F = xr.DataArray(
            add_F,
            coords={"orig_labels": data.labels.values, "variable": index},
            name="F_orig",
        )
        mz44 = data["F_orig"].loc[:, 44]
        add_F.loc[:, 16] = 0.225 * 0.04 * mz44
        add_F.loc[:, 17] = 0.225 * 0.25 * mz44
        add_F.loc[:, 18] = 0.225 * mz44
        add_F.loc[:, 28] = mz44
        mz_vars["F_orig"] = data.F_orig.combine_first(add_F)

        # then also G_orig is and must be renormalized
        component_sums = mz_vars["F_orig"].sum(dim="variable")
        mz_vars["F_orig"] = mz_vars["F_orig"] / component_sums
        mz_vars["G_orig"] = data.G_orig * component_sums

    # F_reference if exists
    if "F_reference" in data:
        add_F = np.zeros(shape=(len(data.reference_component), len(index)))
        add_F = xr.DataArray(
            add_F,
            coords={"reference_component": data.reference_component, "variable": index},
            name="F_reference",
        )
        mz44 = data["F_reference"].loc[:, 44]
        add_F.loc[:, 16] = 0.225 * 0.04 * mz44
        add_F.loc[:, 17] = 0.225 * 0.25 * mz44
        add_F.loc[:, 18] = 0.225 * mz44
        add_F.loc[:, 28] = mz44

        mz_vars["F_reference"] = data.F_reference.combine_first(add_F)

        component_sums = mz_vars["F_reference"].sum(dim="variable")
        mz_vars["F_reference"] = mz_vars["F_reference"] / component_sums

    # mz_vars = {key: item.rename(key) for key, item in mz_vars.items()}

    # mz_vars = xr.merge(mz_vars.values())

    return mz_vars



def extract_pmf_data_in_normal_form(
    filename,
    metadatafile,
    components,
    dataname="Mx_data",
    errorname="Mx_err",
    prname="amus",
    tsname="time",
):

    # Unfortunately to get metadata such as times + columns you need ancillary data

    result = {}

    with h5py.File(filename, "r") as f:
        runs = [key for key in f.keys() if key.startswith("run_")]

        result["F"] = np.stack(
            [
                f[run]["factor_pr"][()].T
                for run in runs
                if (
                    f[run]["factor_pr"][()].T.shape[0] == components
                    and ~np.isnan(f[run]["factor_pr"][()]).any()
                )
            ]
        )

        result["G"] = np.stack(
            [
                f[run]["factor_ts"][()]
                for run in runs
                if (
                    f[run]["factor_ts"][()].shape[1] == components
                    and ~np.isnan(f[run]["factor_ts"][()]).any()
                )
            ]
        )

        # result["columns"] = f["PMF_input"]["amus"][()]
        # add slicing according to the results shape
        # result["X"] = f["PMF_input"][dataname][()][
        #     : result["G"][0].shape[0], : result["F"][0].shape[1]
        # ]
        # result["sigma"] = f["PMF_input"][errorname][()][
        #     : result["G"][0].shape[0], : result["F"][0].shape[1]
        # ]
        # time_index = f["PMF_input"][tsname][()].astype(int)
        # profile_index = f["PMF_input"][prname][()].astype(int)
        try:
            result["X"] = f["PMF_input"][dataname][()][
                np.array(f["PMF_input"]["tseries_index"]).astype(int)
            ][:, np.array(f["PMF_input"]["profile_index"]).astype(int)]
            result["sigma"] = f["PMF_input"][errorname][()][
                np.array(f["PMF_input"]["tseries_index"]).astype(int)
            ][:, np.array(f["PMF_input"]["profile_index"]).astype(int)]
        except:
            print(f["PMF_input"].keys())
        if metadatafile != None:
            meta = pd.read_hdf(metadatafile, "data")
            result["timestep"] = meta.index.values
            result["variable"] = meta.columns.values
        else:
            result["timestep"] = [
                itx_helper.igor_time_convert(t)
                for t in f["PMF_input"][tsname][()][
                    [f["PMF_input"]["tseries_index"][()].astype(int)]
                ]
            ]

            result["variable"] = f["PMF_input"][prname][()][
                [f["PMF_input"]["profile_index"][()].astype(int)]
            ]
    result["n"] = result["G"].shape[1]
    result["m"] = result["F"].shape[2]
    result["p"] = components

    return result


def summarize_frame_from_xarray(data, benchmark=False):

    numbers = {}

    Z = reconstruct(data["F"], data["G"])

    residuals = data["X"].to_numpy()[None, :, :] - Z
    respererr = residuals / data["error"].to_numpy()

    numbers["Mean(abs(residual))"] = np.mean(np.abs(residuals))
    numbers["Median(abs(residual))"] = np.median(np.abs(residuals))
    numbers["Max(abs(residual))"] = np.max(np.abs(residuals))

    numbers["Mean(abs(residual/error))"] = np.mean(np.abs(respererr))
    numbers["Median(abs(residual/error))"] = np.median(np.abs(respererr))
    numbers["Max(abs(residual/error))"] = np.max(np.abs(respererr))

    d_means = Z.mean(axis=(0, 1))
    d_stds = Z.std(axis=(0, 1))

    z_data = (data["X"].to_numpy() - d_means) / d_stds

    numbers["Z scaled mean"] = z_data.mean()
    numbers["Z scaled std"] = z_data.std()

    numbers["Qm"] = np.sum(z_data**2)
    numbers["Qexp"] = np.prod(data["X"].shape)

    del Z

    alphas = [i for i in data.keys() if i.startswith("alpha")]

    for item in alphas:

        print(f"{item} mean")
        means = np.mean(data[item].to_numpy(), axis=0)
        print(f"{item} median")
        medians = np.median(data[item].to_numpy(), axis=0)
        print(f"{item} std")
        stds = np.std(data[item].to_numpy(), axis=0)

        for i in range(means.shape[0]):
            name = data["labels"][i].item()
            numbers[f"{item} {name} mean"] = means[i]
            numbers[f"{item} {name} median"] = medians[i]
            numbers[f"{item} {name} std"] = stds[i]

    if benchmark:
        # Only for comparing labels to external reference

        median_F = data["F"].median(dim="samples")
        median_G = data["G"].median(dim="samples")

        ids = [
            (i, l.item())
            for i, l in enumerate(data["labels"])
            if l.item() in data["orig_labels"]
        ]

        print("Identified components:", ids)

        numbers["identified components"] = len(ids)

        for index, l in ids:

            print(f"{l} statistics")

            data_F = median_F.loc[l, :]
            data_G = median_G.loc[:, l]

            orig_F = data["F_orig"].loc[l, :]
            orig_G = data["G_orig"].loc[:, l]

            mean_label = f"Mean $G/G_o$ {l}"
            median_label = f"Median $G-G_o$ {l}"
            maximum_label = f"Maximum $G-G_o$ {l}"
            spear_label = f"G $ρ_{{{l}}}$"
            pearson_label = f"G $r_{{{l}}}$"

            # Add pearson
            numbers[spear_label] = spear(data_G, orig_G)
            numbers[pearson_label], _ = pearsonr(data_G, orig_G)

            div = np.mean(data_G) / np.mean(orig_G)
            diff = np.abs(data_G - orig_G)

            numbers[mean_label] = div.item()
            numbers[median_label] = np.nanmedian(diff).item()
            numbers[maximum_label] = np.nanmax(diff).item()

            mean_label = f"Mean $F-F_o$ {l}"
            median_label = f"Median $F-F_o$ {l}"
            maximum_label = f"Maximum $F-F_o$ {l}"
            spear_label = f"F $ρ_{{{l}}}$"
            pearson_label = f"F $r_{{{l}}}$"

            numbers[spear_label] = spear(data_F, orig_F)
            numbers[pearson_label], _ = pearsonr(data_F, orig_F)

            div = np.abs(data_F - orig_F)

            numbers[mean_label] = np.mean(div).item()
            numbers[median_label] = np.median(div).item()
            numbers[maximum_label] = np.max(div).item()

    return numbers


def analyze_frame_for_pmf(
    datafile,
    match_file,
    components,
    title,
    dataname="Mx_data",
    errorname="Mx_err",
    prname="amus",
    tsname="time",
    benchmark=False,
    distance_metric="cityblock",
    cutoff=0.1,
    basepath=Path("./plots"),
    add_mz_mode=False,
    force_normalization=False,
    sorting=True,
):
    metrics = {}

    data = extract_pmf_data_in_normal_form(
        datafile, match_file, components, dataname, errorname, prname, tsname
    )

    if force_normalization:
        F_sum = data["F"].sum(axis=2)
        data["F"] = data["F"] / F_sum[:, :, None]
        data["G"] = data["G"] * F_sum[:, None, :]

    # Choose the most popular order of the last samples
    if sorting:
        slicy = slice(-len(data["F"]), None)

        data, indices = sort_by_Z_contribution(data, take=slicy, metric=distance_metric)

    data["Z"] = np.stack(
        [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
    )

    if benchmark:
        data["F_orig"] = pd.read_hdf(match_file, "F").values
        data["G_orig"] = pd.read_hdf(match_file, "G").values
        data["X"] = pd.read_hdf(match_file, "data").values
        data["sigma"] = pd.read_hdf(match_file, "error").values

    # Sort and label with reference F if available.
    if match_file is not None and benchmark:

        try:
            match_profiles = pd.read_hdf(match_file, "F")
            data["orig_labels"] = list(match_profiles.index.values)
        except:
            match_profiles = pd.read_hdf(match_file, "G")
            data["orig_labels"] = list(match_profiles.columns.values)

        median_F = np.median(data["F"], axis=0)

        indices, labels = match_labels(
            median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric
        )

        data["F"] = data["F"][:, indices, :]
        data["G"] = data["G"][:, :, indices]

        data["labels"] = labels

        # data["reference_F"] = match_profiles

    else:
        data["labels"] = np.arange(data["p"]).astype(str)

    data_xarray = xr.Dataset(
        {
            "F": (["samples", "labels", "variable"], data["F"]),
            "G": (["samples", "timestep", "labels"], data["G"]),
            "X": (["timestep", "variable"], data["X"]),
            "error": (["timestep", "variable"], data["sigma"]),
        },
        coords={
            "samples": np.arange(data["F"].shape[0]),
            "labels": data["labels"],
            "variable": data["variable"],
            "timestep": data["timestep"],
        },
    )
    if benchmark and match_file is not None:
        orig_labels = match_profiles.index
        data_xarray["G_orig"] = xr.DataArray(
            data["G_orig"],
            coords={
                "timestep": data["timestep"],
                "orig_labels": orig_labels,
            },
        )
        data_xarray["F_orig"] = xr.DataArray(
            data["F_orig"],
            coords={
                "orig_labels": orig_labels,
                "variable": data["variable"],
            },
        )
        data_xarray["p"] = components
    if add_mz_mode:
        data_xarray = add_mz(data_xarray)
    summary = summarize_frame_from_xarray(data_xarray, benchmark=benchmark)

    return summary, data_xarray


def analyze_frame_wo_plot(
    dataset,
    benchmark=False,
    match_file=None,
    distance_metric="cityblock",
    cutoff=0.1,
    model_data=None,
    get_tags=None,
    basepath=Path("./plots"),
    save_sorted=None,
    add_mz_mode=False,
    sorting=True,
):

    metrics = {}

    if model_data is not None:
        model_labels = model_data
    else:
        model_labels = default_model_labels

    if get_tags is None:
        model = Path(dataset.iloc[0, :]["model"]).stem
        if model not in model_labels.keys():
            raise ValueError(
                f"Tags to extract not specified and model: {model} unknown."
            )
        else:
            tags = model_labels[model]
    else:
        tags = get_tags

    action = dataset.iloc[0, :]["action"]

    first_ids = dataset.iloc[0, :]

    if action == "map" and "lp__" in tags:
        t = tags.copy()
        t.remove("lp__")
        tags = t

    data = get_generic_data_from_multiple_files(dataset["filename"], labels=tags)
    first_chain = dataset.iloc[0, :]["filename"]  # results location
    datafile = dataset.iloc[0, :]["datafile"]  # input data file location

    data["chains"] = len(dataset.filename)

    # Choose the most popular order of the last samples
    if action == "sampling":

        data["samples"] = int(data["G"].shape[0] / 2)  # sample half of the results

        metrics.update(chain_diagnostics(data))

        merge_chains(data, n=data["samples"])  # extract last n draws from each chain

        slicy = slice(-len(dataset["filename"]), None)

        if sorting:
            data, indices = sort_by_Z_contribution(
                data, take=slicy, metric=distance_metric
            )

        data["Z"] = np.stack(
            [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
        )

    elif action == "map":

        data["samples"] = min(data["chains"], 1000)
        if sorting:
            data, indices = sort_by_Z_contribution(
                data, take=slice(0, data["samples"]), metric=distance_metric
            )

        data["Z"] = np.stack(
            [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
        )

    else:
        raise ValueError(f"Unknown action:{action} performed by model")

    # Take metadata (timelist and m/z numbers) from first file (should match across files)
    with h5py.File(first_chain, "r") as infile:
        time_str = infile["time_index"][()].astype("str")
        data["timestep"] = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S")
        data["variable"] = infile["columns"][()].astype("str")

    # Sort and label with reference F if available.
    if match_file is not None:
        if match_file == "datafile":
            orig_file = datafile

        else:
            orig_file = match_file
        data["F_orig"] = np.array(pd.read_hdf(orig_file, "F"))
        data["G_orig"] = np.array(pd.read_hdf(orig_file, "G"))

        match_profiles = pd.read_hdf(orig_file, "F")
        data["orig_labels"] = list(match_profiles.index.values)
        # sorting the benchmark and match the labels
        if action == "sampling" or action == "map":

            # sort the component index of extracted draws from each chain match to the benchmark based on F matrix
            median_F = np.median(data["F"], axis=0)

            indices, labels = match_labels(
                median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric
            )

            all_indices = set(np.arange(data["F"].shape[1]))
            # unmatched component index
            unresolved = list(set(all_indices) - set(indices))

            indices = [*indices, *unresolved]

            data["F"] = data["F"][:, indices, :]
            data["G"] = data["G"][:, :, indices]

            autocorr = [i for i in data.keys() if i.startswith("alpha")]

            for item in autocorr:
                data[item] = data[item][:, indices]

            data["labels"] = labels  # component name

            # assign questionmark to unmatched component result
            if len(unresolved) > 0:
                for kk in range(len(unresolved)):
                    data["labels"] = np.append(data["labels"], "?" + str(kk))

            data["reference_F"] = match_profiles

    else:
        data["labels"] = np.arange(data["p"]).astype(str)

    outfilename = get_plot_filename(basepath, first_chain, sorting)
    outfilename.parent.mkdir(parents=True, exist_ok=True)

    sorted_filename = outfilename.with_suffix(".nc4")

    # Calculate metrics

    summary = summarize_frame(data, benchmark=benchmark)

    metrics.update(summary)

    if "lp__" in data:
        metrics.update(sign_diagnostic(data["lp__"], "log prob"))

    # Save intermediary file containing sorted data
    data_xarray = xr.Dataset(
        {
            "F": (["samples", "labels", "variable"], data["F"]),
            "G": (["samples", "timestep", "labels"], data["G"]),
            "X": (["timestep", "variable"], data["X"]),
            "error": (["timestep", "variable"], data["sigma"]),
            "alpha_a": (["samples", "labels"], data["alpha_a"]),
            "alpha_b": (["samples", "labels"], data["alpha_b"]),
            "G_orig": (["timestep", "orig_labels"], data["G_orig"]),
            "F_orig": (["orig_labels", "variable"], data["F_orig"]),
        },
        coords={
            "samples": np.arange(data["F"].shape[0]),
            "labels": data["labels"],
            "variable": data["variable"],
            "timestep": data["timestep"],
            "orig_labels": data["orig_labels"],
        },
    )

    if add_mz_mode:

        data_xarray = add_mz(data_xarray)

    data_xarray.to_netcdf(sorted_filename)

    return outfilename, data_xarray, metrics


# ------------------------- old ------------------------
def analyze_frame_wo_plot_old(
    dataset,
    benchmark=False,
    match_file=None,
    distance_metric="cityblock",
    cutoff=0.1,
    model_data=None,
    get_tags=None,
    basepath=Path("./plots"),
    save_sorted=None,
    add_mz_mode=False,
):

    metrics = {}

    if model_data is not None:
        model_labels = model_data
    else:
        model_labels = default_model_labels

    if get_tags is None:
        model = Path(dataset.iloc[0, :]["model"]).stem
        if model not in model_labels.keys():
            raise ValueError(
                f"Tags to extract not specified and model: {model} unknown."
            )
        else:
            tags = model_labels[model]
    else:
        tags = get_tags

    action = dataset.iloc[0, :]["action"]

    first_ids = dataset.iloc[0, :]

    if action == "map" and "lp__" in tags:
        t = tags.copy()
        t.remove("lp__")
        tags = t

    data = get_generic_data_from_multiple_files(dataset["filename"], labels=tags)
    first_chain = dataset.iloc[0, :]["filename"]  # results location
    datafile = dataset.iloc[0, :]["datafile"]  # input data file location
    data["chains"] = len(dataset.filename)

    # Choose the most popular order of the last samples
    if action == "sampling":

        data["samples"] = int(data["G"].shape[0] / 2)  # sample half of the results

        metrics.update(chain_diagnostics(data))

        merge_chains(data, n=data["samples"])  # extract last n draws from each chain

        slicy = slice(-len(dataset["filename"]), None)

        data, indices = sort_by_Z_contribution(data, take=slicy, metric=distance_metric)

        data["Z"] = np.stack(
            [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
        )

    elif action == "map":

        data["samples"] = min(data["chains"], 1000)

        data, indices = sort_by_Z_contribution(
            data, take=slice(0, data["samples"]), metric=distance_metric
        )

        data["Z"] = np.stack(
            [data["G"][i, :, :] @ data["F"][i, :, :] for i in range(data["G"].shape[0])]
        )

    else:
        raise ValueError(f"Unknown action:{action} performed by model")

    # Take metadata (timelist and m/z numbers) from first file (should match across files)
    with h5py.File(first_chain, "r") as infile:
        time_str = infile["time_index"][()].astype("str")
        data["timestep"] = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S")
        data["variable"] = infile["columns"][()].astype("str")

    # TODO split benchmark and match file
    if benchmark:
        first_chain = dataset.iloc[0, :]["filename"]
        # Assuming these are normalized correctly and the right size.
        with h5py.File(first_chain, "r") as infile:
            data["F_orig"] = infile["F_orig"][()]
            data["G_orig"] = infile["G_orig"][()]

    # Sort and label with reference F if available.
    if match_file is not None:

        if match_file == "datafile":
            match_profiles = pd.read_hdf(datafile, "F")
        else:
            match_profiles = pd.read_hdf(match_file, "F")

        if benchmark:
            data["orig_labels"] = list(match_profiles.index.values)

        if action == "sampling" or action == "map":

            # sort the component index of extracted draws from each chain match to the benchmark based on F matrix
            median_F = np.median(data["F"], axis=0)

            indices, labels = match_labels(
                median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric
            )

            all_indices = set(np.arange(data["F"].shape[1]))
            # unmatched component index
            unresolved = list(set(all_indices) - set(indices))

            indices = [*indices, *unresolved]

            data["F"] = data["F"][:, indices, :]
            data["G"] = data["G"][:, :, indices]

            autocorr = [i for i in data.keys() if i.startswith("alpha")]

            for item in autocorr:
                data[item] = data[item][:, indices]

            data["labels"] = labels  # component name

            # assign questionmark to unmatched component result
            if len(unresolved) > 0:
                for kk in range(len(unresolved)):
                    data["labels"] = np.append(data["labels"], "?" + str(kk))

            data["reference_F"] = match_profiles

    else:
        data["labels"] = np.arange(data["p"]).astype(str)

    outfilename = get_plot_filename(basepath, first_chain)
    outfilename.parent.mkdir(parents=True, exist_ok=True)

    sorted_filename = outfilename.with_suffix(".nc4")
    print(outfilename, flush=True)

    # Calculate metrics

    summary = summarize_frame(data, benchmark=benchmark)

    metrics.update(summary)

    if "lp__" in data:
        metrics.update(sign_diagnostic(data["lp__"], "log prob"))

    # Save intermediary file containing sorted data
    data_xarray = xr.Dataset(
        {
            "F": (["samples", "labels", "variable"], data["F"]),
            "G": (["samples", "timestep", "labels"], data["G"]),
            "X": (["timestep", "variable"], data["X"]),
            "error": (["timestep", "variable"], data["sigma"]),
            "alpha_a": (["samples", "labels"], data["alpha_a"]),
            "alpha_b": (["samples", "labels"], data["alpha_b"]),
        },
        coords={
            "samples": np.arange(data["F"].shape[0]),
            "labels": data["labels"],
            "variable": data["variable"],
            "timestep": data["timestep"],
        },
    )

    if benchmark and match_file is not None:
        orig_labels = match_profiles.index
        data_xarray["G_orig"] = xr.DataArray(
            data["G_orig"],
            coords={
                "timestep": data["timestep"],
                "orig_labels": orig_labels,
            },
        )
        data_xarray["F_orig"] = xr.DataArray(
            data["F_orig"],
            coords={
                "orig_labels": orig_labels,
                "variable": data["variable"],
            },
        )
    if add_mz_mode:

        data_xarray = add_mz(data_xarray)

    #     data_xarray.to_netcdf(sorted_filename)

    return outfilename, data_xarray, metrics
