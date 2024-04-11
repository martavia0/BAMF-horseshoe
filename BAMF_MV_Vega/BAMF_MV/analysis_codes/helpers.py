import io
import json
import re
import zipfile
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm


class Collated_output:
    def __init__(self, filename, save_types=["pdf"], compression=zipfile.lzma):

        self.output = zipfile.ZipFile(filename, "w")
        self.save_types = save_types

    def __enter__(self):

        return self

    def savefig(self, figure, name):

        for suffix in self.save_types:
            out = io.BytesIO()
            savename = f"{suffix}/{name}.{suffix}"
            figure.savefig(out, format=suffix, bbox_inches="tight")
            self.output.writestr(savename, out.getvalue())

    def savetxt(self, lines, name):
        suffix = "txt"
        savename = f"{suffix}/{name}.{suffix}"
        Path(savename).parent.mkdir(parents=True, exist_ok=True)

        # csv_lines = lines.to_csv()
        # self.output.writestr(savename, csv_lines)


        s = io.StringIO()
        lines.to_csv(s, index=True, header=True)
        self.output.writestr(savename, s.getvalue())

        
    def __exit__(self, type=None, value=None, traceback=None):

        self.output.close()


def get_plot_filename(
    basepath, filename, sorting=True, removal=re.compile("_seed-[0-9]+")
):
    # Removes the seed (since that is merged by analysis code) and provides a zipfilename to save plots
    removals = removal.findall(filename)

    endfilename = filename

    for item in removals:
        endfilename = endfilename.replace(item, "")

    if sorting == False:
        print("unsorted")
        endfilename = Path(
            Path(endfilename).with_suffix("").name + str("_unsorted")
        ).with_suffix(".zip")
    else:
        endfilename = Path(endfilename).with_suffix(".zip").name


    try:
        return basepath / endfilename
    except TypeError:
        return Path(basepath) / endfilename


def find_lists(x):

    iterables = []

    for key, item in x.items():
        if type(item) is list:
            iterables.append(key)

    return iterables


def generate_options(filepath, result_output):
    """create all the experiments using the input file

    Args:
        filepath (_type_): The json file containing each repeated experiment and
                            their parameters.

    Yields:
        _type_: single experiment with assigned options
    """
    # There is probably a smarter way to do this, but this has already taken
    # long enough.

    # options needed remove suffix while used for naming the output folder
    possible_paths = {
        "datafile",
        "model",
        "restriction",
        "outputbase",
    }
    unhashable = {"initoptions"}
    forbidden = possible_paths | unhashable

    # load the json file with all running options
    with filepath.open() as infile:
        opts = json.load(infile)
    experiments = list(opts.keys())
    experiments.remove("general")
    general = opts["general"]  # Options in general dictionary pertaining to all

    # create folder name for each repeated experiment
    # output_base = Path(general["outputbase"])
    output_base = Path(result_output)
    for ex in experiments:
        output = output_base / ex

        # merge the options in current experiment with options in general
        ex_opts = general.copy()
        ex_opts.update(opts[ex])

        # find all the options key having value with list format
        lists = find_lists(ex_opts)
        # Remove extract
        if "extract" in lists:
            lists.remove("extract")  # since no need & list format

        # iterate through all the list element
        if not lists:
            yield ex_opts  # return if no iterations exists
        else:
            iterables = []
            for item in lists:
                iterables.append(ex_opts[item])

            for values in product(*iterables):
                specifics = dict(zip(lists, values))
                spec_name = "_".join(
                    [
                        str(key) + "-" + str(item)
                        for key, item in specifics.items()
                        if key not in forbidden
                    ]
                )
                spec_name = ex + "_" + spec_name

                # TODO? if you name all your different datafiles with the same name, you are going to have a bad time
                for item in possible_paths:
                    if item in specifics.keys():
                        spec_name = (
                            spec_name
                            + "_"
                            + str(item)
                            + "-"
                            + Path(specifics[item]).stem
                        )

                # Maybe adding everything to filename can only go so far.
                # Some other naming scheme could work with metadata inside the files
                # This does have human readability, but

                # Q: how many initoptions, difference between modelopts and initoptions
                if "modelopts" in specifics.keys():
                    spec_name = (
                        spec_name
                        + "_"
                        + "initoptions"
                        + "-"
                        + str(ex_opts["initoptions"].index(specifics["initoptions"]))
                    )

                spec_name = spec_name + ".h5"
                specifics["filename"] = output / spec_name

                # adding the filename as a new option and remove the outputbase
                spec_opts = ex_opts.copy()
                spec_opts.update(specifics)
                if "outputbase" in spec_opts:
                    del spec_opts["outputbase"]
                yield spec_opts


def dump_hdf5(filename, **data):
    """save the results to the given file position.

    Args:
        filename (_type_): _description_
    """

    # for i, k in data.items():
    #    print(i, type(k))

    with h5py.File(filename, "w") as f:
        for name, d in data.items():
            #    print("Saving:", name)

            if type(d) is list:
                # So apparently hdf5 library doesn't like unicode.
                if type(d[0]) is str:
                    d = np.array(d, dtype="S")
                else:
                    d = np.array(d)

            # Can't have nice things
            if type(d) is dict:
                d = str(d)

            if isinstance(d, Path):
                d = str(d)

            f.create_dataset(name, data=d)


def maximize():
    plt.get_current_fig_manager().window.showMaximized()


def normalize(F, G):
    return F / np.sum(F, axis=-1, keepdims=True), G * np.sum(F, axis=-1)[:, None, :]


def normalize_chains(F, G, factor):
    return {
        "F": F / np.sum(F, axis=-1, keepdims=True),
        "G": factor * G * np.sum(F, axis=-1)[:, :, None, :],
    }


def labels(odict):
    return list(odict.keys())


def sort_by_batch_label(data, labels, axis=2):
    """sort the component sequence for each draw based on sorted index generated from
    function distance_label_batch_by_Z

    Args:
        data (_type_): input sampling draw's data matrix
        labels (_type_): sampling draw's sorted component index
        axis (int, optional): axis position where related to components. 3: G matrix, 2:F matrix, 1d parameters matrix;
         Defaults to 2.

    Raises:
        ValueError: _description_

    Returns:
        _type_: sorted sampling draws data matrix
    """
    # Isin't really the most efficient method ever, but ¯\_(ツ)_/¯
    # Im sure there is a nicer way to do this,
    # but all the obvious ones return memoryerror on 16 Gb machine

    if axis == 3:
        return np.stack([data[n, :, labels[n, :]] for n in range(data.shape[0])])
        # replacement:
        # return np.stack([data[n : n + 1, :, labels[n, :]] for n in range(data.shape[0])]).squeeze()
    elif axis == 2:
        return np.stack([data[n, labels[n, :], :] for n in range(data.shape[0])])
    elif axis == "1d":
        return np.stack([data[n, labels[n, :]] for n in range(data.shape[0])])

    raise ValueError("Axis must be 2, 3, or '1d'")


def select_hungarian(values, exemplars, distance_metric="correlation"):
    """this function solve the linear sum assignment problem.

    Args:
        values (_type_):  Z distribution of the draw needed to be sorted
        exemplars (_type_): example Z distribution
        distance_metric (str, optional):  The distance metric to use.. Defaults to "correlation".

    Returns:
        _type_: An array of input values component index based on exemplars component sequence
    """

    distances = cdist(exemplars, values, distance_metric)

    row_ind, selections = linear_sum_assignment(distances)

    # sorted columns indexs based on the sorted row number (exemplars component sequence) [0,1,2,..n]
    selections = selections[np.argsort(row_ind)]

    return selections


def calculate_Z_contribution(F, G):
    """calculate Z through G*F for each component

    Args:
        F (_type_): matrix
        G (_type_): matrix

    Returns:
        _type_: _description_
    """

    # components = F.shape[0]

    # Z = []

    # for c in range(components):
    #     Z.append(
    #         (((G[:, c][:, None]) @ (F[c, :][None, :])) / np.mean(G[:, c])).flatten()
    #     )
    # return np.stack(Z)
    G_norm = G / np.mean(G, axis=0)
    Z = (np.einsum("nk,km->knm", G_norm, F)).reshape(F.shape[0], -1)
    return Z


def distance_label_batch_by_Z(F, G, exemplars, metric="euclidean"):
    """calculate each draw's Z from G and F, compare it's distance to the last draw,
    then sorts the component sequence based on last draw's component sequence

    Args:
        F (_type_): extracted last sampling F matrix from every chain
        G (_type_): extracted last sampling G matrix from every chain
        exemplars (_type_): Z contribution of each chain's last draw
        metric (str, optional): distance matric. Defaults to "euclidean".

    Returns:
        _type_: component sequence for extracted draws
    """
    # compare the sampled data to last runs
    samples = F.shape[0]

    lists = [
        select_hungarian(
            calculate_Z_contribution(F[n, ...].squeeze(), G[n, ...].squeeze()),
            exemplars,
            metric,
        )
        # for n in tqdm(range(samples))
        for n in range(samples)
    ]

    return np.vstack(lists)


def distance_label_batch(F, exemplars, metric="correlation"):

    samples = F.shape[0]

    lists = [
        select_hungarian(F[n, :, :].squeeze(), exemplars, metric)
        for n in range(samples)
    ]

    return np.vstack(lists)


def distance_label_batch_chains(F, exemplars, metric="correlation"):

    samples = F.shape[0]
    chains = F.shape[1]

    lists = []
    for n in range(samples):
        for i in range(chains):
            lists.append(select_hungarian(F[n, i, :, :].squeeze(), exemplars, metric))

    return np.vstack(lists)


def label_and_sort_data(
    F, G, data, errors, by="Hungarian", distance_metric="correlation"
):

    if by == "Hungarian":
        # Take a "random" (1000 isin't special afaik) one as a starting point, this could be changed
        indices = distance_label_batch(F, F[1000, :, :], metric=distance_metric)
    else:
        ValueError("Unknown labeling techique {}".format(by))

    sorted_F = sort_by_batch_label(F, indices)
    sorted_G = sort_by_batch_label(G, indices, axis=3)

    return {"F": sorted_F, "G": sorted_G}


def get_quantiles_and_median(x, axis=0, quantiles=[0.025, 0.25, 0.75, 0.975]):
    """
    Just getting the default values in a convenient package.
    Yes technically quantile 0 would be median, but I want it in another variable
    """

    median = np.median(x, axis=axis)
    quantiles = np.quantile(x, quantiles, axis=axis)

    return median, quantiles


def quantile_data(F, G):

    result = {}

    result["F"] = np.quantile(F, [0.025, 0.5, 0.975], axis=0)
    result["G"] = np.quantile(G, [0.025, 0.5, 0.975], axis=0)

    return result


def construct_restrictions(
    F_file,
    G_file=None,
    F_to_restrict=None,
    G_to_restrict=None,
    reference_mz=29,
    restricted_mz=None,
    reference_time=None,
    restriction_type="full",
):

    F = pd.read_hdf(F_file, "F")
    F_ref = F.loc[:, reference_mz].to_numpy()[:, None]
    F_div = F / F_ref
    


    F_1_1 = []  # all other mz values
    F_1_2 = []

    F_2_1 = []  # reference mz value
    F_2_2 = []

    F_targets = []

    restricted_matrices = []
    restriction_index = 0
    restriction_names = {}

    if F_to_restrict is not None:

        restricted_matrices.append("F")

        ref_mz_loc = F_div.columns.get_loc(reference_mz)

        # Full restriction of row
        for name in F_to_restrict:

            row_index = F_div.index.get_loc(name)

            restriction_names[name] = restriction_index
            
            ref_mz_loc = F_div.columns.get_loc(reference_mz)
            if restriction_type == "full":
                size = len(F_div.columns)
                first_index = np.delete(np.arange(size), ref_mz_loc)

            elif restriction_type == "partial":
                size = len(restricted_mz)
                first_index = [F.columns.get_loc(x) for x in restricted_mz]
            # having a restriction of 1/1 is a bit redundant
            # remove the reference column

            # first_index = np.delete(np.arange(size), ref_mz_loc)

            F_1_2.append(first_index)
            F_1_1.append(np.repeat(restriction_index, len(first_index)))

            second_index = np.repeat(ref_mz_loc, len(first_index))
            F_2_2.append(second_index)
            F_2_1.append(np.repeat(restriction_index, len(first_index)))

            F_targets.append(F_div.iloc[row_index, first_index])
            restriction_index = restriction_index + 1

        F_1_1 = np.concatenate(F_1_1)
        F_1_2 = np.concatenate(F_1_2)
        F_2_1 = np.concatenate(F_2_1)
        F_2_2 = np.concatenate(F_2_2)

        # all other mz values for each resctricted component
        # restriced value's position in the F matrix
        F_a = np.column_stack([F_1_1, F_1_2])
        # reference mz position for each resctricted component
        F_b = np.column_stack([F_2_1, F_2_2])
        # reference m/z values for each restricted component
        F_targets = np.concatenate(F_targets)

    if G_to_restrict is not None:

        G = pd.read_hdf(G_file, "G")

        G_1_1 = []
        G_1_2 = []

        G_2_1 = []
        G_2_2 = []

        G_targets = []

        restricted_matrices.append("G")

        if type(reference_time) == int:
            G_ref_ind = reference_time
        else:
            G_ref_ind = G.index.get_loc(reference_time)

        G_ref = G.iloc[G_ref_ind, :].to_numpy()[None, :]

        G_div = G / G_ref

        for name in G_to_restrict:

            column_index = G_div.columns.get_loc(name)

            if name in restriction_names:
                # This is so that if one component has both F&G they end up in the same restricted component
                G_restriction_column = restriction_names[name]
            else:
                # F loop increases it at the end, or if no F is zero from begining
                G_restriction_column = restriction_index
                restriction_names[name] = G_restriction_column
                restriction_index = restriction_index + 1

            # TODO: Figure out the indexing

            first_index = np.delete(np.arange(len(G_div.index)), G_ref_ind)

            # Running time axis
            G_1_1.append(first_index)

            # Stationary
            G_1_2.append(np.repeat(G_restriction_column, len(first_index)))

            second_index = np.repeat(G_ref_ind, len(first_index))
            G_2_1.append(second_index)

            G_2_2.append(np.repeat(G_restriction_column, len(first_index)))

            G_targets.append(G_div.iloc[first_index, column_index])

        G_1_1 = np.concatenate(G_1_1)
        G_1_2 = np.concatenate(G_1_2)
        G_2_1 = np.concatenate(G_2_1)
        G_2_2 = np.concatenate(G_2_2)

        G_a = np.column_stack([G_1_1, G_1_2])
        G_b = np.column_stack([G_2_1, G_2_2])

        G_targets = np.concatenate(G_targets)

    result = {
        "restricted_components": np.array(list(restriction_names.keys()), dtype="S"),
        "restriction_indices": list(restriction_names.values()),
    }

    if F_to_restrict is not None:

        result["F_a"] = F_a
        result["F_b"] = F_b
        result["F_values"] = F_targets

    if G_to_restrict is not None:

        result["G_a"] = G_a
        result["G_b"] = G_b
        result["G_values"] = G_targets

    return result
