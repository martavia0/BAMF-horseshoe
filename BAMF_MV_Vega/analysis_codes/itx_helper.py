import pandas as pd
import numpy as np


def igor_time_convert(t):

    return np.timedelta64(int(t), "s") + np.datetime64("1904-01-01")


def remove_mz(array, amus_nb, mz2remove=[16, 17, 18, 19, 20, 28]):
    ls = []

    for i in mz2remove:
        result = np.where(amus_nb == i)
        try:
            ls.append(result[0][0])
        except:
            pass
    array = pd.DataFrame(array).drop(columns=ls)
    return array


def split_matrix(values, keys, df, dict_waves):
    df.columns = keys[1:]
    dict_waves[keys[0]] = df
    return dict_waves


def split_vector(values, keys, df, dict_waves):
    dict_waves[keys[1]] = df
    return dict_waves


def split_waves(values, keys, dict_waves):

    df = pd.DataFrame([list(map(float, line.split("\t")[1:])) for line in values])

    try:
        if len(keys) > 2:
            dict_waves = split_matrix(values, keys, df, dict_waves)
        elif len(keys) == 2:
            dict_waves = split_vector(values, keys, df, dict_waves)
    except:
        raise ("input igor itx file is empty.")
    return dict_waves


def itx_to_pandas(path):
    file = open(path, "r", encoding="cp1252")
    waves = file.read().splitlines()
    file.close()

    beg_idx = [i for i, x in enumerate(waves) if x == "BEGIN"]
    end_idx = [i for i, x in enumerate(waves) if x == "END"]

    dict_waves = {}

    for i in range(len(beg_idx)):
        values = waves[beg_idx[i] + 1 : end_idx[i]]
        keys = waves[beg_idx[i] - 1].split("\t")

        dict_waves = split_waves(values, keys, dict_waves)
    return dict_waves
