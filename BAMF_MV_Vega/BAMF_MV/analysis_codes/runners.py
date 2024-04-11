import datetime
import logging
import multiprocessing as mp
import multiprocessing.connection as connection
import os
import pickle
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO
from operator import itemgetter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pystan
import xarray as xr
from sklearn.decomposition import NMF

from . import analysis, helpers

checked = set()

killwait = 5


class Silencer:
    """
    C++/Fortran subroutines don't care if you redirected via usual means
    This is to stop them from printing everything to stdout/stderr
    Because that is a mess when there are 2+ processes on.
    Pystans logger https://pystan.readthedocs.io/en/latest/logging.html didin't help.
    """
    def __init__(self, filepath):

        self.f = filepath.open("w")
        self.f_no = self.f.fileno()

        self.stdout_actual = os.dup(sys.stdout.fileno())
        self.stderr_actual = os.dup(sys.stderr.fileno())

        os.dup2(self.f_no, sys.stdout.fileno())
        os.dup2(self.f_no, sys.stderr.fileno())

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, traceback=None):

        os.dup2(self.stdout_actual, sys.stdout.fileno())
        os.dup2(self.stderr_actual, sys.stderr.fileno())

        os.close(self.stdout_actual)
        os.close(self.stderr_actual)

        self.f.close()


# the program to monitor the running time for each process, assign the core to new process,
# stop finished process and kill the process which took pretty long time

class Timer_pool:

    active_processes = {}
    start_times = {}
    end_times = {}
    running_id = 0

    table = None

    diagnostics = {
        "start", "stop", "duration", "filename", "status", "result", "seed"
    }

    def __init__(self, nprocs=2, wait_n_times=2, logfile=None):
        """_summary_

        Args:
            nprocs (int, optional): _description_. Defaults to 2.
            wait_n_times (int, optional): a int number, which multiplies the regular running time
                                        is the maximum waiting time. Defaults to 2.
            logfile (_type_, optional): _description_. Defaults to None.
        """

        self.max_processes = nprocs
        self.n = wait_n_times
        self.logfile = logfile

    def execute(self, jobs, absolute_timeout=None, check_timeout=None):
        """this function executes the assigned jobs

        Args:
            jobs (_type_): the job list with defined parameters
            absolute_timeout (_type_, optional): _description_. Defaults to None.
            check_timeout (_type_, optional): _description_. Defaults to None.
        """

        # Using pystans own thing would be much easier.
        # But the optimization never converges for some starting points...
        # and then you get no data even from the converged runs.
        # (Well it will eventually exhaust the treedepth, but that is a long time to wait)
        # So here we are.

        # Note: None of the processes should return anything

        # Data used for bookkeeping

        # table records job's parameters
        self.table = pd.DataFrame(j for i, j in jobs)

        # Compile all models that will be used later at once before multithreading
        for item in self.table["model"].unique():
            compile_model(Path(item))

            premodels = self.table[np.logical_or(
                self.table["init"] == "model_init",
                self.table["init"] == "multimodel",
            )]

            # compile the premodel
            if len(premodels) > 0:
                for item in premodels["initoptions"]:
                    if type(item["premodel"]) is list:
                        for k in item["premodel"]:
                            compile_model(Path(k))
                    else:
                        mod = item["premodel"]
                        compile_model(Path(mod))

        self.table["function"] = [i.__name__ for i, j in jobs]
        self.table["status"] = "undefined"
        self.table["start"] = pd.NaT
        self.table["stop"] = pd.NaT
        self.table["duration"] = np.nan

        # Hmmm, maybe there is a pattern here.
        if "initoptions" in self.table.columns:
            self.table.loc[:, "initoptions"] = self.table["initoptions"].apply(str)
        if "extract" in self.table.columns:
            self.table.loc[:, "extract"] = self.table["extract"].apply(str)
        # Path doesn't like to go quietly in to hdf5
        if "filename" in self.table.columns:
            self.table.loc[:, "filename"] = self.table["filename"].apply(str)

        self.groupers = [
            item for item in self.table.columns if item not in self.diagnostics
        ]

        self.selector = itemgetter(*self.groupers)

        self.absolute_limit = absolute_timeout

        # list of function, keyword pairs
        for i, job in enumerate(jobs):

            self.running_id = i

            pipeline_full = len(self.active_processes) >= self.max_processes

            while pipeline_full:
                self.wait(check_timeout)
                pipeline_full = len(
                    self.active_processes) >= self.max_processes

            self.spawn(job)

        while len(self.active_processes) > 0:
            self.wait(check_timeout)

    def spawn(self, job):
        """assign the job to one Process object, start the Process's activity
        and record the Process id, starting time and status.

        Args:
            job (_type_): the function name and assigned parameters
        """

        p = mp.Process(target=job[0], kwargs=job[1])
        p.start()

        self.active_processes[self.running_id] = p
        self.table.loc[self.running_id, "start"] = pd.Timestamp("now")
        self.table.loc[self.running_id, "status"] = "started"

    def wait(self, timeout):
        # Well waiting like this "wastes" a core but idk what to do about it

        print(pd.Timestamp("now"),
              "running:",
              self.active_processes.keys(),
              flush=True)

        # get the sentinels of each active process, sentinel is a numeric handle of a system object
        # which will become “ready” when the process ends.
        sentinels = [p.sentinel for k, p in self.active_processes.items()]

        # wait certain time to have some process finished. if exists, then get the id of the
        # finished process; if nothing finished, then return empty list.
        starttime = datetime.datetime.now()
        print("start waiting connention for",
              timeout,
              "s :",
              starttime,
              flush=True)
        ready = connection.wait(sentinels, timeout=timeout)
        stoptime = datetime.datetime.now()
        print("finish waiting connention:", stoptime, flush=True)
        # print("waited", stoptime - starttime, flush=True)

        done = {}

        # I don't get why they tell me they're ready, then complain.
        # Sentinel reports ready, then close raises error.
        # So that's why we wait few seconds to let the process get its business in order
        time.sleep(killwait)


        if ready: 

            for identifier, proc in self.active_processes.items():
                if proc.sentinel in ready:
                    done[identifier] = proc.exitcode
                    proc.close()

        self.reap(done)
        self.log(self.logfile)  # record the log into file

    def log(self, filepath):
        """save the log history

        Args:
            filepath (_type_): the file position to save the log
        """
        if (self.logfile is not None) & (self.table is not None):
            #self.table.to_hdf(filepath, key="runs" )
            self.table.to_hdf(filepath, key="runs", mode = 'a') #Marta Via correction


    def reap(self, done):
        """stop the finished process

        Args:
            done (function): the status of current process
        """

        now = pd.Timestamp("now")

        # Remove those process that are done
        removed = []
        for tag, code in done.items():
            self.table.loc[tag, "stop"] = pd.Timestamp("now")
            if code != 0:
                self.table.loc[tag, "status"] = str(code)
            else:
                self.table.loc[tag, "status"] = "finished"
                self.table.loc[tag, "duration"] = (
                    pd.Timestamp("now") -
                    self.table.loc[tag]["start"]).total_seconds()

            removed.append(tag)

        for tag in removed:
            self.active_processes.pop(tag)

        # ----------------below : kill he process which clogged the memory or spend too much time----------------
        # kill and remove those that overstayed their welcome
        removed = []
        for tag, proc in self.active_processes.items():
            if (pd.Timestamp("now") - self.table.loc[tag, "start"]
                ).total_seconds() >= self.absolute_limit:
                proc.kill()
                self.table.loc[tag, "stop"] = pd.Timestamp("now")
                self.table.loc[tag, "status"] = "timeout"
                removed.append(tag)

        if removed != []:
            time.sleep(killwait)

        for tag in removed:
            self.active_processes[tag].close()
            self.active_processes.pop(tag)

        if len(self.active_processes) == 0:
            return  # if no more process is running, then directly exit this deleting process function

        # stuck chain detection, if double regular running time are used to run this process,
        # then it is classifies as stucked.
        removed = []
        means = self.table.groupby(self.groupers)["duration"].mean()
        for tag, proc in self.active_processes.items():
            group = self.selector(self.table.loc[tag, :])

            if (self.n * means[group]) < (
                    pd.Timestamp("now") -
                    self.table.loc[tag]["start"]).total_seconds():
                proc.kill()
                self.table.loc[tag, "stop"] = pd.Timestamp("now")
                self.table.loc[tag, "status"] = "chain timeout"
                removed.append(tag)

        if removed != []:
            time.sleep(killwait)

        for tag in removed:
            self.active_processes[tag].close()
            self.active_processes.pop(tag)

        return

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, traceback=None):

        for tag, proc in self.active_processes.items():
            proc.kill()

        time.sleep(killwait)

        # Give a sec for kill to take effect.
        for tag, proc in self.active_processes.items():
            proc.close()

        print(pd.Categorical(self.table["status"]).describe(), flush=True)

        self.log(self.logfile)

# ----------------------------------------running model------------------------


def compile_model(modelpath, force=False):
    """This function compiles all models that will be run later, the model.stan files
    will be compiled and saved as model.pkl file to avoid repeated recompilation.

    Args:
        modelpath (_type_): The path of models to be run later.
        force (bool, optional): Forces the program to compile the model regardless of
        the presence of the precompiled model. Defaults to false.
    """

    # Compiling them in the worker processes would mess with timing,
    # Also making sure it only gets compiled once

    # if the compiled model exists, then skip compile
    picklename = modelpath.with_suffix(".pkl")
    if picklename in checked:
        return

    # check whether the compiled model is created from the stan file
    use_pickle = False
    if picklename.exists():
        sp = picklename.stat()
        sm = modelpath.stat()
        if sp.st_mtime >= sm.st_mtime:
            use_pickle = True

    if use_pickle and (not force):
        checked.add(picklename)
        print(f"Using precompiled model: {picklename}", flush=True)
        return

    # compile the model
    with modelpath.open("r") as infile:
        model_definition = infile.read()
    model = pystan.StanModel(model_code=model_definition)

    # save the compiled model into pkl file
    with picklename.open("wb") as f:
        pickle.dump(model, f)
        print("Compiled model:", picklename, flush=True)
        checked.add(picklename)

def do_nothing(data, meta, extra):
    meta["norm_const"] = 1
    return data, meta, extra

def basic_normalization(data, meta, extra):
    """normalization of the X and sigma matrix based on the mean value of X matrix.

    Args:
        data (_type_): input data matrix
        meta (_type_): meta matrix as position to record information
        extra (_type_): _description_

    Returns:
        _type_: _description_
    """

    const = np.nanmean(data["X"])

    data["X"] = data["X"] / const
    data["sigma"] = data["sigma"] / const

    meta["norm_const"] = const

    return data, meta, extra

def restricted(data, meta, extra):
    """ read the input file when the constrained files exists
    """
    data, meta, extra = basic_normalization(data, meta, extra)

    with h5py.File(meta["restrictionfile"], "r") as infile:

        if "F_a" in infile:
            # The indices start from 1 in STAN
            # So while prepping them it is easier to use python indices, but we need to add +1 to them
            data["F_a"] = infile["F_a"][()] + 1
            data["F_b"] = infile["F_b"][()] + 1
            data["F_values"] = infile["F_values"][()]
            data["r"] = data["F_a"].shape[0]
            data["F_sigma"] = meta["F_weight"] * np.ones(data["r"])

        if "G_a" in infile:
            data["G_a"] = infile["G_a"][()] + 1
            data["G_b"] = infile["G_b"][()] + 1
            data["r_g"] = data["G_a"].shape[0]
            data["G_values"] = infile["G_values"][()]
            data["G_sigma"] = meta["G_weight"] * np.ones(data["r_g"])

        # Lists of the names of the cosntrained components and what their location is
        meta["constrained_components"] = infile["restricted_components"][()]
        meta["constrained_indices"] = infile["restriction_indices"][()]

    return data, meta, extra

def read_benchmark(filepath):
    """read the input file, also read the benchmark file as the original data
    """

    data, meta, extra = read_input(filepath)

    # The original data is stored in the result file, so the postprocessing doesn't have to mix and match
    original_data = {}
    original_data["F_orig"] = pd.read_hdf(filepath, "F").values
    original_data["G_orig"] = pd.read_hdf(filepath, "G").values


    meta["components"] = pd.read_hdf(filepath,
                                     "F").index.to_numpy().astype("S")
    meta.update(original_data)

    return data, meta, extra

def read_input(filepath):
    """This function read the data from the input file and precoess the data into unified units for analysing.
    """

    data = pd.read_hdf(filepath, "data")
    error = pd.read_hdf(filepath, "error")

    actual_times = data.index
    actual_times= pd.to_datetime(actual_times) #Marta modified this line
    scaled_time, _ = analysis.scale_times(actual_times)
    times = pd.to_datetime(data.index).to_series().dt.strftime("%Y-%m-%d %H:%M:%S").values.astype("S20") #Marta modification
    #times = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S").values.astype("S20")
    columns = data.columns.values.astype("S")
    # time difference between every entries
    timedeltas = (pd.to_datetime(data.index).to_series().diff().dt.total_seconds()[1:].values)  #Marta modification
#    timedeltas = (data.index.to_series().diff().dt.total_seconds()[1:].values)  

    # Stuff that goes to model
    data_out = {
        "X": data.values,
        "m": data.shape[1],
        "n": data.shape[0],
        "sigma": error.values,
        "timesteps": timedeltas,
        "beta_tau": np.log(np.prod(data.shape)),
    }

    # A lower bound on a variable in the model
    data_out["minerr"] = data_out["sigma"].min() / 10

    # Stuff that goes to output file
    meta = {"columns": columns, "time_index": times}

    # Stuff that goes to preprocessing (if applicable)
    extra = {"time_actual": actual_times, "time_delta": timedeltas}

    return data_out, meta, extra


preprocessing_functions = {
    "nothing": do_nothing,
    "basic": basic_normalization,
    "restricted": restricted,
}

loaders = {"basic": read_input, "benchmark": read_benchmark}


def add_metadata(meta, F_weight, G_weight, restrictionfile, preprocessing):
    if F_weight is not None:
        meta["F_weight"] = F_weight

    if G_weight is not None:
        meta["G_weight"] = G_weight

    if restrictionfile is not None:
        meta["restrictionfile"] = restrictionfile

    if preprocessing not in preprocessing_functions:
        raise ValueError(f"Unknown preprocesser {preprocessing}")
    return meta


def run_model(action,
              datafile,
              description,
              init,
              loader,
              model,
              iterations,
              components,
              preprocessing,
              filename,
              initoptions,
              delta,
              treedepth,
              seed,
              extract=["tau", "F", "G", "alpha_a", "alpha_b", "lp__"],
              F_weight=None,
              G_weight=None,
              restrictionfile=None,
              G_file=None,
              partial_const_file=None,
              subset_amus=10,
              subset_method="ev"):
    """
    Should extract be explicitly always given?
    You can now still override it.
    """

    outpath = Path(filename)  # path to save the results
    logpath = outpath.with_suffix(".log")  # path to save the log
    outpath.parent.mkdir(parents=True,
                         exist_ok=True)  # Make sure output directory exists

    def model_init():

        preoptions = {}

        if "premodel_options" in initoptions:
            preoptions.update(initoptions["premodel_options"])

        inits = premodel.optimizing(data,
                                    **preoptions,
                                    as_vector=False,
                                    seed=seed)

        return inits["par"]  # TODO reason to change this
        # return inits

    def init_nothing():
        return {}

    def nmf_init():

        preoptions = {"max_iter": 2000}

        if "premodel_options" in initoptions:
            preoptions.update(initoptions["premodel_options"])

        print("Running NMF initialization.", preoptions)

        nonzerodata = data["X"].copy()
        nonzerodata[nonzerodata <= 0] = 0

        model_defined = NMF(
            n_components=data["p"],
            max_iter=preoptions["max_iter"],
            init="random",
            random_state=seed,
        )
        G = model_defined.fit_transform(nonzerodata)
        F = model_defined.components_

        # Unfortunately our model doesn't accept 0 exact
        G[G == 0] = 1e-10
        F[F == 0] = 1e-10

        def norm(F, G):

            sums = F.sum(axis=1)

            F = F / sums[:, None]
            G = G * sums

            return F, G

        F, G = norm(F, G)

        print("Ran NMF.")

        return {"F": F, "G": G}

    def G_init():
        # nc_file
        # partial_const

        G = xr.open_dataset(G_file)["G"].values
        G_median = pd.DataFrame(
            np.median(G, axis=0),
            columns=xr.open_dataset(G_file)["labels"].values,
            index=xr.open_dataset(G_file)["timestep"].values,
        )
        partial_const = pd.read_hdf(
            partial_const_file,
            key="partial_const",
        )

        return {"G": G_median * partial_const.values}

    with Silencer(logpath):

        picklename = Path(model).with_suffix(".pkl")
        loaded_model = pickle.load(picklename.open("rb"))

        premodel = None

        if "premodel" in initoptions:
            if type(initoptions["premodel"]) is list:
                premodel = []
                for k in initoptions["premodel"]:
                    prepicklename = Path(k).with_suffix(".pkl")
                    premodel.append(pickle.load(prepicklename.open("rb")))
            else:
                prepicklename = Path(
                    initoptions["premodel"]).with_suffix(".pkl")
                premodel = pickle.load(prepicklename.open("rb"))

        # model_name = model[model.rfind("/") + 1:model.rfind(".stan")]
        model_name = Path(model).stem
        
        # Metadata is added to output
        data, meta, extra = loaders[loader](Path(datafile))

        meta = add_metadata(meta, F_weight, G_weight, restrictionfile,
                            preprocessing)

        data, meta, extra = preprocessing_functions[preprocessing](data, meta,
                                                                   extra)
  
        if "samples" in initoptions:    
            extra["samples"] = initoptions["samples"]

        extra["seed"] = seed

        data["p"] = components

        
        if ("subset" in model_name) or ("anchor" in model_name):
            if subset_method == "ev":
                F = pd.read_hdf(Path(datafile), "F")
                G = pd.read_hdf(Path(datafile), "G")
                error = pd.read_hdf(Path(datafile), "error")

                total = (np.dot(G, F) / error).sum(axis=0).to_frame().T
                explained_var = pd.concat(
                    [
                        (
                            (np.outer(G.iloc[:, i], F.iloc[i, :]) / error)
                            .sum(axis=0)
                            .to_frame()
                            .T
                        )
                        / total
                        for i in range(data["p"])
                    ],
                    axis=0,
                ).set_axis(G.columns, axis=0)

                selected_ls_GF_idx = explained_var.std().values.argsort()[::-1][:subset_amus]
                
                data["S"] = len(selected_ls_GF_idx)
                # Reason to plus one here, difference of indexing system between stan and python,
                # in python strat from 0, while in stan , it starts from 1. 
                data["selected_indexes"] = np.sort(selected_ls_GF_idx +1) 
                print("selected m/zs:", F.columns[np.sort(selected_ls_GF_idx)], flush=True)
            else:
                F = pd.read_hdf(Path(datafile), "F")
                amu_prob = pd.read_csv("/data/user/wu_j1/bamf/samples_prob_df.csv")
                selected_amu = amu_prob.sort_values(by=["probability","mean_cor"],ascending=[False,True]).iloc[:10].amu.values
                selected_indexes = [list(F.columns).index(i) for i in selected_amu]
                data["selected_indexes"] = np.sort(np.array(selected_indexes)+1)
                data["S"] = len(selected_indexes)
                print("selected m/zs:", F.columns[np.sort(selected_indexes)], flush=True)
            if "anchor" in model_name:
                logging.info("anchoring data")
                data["F"] = F.values[:, selected_ls_GF_idx]
                # extract.remove("F")
                data["not_selected_indexes"] = np.sort(explained_var.std().values.argsort()[::-1][subset_amus:] +1)
        # model arguments input
        args = [
            "action",
            "datafile",
            "init",
            "loader",
            "model",
            "iterations",
            "components",
            "preprocessing",
            "filename",
            "initoptions",
            "delta",
            "treedepth",
            "seed",
            "extract",
        ]

        for key in args:
            locs = locals()
            put = locs[key]
            if put is not dict:
                meta["arg_" + key] = put
            else:
                meta["arg_" + key] = str(put)

        # Pystans runner really doesn't like partial functions which is the more elegant way to do this

        init_functions = {
            "model_init": model_init,
            "nothing": init_nothing,
            "nmf_init": nmf_init,
            "G_init": G_init,
        }

        if init not in init_functions:
            raise ValueError(f"Unknown init {init}")

        init_function = init_functions[init]

        wall_start = pd.Timestamp("now")

        if action == "sampling":
            # to get the seed works for initilization
            # fitted = None
            # while fitted == None:
            #     try:

            fitted = loaded_model.sampling(
                data=data,
                iter=iterations,
                init=init_function,
                chains=1,
                n_jobs=1,
                seed=seed,
                control={
                    "adapt_delta": delta,
                    "max_treedepth": treedepth
                },
            )

            # except:
            #     seed = seed + 1
            #     pass

            results = fitted.extract(pars=extract,
                                     permuted=False,
                                     inc_warmup=True)
        elif action == "map":
            # We want random init here
            results = loaded_model.optimizing(data=data,
                                              iter=iterations,
                                              as_vector=True,
                                              seed=seed)
        elif action == "vb":
            fitted = loaded_model.vb(data=data,
                                        init=init_function,
                                              iter=iterations,
                                              seed=seed)
            results = {}
            for e in extract:
                e_index = np.array([fitted["sampler_param_names"].index(i) for i in (fitted["sampler_param_names"]) if e in i])
                if e == "G":
                    result=np.array(fitted["sampler_params"])[e_index].reshape(1,-1,data["p"],fitted["args"]["output_samples"])
                elif e == "F":
                    result=np.array(fitted["sampler_params"])[e_index].reshape(1,data["p"],-1,fitted["args"]["output_samples"])
                elif e != "lp__":
                    result=np.array(fitted["sampler_params"])[e_index].reshape(1,data["p"],fitted["args"]["output_samples"])
                else:
                    result=np.array(fitted["sampler_params"])[e_index].reshape(1,fitted["args"]["output_samples"])
                result = np.moveaxis(result, -1, 0)
                results[e] = result
                
                # np.array(fitted["sampler_params"])[fitted["args"]["output_samples"]]
            results = OrderedDict(results)
            # results = fitted.extract(pars=extract,
            #                          permuted=False,
            #                          inc_warmup=True)
                                     
        
        else:
            raise ValueError(f"Unknown action {action}")
        
        # if "anchor" in model_name:
        #     items  = list(results.items())
        #     items.append(("selected_idxs",fitted.data["selected_indexes"]))
        #     items.append(("selected_F",fitted.data["F"]))
        #     items.sort()
        #     results = OrderedDict(items)

        wall_end = pd.Timestamp("now")
        wall_duration = (wall_end - wall_start).total_seconds()
        print("seed:", seed, flush=True)
        
        
        helpers.dump_hdf5(
            outpath,
            **results,
            description=description,
            wall_duration=wall_duration,
            **data,
            **meta,
        )

