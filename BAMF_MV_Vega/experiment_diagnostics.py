import pandas as pd
from time import ctime
from analysis_codes.diagnostics import *
from analysis_codes import analysis,helpers
from tqdm import tqdm
import seaborn as sns
import argparse
from pathlib import Path

sns.set_palette(sns.color_palette("colorblind"))


def cal_rhat(data,labels):

    n_samples = int(len(data.samples)/len(data.chains))
    n_chains = int(len(data.chains))
    rhat_ls = {}
    for label in labels:
        try:
            data_extract = data[label].values.reshape(n_samples,n_chains,*data[label].values.shape[1:])
        except:
            data_extract = data[label].values[n_samples:]
        if len(data_extract.shape) == 4:
            for i in (range(data_extract.shape[2])):
                for j in range(data_extract.shape[3]):
                    name=label + "["+str(i+1) + ","+str(j+1) +"]"
                    rhat_ls[name] = split_potential_scale_reduction_sorted(n_samples,n_chains,data_extract,i,j)
        elif len(data_extract.shape) == 3:
            for i in (range(data_extract.shape[-1])):
                name= label + "["+str(i+1) + ","+str(j+1) +"]"
                rhat_ls[name] = split_potential_scale_reduction_sorted(n_samples,n_chains,data_extract,i)
        else:
            rhat_ls[label] = split_potential_scale_reduction_sorted(n_samples,n_chains,data_extract)
    return rhat_ls


def trace_plot(data,rhat_ls,labels,results_path,result_file,sorting):
    print("trace plotting...")
    n_samples = int(len(data.samples)/len(data.chains))
    n_chains = int(len(data.chains))
    # first_chain = frame.iloc[0, :]["filename"] TODO
    
    outfilename = helpers.get_plot_filename(Path(results_path+"/diagnostics/"),frame.iloc[0, :]["filename"],sorting)
    outfilename.parent.mkdir(parents=True, exist_ok=True)
    if sorting == False:
        sorted = "_unsorted"
    else:
        sorted = ""


    palette = sns.color_palette(n_colors=n_chains)
    with helpers.Collated_output(outfilename) as outfile:

        fig = plt.figure()
        plt.title("R-hat" + sorted)
        plt.hist(rhat_ls.values(),bins = 50)
        plt.axvline(1.05,color="gray",linestyle="--",label=0.1)
        # plt.axvline(0.5,color="gray",linestyle="--",label=0.1)
        plt.axvline(1,color="gray",linestyle="--",label=0.1)
        plt.text(int(max(rhat_ls.values())/3),int(len(rhat_ls)/10),s = "R-hat mean:"+"{:.2f}".format(sum(rhat_ls.values()) / len(rhat_ls)))
        
        plt.grid()
        outfile.savefig(fig, "R-hat"+ sorted )
        plt.close()

        for label in labels:
            try:
                data_extract = data[label].values.reshape(n_samples,n_chains,*data[label].values.shape[1:])
            except:
                data_extract = data[label].values[n_samples:]
            if len(data_extract.shape) == 4:
                fig = plt.figure()
                for i in range(data_extract.shape[1]):
                    idx1,idx2 = 1,0
                    plt.plot(range(n_samples),data_extract[:,i,idx1,idx2],label="chain "+str(i+1),color = palette[i])
                plt.legend()
                plt.title(label + "["+str(idx1)+","+str(idx2)+"]"+sorted)
                outfile.savefig(fig, label  + "["+str(idx1)+","+str(idx2)+"]"+sorted)
                plt.close()
            elif len(data_extract.shape) == 3:
                fig=plt.figure()
                for i in range(data_extract.shape[1]):
                    idx1=0
                    plt.plot(range(n_samples),data_extract[:,i,idx1],label="chain "+str(i+1),color = palette[i])
                plt.legend()
                plt.title(label + "["+str(idx1)+"]_"+sorted)
                outfile.savefig(fig, label + "["+str(idx1)+"]"+sorted)
                plt.close()

def bool_validator(value):
    if value == "True":
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--benchmark", default=False, type=bool_validator, required=False
    )
    parser.add_argument("--reference_library", default=None, type=str, required=False)
    parser.add_argument(
        "--distance_metric", default="cityblock", type=str, required=False
    )
    parser.add_argument("--mass_thres", default=0, type=float, required=False)
    parser.add_argument("--cutoff", default=5, type=float, required=False)
    parser.add_argument("--sorting", default=True, type=bool_validator, required=False)
    parser.add_argument("--add_mz", default=True, type=bool_validator,required=False)

    # changed: arguments = vars(parser.parse_args(sys.argv[1:]))
    arguments = vars(parser.parse_args())

    runfile = pd.read_hdf(arguments["input"], "runs")

    complete = runfile[runfile["status"] == "finished"]

    diagnostics = {"start", "stop", "duration", "filename", "status", "result", "seed"}
    groupers = [h for h in complete.columns if h not in diagnostics]
    gb = complete.groupby(groupers)

    print("Groups to analyze:", len(gb), ".", ctime())

    rows = []
    results = []
    for group in gb:

        full_row = {g: v for g, v in zip(groupers, group[0])}
        frame = group[1]
        
        if frame["datafile"].unique()[0] == '/data/user/wu_j1/Zurich-ToF-ACSM/processed/input_matrices_martavia_noised_20220315_winter.h5':
            frame["datafile"]="/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
        if frame.iloc[0, :]["action"] == "sampling":

            sorted_data, metrics, indices_chains = analysis.analyze_frame(
                frame,
                benchmark=arguments["benchmark"],
                basepath=Path(arguments["results"]),
                distance_metric=arguments["distance_metric"],
                match_file=arguments["reference_library"],
                cutoff=arguments["cutoff"],
                mass_thres = arguments["mass_thres"],
                sorting = arguments["sorting"],
                save_plot=False,
                add_mz_mode = arguments["add_mz"]
            )
        
            rhat_ls = cal_rhat(sorted_data,["F","G","alpha_a","alpha_b","lp__"])

            trace_plot(sorted_data,rhat_ls,["F","G","alpha_a","alpha_b","lp__"],arguments["results"],frame.iloc[0, :]["filename"],arguments["sorting"])
