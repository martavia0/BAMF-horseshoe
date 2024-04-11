import argparse
from email.policy import default
import pandas as pd
import sys
from pathlib import Path
from analysis_codes import analysis
from time import ctime


def bool_validator(value):
    if value == "True":
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument("--input", type=str, required=True, help="modeled results from runner")
    parser.add_argument("--output", type=str, required=True,help="output analysis results position")
    parser.add_argument("--results", type=str, required=True,help="plot save position")
    parser.add_argument(
        "--benchmark", default=False, type=bool_validator, required=False,help="whether the benchmark exists"
    )
    parser.add_argument("--reference_library", default=None, type=str, required=False,help="benchmark file position")
    parser.add_argument(
        "--distance_metric", default="cityblock", type=str, required=False, help ="distance metric used to calculate the iterations distances between each other"
    )
    parser.add_argument("--cutoff", default=5, type=float, required=False,help="limit value to decide whether one component could be assigned using the benchmark")
    parser.add_argument("--mass_thres", default=0, type=float, required=False, help="mass conservation threshold to decide whether one iteration could be preserved")
    parser.add_argument("--sorting", default=True, type=bool_validator, required=False,help="sorting the iterations or not")
    parser.add_argument("--add_mz", default=True, type=bool_validator, required=False,help =" add the 4 m/zs based on m/z 44 or not")

    # changed: arguments = vars(parser.parse_args(sys.argv[1:]))
    arguments = vars(parser.parse_args())

    runfile = pd.read_hdf(arguments["input"], "runs")

    complete = runfile[runfile["status"] == "finished"]

    diagnostics = {"start", "stop", "duration", "filename", "status", "result", "seed"}
    groupers = [h for h in complete.columns if h not in diagnostics]
    gb = complete.groupby(groupers)

    print("Groups to analyze:", len(gb), ".", ctime(),flush=True)

    rows = []
    results = []
    # i  =0
    for group in gb:
        # i = i +1
        # if i ==3:

        full_row = {g: v for g, v in zip(groupers, group[0])}
        frame = group[1]
        if frame["datafile"].unique()[0] == '/data/user/wu_j1/Zurich-ToF-ACSM/processed/input_matrices_martavia_noised_20220315_winter.h5':
            frame["datafile"]="/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"
        sorted_data, metrics, indices_chains = analysis.analyze_frame(
            frame,
            benchmark=arguments["benchmark"],
            basepath=Path(arguments["results"]),
            distance_metric=arguments["distance_metric"],
            match_file=arguments["reference_library"],
            cutoff=arguments["cutoff"],
            mass_thres = arguments["mass_thres"],
            sorting = arguments["sorting"],
            add_mz_mode = arguments["add_mz"]
        )

        
        # results.append(sorted_data)

        full_row.update(metrics)
        rows.append(full_row)

    

    outpath = Path(arguments["output"])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    # pd.DataFrame(results).astype(str).to_hdf(arguments["output"], "results")
    pd.DataFrame(rows).to_hdf(arguments["output"], "metrics")
