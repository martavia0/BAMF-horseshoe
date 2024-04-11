import argparse
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
from analysis_codes import helpers, runners_same_init
# this file would have the same optimization results as the strating point for every parallel process, as the seed is the same as 1
# in the function model_init() inside the file runners_same_init.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument(
        "--procs", type=int, required=True, help="the number of processors"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="json file lists all options relating to the actual runs",
    )
    parser.add_argument(
        "--wait_n",
        default=3,
        required=False,
        help="time to wait before killing the process if nothing respond",
    )
    parser.add_argument(
        "--max_time_per_job",
        # default=pd.Timedelta("24 hours"),
        default=pd.Timedelta("7 days"),
        type=pd.Timedelta,
        required=False,
        help="maximum time to run one job",
    )
    parser.add_argument(
        "--check_interval",
        default=pd.Timedelta("10 min"),
        # 10 min
        type=pd.Timedelta,
        required=False,
        help="Q: time to wait before next move while detecting too much processes are running",
    )
    parser.add_argument(
        "--function",
        default="run",
        type=str,
        required=False,
        help="Q: string to define run or not run the model",
    )
    parser.add_argument("--log", type=str, required=True, help="log file position")


    arguments = vars(parser.parse_args())
    # changed from : arguments = vars(parser.parse_args(sys.argv[1:]))
    t_start = datetime.now()
    # unify the unit to second
    arguments["max_time_per_job"] = arguments["max_time_per_job"].total_seconds()
    arguments["check_interval"] = arguments["check_interval"].total_seconds()
    print(arguments)

    functions = {"run": runners_mjobs.run_model}
    output = Path(arguments["log"])

    result_output = str(output).replace("_info", "").replace(".h5", "")
    # generate all jobs with defined options in input file
    jobs = [
        (functions[arguments["function"]], opt)
        for opt in helpers.generate_options(Path(arguments["input"]), result_output)
    ]

    # execute the jobs
    with runners_mjobs.Timer_pool(
        nprocs=arguments["procs"], wait_n_times=arguments["wait_n"], logfile=output
    ) as tp:
        tp.execute(
            jobs,
            absolute_timeout=arguments["max_time_per_job"],
            check_timeout=arguments["check_interval"],
        )
    t_end = datetime.now()
    print("total time:", t_end - t_start, flush=True)
