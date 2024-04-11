from analysis_codes import analysis, plotting, helpers
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


components = 5
filename = "Krakow_Mar"
output_folder = "/data/user/wu_j1/PMF_results/Krakow/"
reference_file_location = None

filename = "ZH_march_mz115"
output_folder = "/data/user/wu_j1/PMF_results/mz115/"
reference_file_location = "/data/user/wu_j1/Zurich-ToF-ACSM/mz115/processed/input_matrices_martavia_noised_20220315_winter.h5"


data_location = "ME2_results/" + filename + ".h5"

frame, netcdf = analysis.analyze_frame_for_pmf(
    output_folder + data_location,
    reference_file_location,
    components,
    str(components) + " component benchmark",
    benchmark=reference_file_location,
    dataname="Org_Specs_30",
    errorname="OrgSpecs_err_30",
    prname="amus",
    tsname="acsm_utc_time_30",
    # dataname="Specs",
    # errorname="errors",
    # prname="amus_nb",
    # tsname="timelist",
    cutoff=10,
    force_normalization=True,
    add_mz_mode=True,
)
full_frame = []
outfile = output_folder + (
    "analysis/" + filename + "_" + str(components) + "_components.nc4"
)
# netcdf.to_netcdf(outfile)

frame["experiment_name"] = filename
full_frame.append(frame)

metrics = pd.DataFrame(full_frame)

# metrics.to_hdf(output_folder / "metrics.h5", "metrics")
benchmark = True

outfilename = helpers.Path(
    output_folder + "plot/" + filename + "_" + str(components) + "_components.zip"
)
outfilename.parent.mkdir(parents=True, exist_ok=True)
with analysis.Collated_output(outfilename, save_types=["pdf"]) as outfile:
    f = plotting.mass_conservation(netcdf,"median")
    outfile.savefig(f, "conservation")
    plt.close()

    # plot standard G,F plots, with truth if benchmark
    f,F_metrics = plotting.plot_F(netcdf, benchmark=reference_file_location)
    outfile.savefig(f, "F")
    outfile.savetxt(F_metrics, "F")
    plt.close()

    f,G_metrics = plotting.plot_G(netcdf, benchmark=reference_file_location)
    outfile.savefig(f, "G")
    outfile.savetxt(G_metrics, "G")
    plt.close()

    f,median_G_metrics = plotting.plot_median_G(netcdf, benchmark=reference_file_location)
    outfile.savefig(f, "median_G")
    outfile.savetxt(median_G_metrics, "median_G")
    plt.close()
