import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.stats import ks_2samp, binom_test, kruskal, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error,mean_absolute_error

register_matplotlib_converters()

plt.rcParams["figure.figsize"] = (40, 14)
plt.rcParams["font.size"] = 40
plt.tight_layout()

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2.0)
sns.set_palette(sns.color_palette("colorblind"))


def cal_metrics(data1, data2, metric):
    coeffs = []
    try:
        if len(data1.shape) == 1:
            data1 = data1.reshape(1, -1)
    except:
        raise ("input data is empty.")
    for i in range(data1.shape[0]):
        if metric == mean_squared_error:
            coeff = metric(data2, data1[i, :], squared=True)/np.mean(data2)
        elif metric == mean_absolute_error:
            coeff = metric(data2, data1[i, :])/np.mean(data2)
        else:
            try:
                coeff, _ = metric(data2, data1[i, :], axis=None)
            except:
                coeff, _ = metric(data2, data1[i, :])
        coeffs.append(coeff)
    return coeffs


def plot_benchmark(data, palette=None, label="G"):

    if label == "F":
        idx_factor = 0
        factor_position = data.index
        x_position = data.columns
    else:
        idx_factor = 1
        factor_position = data.columns
        x_position = data.index

    fig, axes = plt.subplots(data.shape[idx_factor], sharex=True)
    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        ylabel = np.array(factor_position)[i]
        if label == "F":
            x.bar(
                np.array(x_position),
                np.array(data[data.index == ylabel]).ravel(),
                label=np.array(factor_position)[i],
                color=palette[i],
            )

        else:
            x.plot(data.index, data[ylabel], "--", label=ylabel, color=palette[i])

        twin = axes[i].twinx()

        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)


#     return fig


def plot_diagnostic(di, name, thresh=0.05):

    n = int(di.shape[0] / 2)
    half = n + int(n / 2)

    # first half part of sampling draws
    first = di[n:half, :].ravel()
    second = di[half:, :].ravel()

    ks, p = ks_2samp(first, second)

    fig = plt.figure()

    ax = sns.kdeplot(first, shade=True, alpha=0.5, color="r", label="first half")
    ax = sns.kdeplot(second, shade=True, alpha=0.5, color="b", label="second half")

    if p < thresh:
        textcol = "r"
    else:
        textcol = "g"

    ax.set_title(f"{name}, p-value:{p:1.2f}", c=textcol)
    ax.legend()
    return fig, p


def errorbar_batched(
    fig,axes,data, labels, title, n_rows=5, xlabel="", palette=None, fontsize=22, label="factor"
):


    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    axes[0].set_title(title)

    if label == "factor":
        if len(data["F"].shape) == 2:
            quants = np.array([data["F"], data["F"], data["F"]])
        else:
            quants = np.quantile(data["F"], q=[0.25, 0.5, 0.75], axis=0)
        # if "selected_variable" in data:
        if "selected_variable" in data and data.F.shape[-1] == len(data.selected_variable):
            variable = data["selected_variable"].values
        else:
            variable = data["variable"].values
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                variable,
                quants[1, i, :],
                np.abs(quants[[0, 2], i, :] - quants[1, i, :]),
                color=palette[i],
                marker="o",
                markeredgecolor=palette[i],
                capthick=1,
                linewidth=0,
                elinewidth=1,
                ecolor=palette[i],
                capsize=3,
                markersize=5,
            )
            plt.xlabel(xlabel)
            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend([labels[i]], loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)
            x.set_ylim([0, x.get_ylim()[1]]) 
        plt.xticks(rotation=90)
    elif label == "time":
        if len(data["G"].shape) == 2:
            quants = np.array([data["G"], data["G"], data["G"]])
        else:
            quants = np.quantile(data["G"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):
            # plot model results
            x.plot(
                data["timestep"],
                quants[1, :, i],
                "x",
                color=palette[i],
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data["timestep"]),
                quants[0, :, i],
                quants[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )
    return fig, axes


def plot_F(data, benchmark=False, extra_data=None):

    fig, axes = plt.subplots(data["G"].shape[-1], sharex=True)
    df = pd.DataFrame(columns=data["labels"],index=["F_spear","F_rmse","F_mae","F_spear_std","F_rmse_std","F_mae_std"])
    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        if benchmark and "F_orig" in data.keys():
            if np.array(data["labels"])[i] in np.array(data["orig_labels"]):
                orig_index = np.where(
                    np.array(data["orig_labels"]) == np.array(data["labels"])[i]
                )[0]
                x.bar(
                    np.array(data["variable"]),
                    np.array(data["F_orig"])[orig_index, :].ravel(),
                    width=0.5,
                    color="k",
                    alpha=0.3,
                    label="reference",
                )
                if "selected_variable" in data:
                    x.bar(
                    np.setdiff1d(data["variable"].values,data["selected_variable"].values),
                    np.array(data["F_orig"].loc[:,np.setdiff1d(data["variable"].values,data["selected_variable"].values)])[orig_index, :].ravel(),
                    width=0.5,
                    color="r",
                    alpha=0.7,
                    label="reference",
                    )
                if len(data["F"].shape) == 2:
                    modeled_F = np.array(data["F"])[i, :]
                else:
                    modeled_F = np.array(data["F"])[:, i, :]

                if "selected_variable" in data and data.F.shape[-1] == len(data.selected_variable):
                    benchmark_F = np.array(data["F_orig"].loc[:,data["selected_variable"].values])[orig_index, :].ravel()
                else: 
                    benchmark_F = np.array(data["F_orig"])[orig_index, :].ravel()

                coeff_spear = cal_metrics(
                    modeled_F,
                    benchmark_F,
                    spearmanr,
                )

                coeff_rmse = cal_metrics(
                    modeled_F,
                    benchmark_F,
                    mean_squared_error,
                )
                coeff_mae = cal_metrics(
                    modeled_F,
                    benchmark_F,
                    mean_absolute_error,
                )

                df.iloc[:,i] = [np.mean(coeff_spear),np.mean(coeff_rmse),np.mean(coeff_mae),np.std(coeff_spear),np.std(coeff_rmse),np.std(coeff_mae)]
                ylabel = (
                    str(np.array(data["labels"])[i])
                    + "\nSpearman's ρ:\n"
                    + "{:.4f}".format(np.mean(coeff_spear))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_spear))
                    + "\nRMSE:\n"
                    + "{:.4f}".format(np.mean(coeff_rmse))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse))
                    + "\nMAE:\n"
                    + "{:.4f}".format(np.mean(coeff_mae))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_mae))
                )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            ylabel,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_xticks(ticks=data["variable"],labels=data["variable"].values)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)
    fig, axes = errorbar_batched(
        fig,axes,
        data,
        labels=np.array(data["labels"]),
        title="F",
        n_rows=data["G"].shape[-1],
        label="factor",
    )

    return fig,df


def plot_individual_F(data, compound_idx, title="F", benchmark=False, sample_size=7):

    fig, axes = plt.subplots(sample_size, 1, sharex=True)

    compound = data["labels"].values[compound_idx]

    if benchmark:
        orig_index = np.where(np.array(data["orig_labels"]) == compound)[0]
        title = title + ", " + str(np.array(data["orig_labels"])[orig_index])
    else:
        title = title + f", {compound}"

    colors = sns.color_palette(n_colors=len(axes))
    if "selected_variable" in data and data.F.shape[-1] == len(data.selected_variable):
        variable = data["selected_variable"].values
    else:
        variable = data["variable"].values
    for i, x in enumerate(axes):
        ylabel = "sample " + str(i)
        x.plot(
            variable,
            np.array(data["F"])[-i, compound_idx, :],
            linestyle="",
            marker="o",
            color=colors[i],
            alpha=0.5,
            label=f"{i}",
        )
        if "selected_variable" in data and data.F.shape[-1] == len(data.selected_variable):
            benchmark_F = np.array(data["F_orig"].loc[:,data["selected_variable"].values])[orig_index, :].ravel()
        else: 
            benchmark_F = np.array(data["F_orig"])[orig_index, :].ravel()
        if benchmark and compound in np.array(data["orig_labels"]):
            orig_index = np.where(np.array(data["orig_labels"]) == compound)[0]

            x.bar(
                np.array(data["variable"]),
                np.array(data["F_orig"])[orig_index, :].ravel(),
                width=0.5,
                color="k",
                alpha=0.25,
            )

            coeff_spear = cal_metrics(
                np.array(data["F"])[-i, compound_idx, :],
                benchmark_F,
                spearmanr,
            )

            coeff_rmse = cal_metrics(
                np.array(data["F"])[-i, compound_idx, :],
                benchmark_F,
                mean_squared_error,
            )


            coeff_mae = cal_metrics(
                np.array(data["F"])[-i, compound_idx, :],
                benchmark_F,
                mean_absolute_error,
                )

            ylabel = (
                "sample "
                + str(i)
                + "\nSpearman's ρ:"
                + "{:.4f}".format(np.mean(coeff_spear))
                + "\nRMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse))
                + "\nMAE:"
                    + "{:.4f}".format(np.mean(coeff_mae))
            )
        twin = axes[i].twinx()
        twin.set_ylabel(
            # ylabel,
            # labelpad=len("sample " + str(i)),
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])

        x.set_xticks(data["variable"], minor=True)
        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)

    return fig


def plot_G(data, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]):
    fig, axes = plt.subplots(data["G"].shape[-1], sharex=True)
    
    # plot benchmark
    df = pd.DataFrame(columns=data["labels"],index=["G_pear","G_rmse","G_mae","G_pear_std","G_rmse_std","G_mae_std"])
    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        if benchmark and np.array(data["labels"])[i] in np.array(data["orig_labels"]):

            orig_index = np.where(
                np.array(data["orig_labels"]) == np.array(data["labels"])[i]
            )[0]

            x.plot(
                data["timestep"],
                np.array(data["G_orig"])[:, orig_index],
                "k--",
                label="Truth",
            )

            # pearson coeff notation
            if len(data["G"].shape) == 2:
                modeled_G = np.array(data["G"])[:, i]
            else:
                modeled_G = np.array(data["G"])[:, :, i]
                
            coeff_pear = cal_metrics(
                modeled_G,
                np.array(data["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )

            coeff_rmse = cal_metrics(
                modeled_G,
                np.array(data["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )
            coeff_mae = cal_metrics(
                modeled_G,
                np.array(data["G_orig"])[:, orig_index].ravel(),
                mean_absolute_error,
                )

            df.iloc[:,i] = [np.mean(coeff_pear),np.mean(coeff_rmse),np.mean(coeff_mae),np.std(coeff_pear),np.std(coeff_rmse),np.std(coeff_mae)]
            ylabel = (
                str(np.array(data["labels"])[i])
                + "\nPearson's r:\n"
                + "{:.4f}".format(np.mean(coeff_pear))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_pear))
                + "\nRMSE:\n"
                + "{:.4f}".format(np.mean(coeff_rmse))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse))
                + "\nMAE:\n"
                + "{:.4f}".format(np.mean(coeff_mae))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_mae))
            )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            ylabel,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    fig.autofmt_xdate()
    fig, axes = errorbar_batched(
        fig,axes,
        data,
        labels=np.array(data["labels"]),
        title=title,
        n_rows=data["G"].shape[-1],
        label="time",
    )
    return fig,df


def plot_individual_G(data, title="G", benchmark=False, sample_size=7):
    """
    plot the multiple G results on each sub plot.
    """

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=sample_size)

    for i, x in enumerate(axes):
        orig_index = np.where(
            np.array(data["orig_labels"]) == np.array(data["labels"])[i]
        )[0]

        for j in range(sample_size):

            x.plot(
                data["timestep"],
                data["G"][-j, :, i],
                "x--",
                color=colors[j],
                alpha=0.5,
                label=f"{j}",
            )

        if benchmark:
            x.plot(
                data["timestep"], data["G_orig"][:, orig_index], "k--", label="Truth"
            )
            # pearson coeff notation

        twin = axes[i].twinx()
        twin.set_ylabel(
            str(np.array(data["labels"])[i]),
            # labelpad=len(data["labels"][i]),
            # # rotation=90,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            # ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])
        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        tick.label.set_rotation(90)

    fig.autofmt_xdate()

    return fig


def plot_median_G(data, title="G_median", benchmark=False, quantiles=[0.5]):

    q = np.quantile(data["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes))
    df = pd.DataFrame(columns=data["labels"],index=["G_pear","G_rmse","G_mae","G_pear_std","G_rmse_std","G_mae_std"])
    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        x.plot(
            data["timestep"],
            q[0, :, i],
            "--",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
        )

        if benchmark:
            if np.array(data["labels"])[i] in np.array(data["orig_labels"]):
                orig_index = np.where(
                    np.array(data["orig_labels"]) == np.array(data["labels"])[i]
                )[0]
                x.plot(
                    data["timestep"],
                    np.array(data["G_orig"])[:, orig_index],
                    "k--",
                    label="Truth",
                )

                coeff_pear = cal_metrics(
                    q[0, :, i],
                    np.array(data["G_orig"])[:, orig_index].ravel(),
                    pearsonr,
                )
                coeff_rmse = cal_metrics(
                    q[0, :, i],
                    np.array(data["G_orig"])[:, orig_index].ravel(),
                    mean_squared_error,
                )
                coeff_mae = cal_metrics(
                    q[0, :, i],
                    np.array(data["G_orig"])[:, orig_index].ravel(),
                    mean_absolute_error,
                )

                
                ylabel = (
                    str(np.array(data["labels"])[i])
                    + "\nPearson's r:"
                    + "{:.4f}".format(coeff_pear[0])
                    + "\nRMSE:"
                    + "{:.4f}".format(coeff_rmse[0])
                    + "\nMAE:"
                    + "{:.4f}".format(np.mean(coeff_mae))
                )
                
                df.iloc[:,i] = [np.mean(coeff_pear),np.mean(coeff_rmse),np.mean(coeff_mae),np.std(coeff_pear),np.std(coeff_rmse),np.std(coeff_mae)]

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            ylabel,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks



        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()

    return fig,df


def mass_conservation(data, sample=-1):

    total_orig = data["X"].sum(axis=-1)
    if isinstance(sample, int):
        # total = data["G"].sum(axis=-1)[sample, ...]
        total = data["G"][sample, :].sum(axis=-1)
        title = "Mass conservation from last sample"
    elif sample == "median":
        # total = np.median(data["G"], axis=0).sum(axis=-1)
        total = np.sum(data["G"], axis=-1).median(axis=0)
        # np.median(data["G"].sum(axis=-1), axis=0)
        title = "Mass conservation from median G value"

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title(title)

    axes[0].scatter(total_orig, total, s=20)
    axes[1].scatter(total_orig, total / total_orig, s=20)
    axes[2].scatter(total_orig, total / total_orig, s=20)
    axes[0].axline((0, 0), slope=1., color='r', label='by slope')
    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.8)
    axes[2].axhline(y=1, color="r", linestyle="--", alpha=0.8)

    axes[2].set_xscale("log")
    axes[0].set_ylabel("Sum(G)", fontsize=20)
    axes[1].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_xlabel("Sum(Input)", fontsize=20)
    for i in range(3):
        axes[i].tick_params(axis="y", labelsize=18)
        axes[i].tick_params(axis="x", labelsize=18)

    return fig


def cal_errorbar(data, q=[0.25, 0.5, 0.75]):

    quants = np.quantile(data, q, axis=0)
    return quants


def mass_conservation_comparison(data1, data2, sample):

    total_orig = data1["X"].sum(axis=-1)
    if isinstance(sample, int):
        # total = data["G"].sum(axis=-1)[sample, ...]
        total = data1["G"][sample, :].sum(axis=-1)
        total2 = data2["G"][sample, :].sum(axis=-1)
        title = "Mass conservation from the Nr. " + str(sample) + " sample"
    elif sample == "median":
        total = np.median(data1["G"], axis=0).sum(axis=-1)
        total2 = np.median(data2["G"], axis=0).sum(axis=-1)
        # np.median(data["G"].sum(axis=-1), axis=0)
        title = "Mass conservation from median G value"

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title(title)

    axes[0].scatter(total_orig, total, s=20, color="blue", label="BAMF")
    axes[1].scatter(total_orig, total / total_orig, s=20, color="blue", label="BAMF")
    axes[2].scatter(total_orig, total / total_orig, s=20, color="blue", label="BAMF")

    axes[0].scatter(total_orig, total2, s=20, color="green", label="PMF")
    axes[1].scatter(total_orig, total2 / total_orig, s=20, color="green", label="PMF")
    axes[2].scatter(total_orig, total2 / total_orig, s=20, color="green", label="PMF")

    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.8)
    axes[2].axhline(y=1, color="r", linestyle="--", alpha=0.8)

    axes[2].set_xscale("log")
    axes[0].set_ylabel("Sum(G)", fontsize=20)
    axes[1].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_xlabel("Sum(Input)", fontsize=20)
    for i in range(3):
        axes[i].legend(loc="upper right", fontsize=18)
        axes[i].tick_params(axis="y", labelsize=18)
        axes[i].tick_params(axis="x", labelsize=18)

    return fig


def errorbar_batched_comparison(
    data_bamf,
    data_pmf,
    labels,
    title,
    n_rows=5,
    xlabel="",
    palette=None,
    fontsize=22,
    label="factor",
):

    fig, axes = plt.subplots(n_rows, sharex=True)

    if palette is None:
        palette = sns.color_palette(n_colors=len(axes) * 2)

    axes[0].set_title(title)

    if label == "factor":
        quants_bamf = np.quantile(data_bamf["F"], q=[0.25, 0.5, 0.75], axis=0)
        quants_pmf = np.quantile(data_pmf["F"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                data_bamf["variable"],
                quants_bamf[1, i, :],
                np.abs(quants_bamf[[0, 2], i, :] - quants_bamf[1, i, :]),
                color=palette[i],
                marker="o",
                markeredgecolor=palette[i],
                label="BAMF",
                capthick=1,
                alpha=0.75,
                linewidth=0,
                elinewidth=1,
                ecolor=palette[i],
                capsize=3,
                markersize=5,
            )
            bp = axes[i].errorbar(
                data_pmf["variable"],
                quants_pmf[1, i, :],
                np.abs(quants_pmf[[0, 2], i, :] - quants_pmf[1, i, :]),
                color=palette[i + len(axes)],
                marker="o",
                markeredgecolor=palette[i + len(axes)],
                capthick=1,
                alpha=0.75,
                linewidth=0,
                label="PMF",
                elinewidth=1,
                ecolor=palette[i + len(axes)],
                capsize=3,
                markersize=5,
            )
            #             plt.xlabel(xlabel)
            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend(loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)

        plt.xticks(rotation=90)
    elif label == "time":
        quants_bamf = np.quantile(data_bamf["G"], q=[0.25, 0.5, 0.75], axis=0)
        quants_pmf = np.quantile(data_pmf["G"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):
            # plot model results
            x.plot(
                data_bamf["timestep"],
                quants_bamf[1, :, i],
                "x",
                color=palette[i],
                label="BAMF",
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data_bamf["timestep"]),
                quants_bamf[0, :, i],
                quants_bamf[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )

            x.plot(
                data_pmf["timestep"],
                quants_pmf[1, :, i],
                "x",
                color=palette[i + len(axes)],
                label="PMF",
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data_pmf["timestep"]),
                quants_pmf[0, :, i],
                quants_pmf[-1, :, i],
                alpha=0.5,
                color=palette[i + len(axes)],
            )

            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend(loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)
    return fig, axes


def plot_F_comparison(data_bamf, data_pmf, benchmark=False, extra_data=None):

    fig, axes = errorbar_batched_comparison(
        data_bamf,
        data_pmf,
        labels=np.array(data_bamf["labels"]),
        title="F",
        n_rows=data_bamf["G"].shape[-1],
        label="factor",
    )

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_pmf["labels"])[i])
        if benchmark and "F_orig" in data_bamf.keys():
            if np.array(data_bamf["labels"])[i] in np.array(data_bamf["orig_labels"]):
                orig_index = np.where(
                    np.array(data_bamf["orig_labels"])
                    == np.array(data_bamf["labels"])[i]
                )[0]

                x.bar(
                    data_bamf["variable"],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    width=0.5,
                    color="k",
                    alpha=0.6,
                    label="reference",
                )

                coeffs_spear_pmf = cal_metrics(
                    np.array(data_pmf["F"])[:, i, :],
                    np.array(data_pmf["F_orig"])[orig_index, :].ravel(),
                    spearmanr,
                )

                coeffs_spear_bamf = cal_metrics(
                    np.array(data_bamf["F"])[:, i, :],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    spearmanr,
                )

                coeff_rmse_pmf = cal_metrics(
                    np.array(data_pmf["F"])[:, i, :],
                    np.array(data_pmf["F_orig"])[orig_index, :].ravel(),
                    mean_squared_error,
                )

                coeff_rmse_bamf = cal_metrics(
                    np.array(data_bamf["F"])[:, i, :],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    mean_squared_error,
                )
                coeff_mae_pmf = cal_metrics(
                    np.array(data_pmf["F"])[:, i, :],
                    np.array(data_pmf["F_orig"])[orig_index, :].ravel(),
                    mean_absolute_error,
                )
                coeff_mae_bamf = cal_metrics(
                    np.array(data_bamf["F"])[:, i, :],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    mean_absolute_error,
                )


                ylabel = (
                    str(np.array(data_pmf["labels"])[i])
                    + "\nBAMF_Spearman's ρ:"
                    + "{:.4f}".format(np.mean(coeffs_spear_bamf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs_spear_bamf))
                    + "\nBAMF_RMSE:"
                    + "{:.4f}".format(np.mean(coeff_rmse_bamf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse_bamf))
                    + "\nBAMF_MAE:"
                    + "{:.4f}".format(np.mean(coeff_mae_bamf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_mae_bamf))
                    + "\nPMF_Spearman's ρ:"
                    + "{:.4f}".format(np.mean(coeffs_spear_pmf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs_spear_pmf))
                    + "\nPMF_RMSE:"
                    + "{:.4f}".format(np.mean(coeff_rmse_pmf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse_pmf))
                    + "\nPMF_MAE:"
                    + "{:.4f}".format(np.mean(coeff_mae_pmf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_mae_pmf))                   
                )
        # set the subplot's y label
        twin = axes[i].twinx()

        twin.set_ylabel(
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )

        twin.get_yaxis().set_ticks([])  # hiding the ticks
        x.set_xticks(data_bamf["variable"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)
    return fig


def plot_G_comparison(
    data_bamf, data_pmf, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]
):

    fig, axes = errorbar_batched_comparison(
        data_bamf,
        data_pmf,
        labels=np.array(data_bamf["labels"]),
        title=title,
        n_rows=data_bamf["G"].shape[-1],
        label="time",
    )
    # plot benchmark

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_pmf["labels"])[i])
        if benchmark and np.array(data_pmf["labels"])[i] in np.array(
            data_pmf["orig_labels"]
        ):

            orig_index = np.where(
                np.array(data_bamf["orig_labels"]) == np.array(data_bamf["labels"])[i]
            )[0]

            x.plot(
                data_bamf["timestep"],
                data_bamf["G_orig"][:, orig_index],
                "k--",
                label="Truth",
            )

            # pearson coeff notation
            coeffs_pear_pmf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )
            coeffs_pear_bamf = cal_metrics(
                np.array(data_bamf["G"])[:, :, i],
                np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )
            coeff_rmse_pmf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )
            coeff_rmse_bamf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )
            coeff_mae_pmf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_absolute_error,
                )
            coeff_mae_bamf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_absolute_error,
                )

            ylabel = (
                str(np.array(data_pmf["labels"])[i])
                + "\nBAMF_Spearman's ρ:"
                + "{:.4f}".format(np.mean(coeffs_pear_bamf))
                + " +/- "
                + "{:.4f}".format(np.std(coeffs_pear_bamf))
                + "\nBAMF_RMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse_bamf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse_bamf))
                + "\nBAMF_MAE:"
                + "{:.4f}".format(np.mean(coeff_mae_bamf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_mae_bamf))
                + "\nPMF_Spearman's ρ:"
                + "{:.4f}".format(np.mean(coeffs_pear_pmf))
                + " +/- "
                + "{:.4f}".format(np.std(coeffs_pear_pmf))
                + "\nPMF_RMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse_pmf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse_pmf))
                + "\nPMF_MAE:"
                + "{:.4f}".format(np.mean(coeff_mae_pmf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_mae_pmf))      
                    )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            ylabel,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    fig.autofmt_xdate()
    return fig


def plot_median_G_comparison(
    data_bamf, data_pmf, title="G_median", benchmark=False, quantiles=[0.5]
):

    q_bamf = np.quantile(data_bamf["G"], quantiles, axis=0)
    q_pmf = np.quantile(data_pmf["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data_bamf["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes) * 2)

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_bamf["labels"])[i])
        x.plot(
            data_bamf["timestep"],
            q_bamf[0, :, i],
            "o",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
            label="BAMF",
        )
        x.plot(
            data_pmf["timestep"],
            q_pmf[0, :, i],
            "x",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i + len(axes)],
            label="PMF",
        )

        if benchmark:
            if np.array(data_bamf["labels"])[i] in np.array(data_bamf["orig_labels"]):
                orig_index = np.where(
                    np.array(data_bamf["orig_labels"])
                    == np.array(data_bamf["labels"])[i]
                )[0]
                x.plot(
                    data_bamf["timestep"],
                    data_bamf["G_orig"][:, orig_index],
                    "k--",
                    label="Truth",
                )

                coeff_pearson_bamf = cal_metrics(
                    
                    q_bamf[0, :, i],
                    np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                    pearsonr,
                )
                coeff_pearson_pmf = cal_metrics(
                    
                    q_pmf[0, :, i],
                    np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                    pearsonr,
                )

                coeff_mse_bamf = cal_metrics(
                    q_bamf[0, :, i],
                    np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                    mean_squared_error,
                )
                coeff_mse_pmf = cal_metrics(
                    q_pmf[0, :, i],
                    np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                    mean_squared_error,
                )
                ylabel = (
                    str(np.array(data_bamf["labels"])[i])
                    + "\nBAMF_Pearson's r: "
                    + "{:.4f}".format(np.mean(coeff_pearson_bamf))
                    + "\nBAMF_RMSE: "
                    + "{:.4f}".format(np.mean(coeff_mse_bamf))
                    + "\nPMF_Pearson's r: "
                    + "{:.4f}".format(np.mean(coeff_pearson_pmf))
                    + "\nPMF_RMSE: "
                    + "{:.4f}".format(np.mean(coeff_mse_pmf))
                )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            ylabel,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()

    return fig
