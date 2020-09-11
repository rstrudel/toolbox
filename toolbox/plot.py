import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from toolbox.lines import Lines

matplotlib.use("Agg")


def plot(
    logs,
    labels,
    output,
    padding,
    start,
    limit,
    xaxis,
    xscale,
    yscale,
    smooth,
    xmin,
    xmax,
    vmin,
    vmax,
    figsize,
    title,
    xlabel,
    ylabel,
    sci_notation,
    legend,
    bbox_to_anchor,
    resolution,
    grid,
    logx,
    logy,
):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    domains = []
    lines = []
    for log_name, log in logs.items():
        steps, timestamps, values = log["steps"], log["timestamps"], log["values"]
        if timestamps is None:
            timestamps = np.zeros(2)
            assert xaxis != "time"
        logging.info("\nLine {}".format(log_name))
        timestamps -= timestamps[0]
        steps *= xscale
        values *= yscale
        if start:
            start_idx = np.searchsorted(steps, start, "left")
            steps = steps[start_idx:]
            timestamps = timestamps[start_idx:]
            values = values[start:]
        if limit:
            limit_idx = np.searchsorted(steps, limit, "left")
            steps = steps[:limit_idx]
            timestamps = timestamps[:limit_idx]
            values = values[:limit_idx]
        if xaxis == "time":
            domains.append(timestamps / 3600)
        elif xaxis == "steps":
            domains.append(steps)
        else:
            raise ValueError("xaxis should be in ('time', 'steps')")
        lines.append(values)
    if not domains:
        print("No experiments to plot.")
        return

    plot_lines = Lines(resolution=resolution, smooth=smooth)
    plot_lines.LEGEND["loc"] = legend["loc"]
    plot_lines.LEGEND["fontsize"] = legend["fontsize"]
    plot_lines.LEGEND["bbox_to_anchor"] = (
        -0.4,
        -0.2 - 0.03 * len(logs),
    )
    # plot_lines.LEGEND["bbox_to_anchor"] = (
    #     bbox_to_anchor[0],
    #     bbox_to_anchor[1] * (len(logs) + 1),
    # )
    if labels:
        labels_logs = []
        for log_key in logs.keys():
            for label_log, label_plot in labels.items():
                if label_log in log_key:
                    labels_logs.append(label_plot)
    else:
        labels_logs = list(logs.keys())
    plot_lines(ax, domains, lines, labels_logs)
    ax.grid(True, alpha=grid)
    if logx:
        ax.set_xscale("log", nonposx="clip")
    if logy:
        ax.set_yscale("log", nonposy="clip")

    if xmin is None:
        xmin = min(min(x) for x in domains)
    if xmax is None:
        xmax = max(max(x) for x in domains)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(vmin, vmax)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(plt.LinearLocator(numticks=7))

    if sci_notation:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # if title:
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # else:
    #     fig.tight_layout()
    img_path = output
    fig.savefig(
        img_path, bbox_inches="tight", pad_inches=padding, transparent=False, dpi=300
    )
    plt.close(fig)


# @click.command(help="plot logs with matplotlib")
# @click.argument("logs", nargs=-1, type=str)
# @click.option("-k", "--key", type=str, required=True)
# @click.option("-o", "--output", type=str, help="output path", required=True)
# @click.option("-pad", "--padding", type=float, default=0.1)
# # Data
# # names and xscales should have multiple args
# @click.option("-s", "--start", default=None, type=int)
# @click.option("-l", "--limit", default=None, type=int)
# @click.option("-nv", "--num-values", default=10000, type=int)
# @click.option("--xaxis", default="steps")  # choices=['steps', 'time'])
# @click.option("--xscale", default=1, type=float)
# @click.option("--yscale", default=1, type=float)
# @click.option("--smooth", default=0, type=float)
# @click.option("--stats-key", default="")
# @click.option("--xmin", default=None, type=float)
# @click.option("--xmax", default=None, type=float)
# @click.option("--vmin", default=None, type=float)
# @click.option("--vmax", default=None, type=float)
# # Labels
# # @click.option('--figsize', default=(3, 2.5))  # Paper
# @click.option("--figsize", default=(4, 3))  # Blog
# @click.option("--title", default=None)
# @click.option("--xlabel", default=None)
# @click.option("--ylabel", default=None)
# @click.option("--sci-notation/--no-sci-notation", default=False, is_flag=True)
# # Plotting
# @click.option("--legend-loc", default="best")
# @click.option("--legend-fontsize", default="middle")
# @click.option("--bbox-to-anchor", default=(0.0, 0.0))  # Blog
# @click.option("--resolution", default=50, type=int)
# @click.option("--grid", default=1)
# @click.option("--logx/--no-logx", default=False, is_flag=True)
# @click.option("--logy/--no-logy", default=False, is_flag=True)
# @click.option("--sort/--no-sort", default=False, is_flag=True)
# def main(
#     logs,
#     key,
#     output,
#     padding,
#     start,
#     limit,
#     num_values,
#     xaxis,
#     xscales,
#     yscale,
#     smooth,
#     stats_key,
#     xmin,
#     xmax,
#     vmin,
#     vmax,
#     figsize,
#     title,
#     xlabel,
#     ylabel,
#     sci_notation,
#     legend_loc,
#     legend_fontsize,
#     bbox_to_anchor,
#     resolution,
#     grid,
#     logx,
#     logy,
#     sort,
# ):
#     legend = {"loc": legend_loc, "fontsize": legend_fontsize}
#     # if xscales:
#     #     assert len(names) == len(xscales)
#     logging.basicConfig(format="%(message)s", level=logging.INFO)
#     plot(
#         logs,
#         key,
#         output,
#         padding,
#         start,
#         limit,
#         num_values,
#         xaxis,
#         xscales,
#         yscale,
#         smooth,
#         stats_key,
#         xmin,
#         xmax,
#         vmin,
#         vmax,
#         figsize,
#         title,
#         xlabel,
#         ylabel,
#         sci_notation,
#         legend,
#         bbox_to_anchor,
#         resolution,
#         grid,
#         logx,
#         logy,
#         sort,
#     )


# if __name__ == "__main__":
#     main()
