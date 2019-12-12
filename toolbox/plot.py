import click
import logging
import os
import collections

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox.plot_util import Lines
from toolbox.logs_util import read_tensorboard


def map_nested_dicts(ob, key, func, func_filt):
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, k, func, func_filt) for k, v in ob.items()}
    else:
        return func(ob) if func_filt(key) else ob


def load_runs(logs_paths, num_values, prefix, keys, stats_key, vmin, vmax):
    steps = np.arange(num_values)
    logs = read_tensorboard(logs_paths, num_values, prefix, keys, stats_key)
    keys = list(logs.keys())
    # for key in keys:
    #     if "frac0.8" in key or "256" in key or "1024" in key or "frac0.5" in key:
    #         logs.pop(key)
    if vmin or vmax:
        logs = map_nested_dicts(
            logs, "", lambda x: np.clip(x, vmin, vmax), lambda k: "values" in k
        )
    return logs


def plot(
    logs_paths,
    prefix,
    key,
    output,
    padding,
    start,
    limit,
    num_values,
    xaxis,
    xscales,
    yscale,
    smooth,
    stats_key,
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
    sort,
):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if not ylabel:
        ylabel = key
    ax.set_ylabel(ylabel)
    domains = []
    lines = []
    logs = load_runs(logs_paths, num_values, prefix, key, stats_key, vmin, vmax)
    for log_name, log in logs.items():
        steps, timestamps, values = log["steps"], log["timestamps"], log["values"]
        logging.info("\nLine {}".format(log_name))
        timestamps -= timestamps[0]
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
        if xscales:
            steps /= xscales[index]
    plot_lines = Lines(resolution=resolution, smooth=smooth)
    plot_lines.LEGEND["loc"] = legend
    plot_lines.LEGEND["bbox_to_anchor"] = (
        bbox_to_anchor[0],
        bbox_to_anchor[1] * (len(logs) + 1),
    )
    plot_lines(ax, domains, lines, logs.keys())
    ax.grid(True, alpha=grid)
    if logx:
        ax.set_xscale("log", nonposx="clip")
    if logy:
        ax.set_yscale("log", nonposy="clip")
        ax.set_xlim(min(min(x) for x in domains), max(max(x) for x in domains))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if sci_notation:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        if title:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            fig.tight_layout()
    img_path = output
    fig.savefig(
        img_path, bbox_inches="tight", pad_inches=padding, transparent=False, dpi=300
    )


@click.command(help="plot logs with matplotlib")
@click.argument("logs", nargs=-1, type=str)
@click.option("-k", "--key", type=str, required=True)
@click.option("-o", "--output", type=str, help="output path", required=True)
@click.option("-pad", "--padding", type=float, default=0.1)
# Data
# names and xscales should have multiple args
@click.option("-s", "--start", default=None, type=int)
@click.option("-l", "--limit", default=None, type=int)
@click.option("-nv", "--num-values", default=10000, type=int)
@click.option("--xaxis", default="steps")  # choices=['steps', 'time'])
@click.option("--xscales", type=float)
@click.option("--yscale", default=1, type=float)
@click.option("--smooth", default=0, type=float)
@click.option("--stats-key", default="")
@click.option("--vmin", default=None, type=float)
@click.option("--vmax", default=None, type=float)
# Labels
# @click.option('--figsize', default=(3, 2.5))  # Paper
@click.option("--figsize", default=(4, 3))  # Blog
@click.option("--title", default=None)
@click.option("--xlabel", default=None)
@click.option("--ylabel", default=None)
@click.option("--sci-notation/--no-sci-notation", default=False, is_flag=True)
# Plotting
@click.option("--legend", default="best")
@click.option("--bbox-to-anchor", default=(0.0, 0.0))  # Blog
@click.option("--resolution", default=50, type=int)
@click.option("--grid", default=1)
@click.option("--logx/--no-logx", default=False, is_flag=True)
@click.option("--logy/--no-logy", default=False, is_flag=True)
@click.option("--sort/--no-sort", default=False, is_flag=True)
def main(
    logs,
    key,
    output,
    padding,
    start,
    limit,
    num_values,
    xaxis,
    xscales,
    yscale,
    smooth,
    stats_key,
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
    sort,
):
    # if xscales:
    #     assert len(names) == len(xscales)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    plot(
        logs,
        key,
        output,
        padding,
        start,
        limit,
        num_values,
        xaxis,
        xscales,
        yscale,
        smooth,
        stats_key,
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
        sort,
    )


if __name__ == "__main__":
    main()
