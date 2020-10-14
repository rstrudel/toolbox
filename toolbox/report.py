import os

import click
import numpy as np
import yaml
from termcolor import colored


from toolbox.logs_util import read_tensorboard
from toolbox.plot import plot
from toolbox.settings import BASE_DIR


def load_runs(logs_paths, keys, stats_key, filters_exp):
    logs = read_tensorboard(logs_paths, keys, stats_key, filters_exp)
    return logs


def print_color(s, color):
    print(colored(s, color))


def report_values(log_key, logs, factor=1, max_value=False):
    print_color(log_key, "blue")
    for k, v in logs.items():
        if max_value:
            idx = v["values"][:, 0].argmax()
        else:
            idx = -1
        v_last = v["values"][idx, 0]
        if "Accuracy" in log_key or "Success" in log_key:
            v_last *= 100
        print_color(
            "{}: {:.2f}".format(k, v_last), "green",
        )


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("--stats-key", "-sk", type=str, default="/seed")
def main(experiment, stats_key):
    plot_dict = yaml.load(
        open(os.path.join(BASE_DIR, "experiments", "plot.yml"), "r"),
        Loader=yaml.FullLoader,
    )
    exp_dict = yaml.load(
        open(os.path.join(BASE_DIR, "experiments", experiment), "r"),
        Loader=yaml.FullLoader,
    )
    savedir = plot_dict.pop("savedir")
    for exp_paths, plot_name, labels, filters_exp in zip(
        exp_dict["paths"],
        exp_dict["plot_name"],
        exp_dict["labels"],
        exp_dict["filters"],
    ):
        if not isinstance(exp_paths, list):
            raise ValueError(
                "paths to experiments should be a list, not a single string. It should be of the form:\n"
                "paths:\n"
                "- - /path_to_experiment"
            )
        print("Processing {} located in {} ...".format(plot_name, exp_paths))
        log_keys = list(exp_dict["log_keys"].keys()) + ["trainer/epoch"]
        logs = load_runs(exp_paths, log_keys, stats_key, filters_exp)

        epochs = logs.pop("trainer/epoch")
        for log_key, log_prop in exp_dict["log_keys"].items():
            log_plot_dict = plot_dict.copy()
            log_plot_dict["labels"] = labels
            if log_prop["label"]:
                log_plot_dict["ylabel"] = log_prop["label"]
            else:
                log_plot_dict["ylabel"] = log_key
            log_plot_dict["output"] = os.path.join(
                savedir, "{}_{}.png".format(plot_name, log_prop["filename"])
            )
            log_plot_dict["xsteps"] = epochs
            log_plot_dict["logs"] = logs[log_key]
            factor = 1
            if "Success" in log_key or "Accuracy" in log_key:
                factor = 100
            max_value = False
            report_values(log_key, logs[log_key], factor, max_value)
            if log_prop["kwargs"]:
                log_plot_dict.update(log_prop["kwargs"])
            plot(**log_plot_dict)
        print("Plots saved in {}/{}_*.png".format(savedir, plot_name))


if __name__ == "__main__":
    main()
