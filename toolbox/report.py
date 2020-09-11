import os

import click
import numpy as np
import yaml
from termcolor import colored


from toolbox.logs_util import read_tensorboard
from toolbox.plot import plot
from toolbox.settings import BASE_DIR


def map_nested_dicts(ob, key, func, func_filt):
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, k, func, func_filt) for k, v in ob.items()}
    else:
        return func(ob) if func_filt(key) else ob


def load_runs(logs_paths, num_values, prefix, keys, stats_key, filters_exp):
    steps = np.arange(num_values)
    logs = read_tensorboard(
        logs_paths, num_values, prefix, keys, stats_key, filters_exp
    )
    return logs


def print_color(s, color):
    print(colored(s, color))


def report_values(logs, factor=1, max_value=False):
    for k, v in logs.items():
        if max_value:
            idx = v["values"][:, 0].argmax()
        else:
            idx = -1
        v_mean, v_min, v_max = v["values"][idx]
        print_color(k, "blue")
        print_color(
            "{:.2f} - {:.2f}(-) {:.2f}(+)".format(
                factor * v_mean, factor * v_min, factor * v_max
            ),
            "green",
        )


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("--num-values", "-nv", type=int, default=500)
@click.option("--stats-key", "-sk", type=str, default="_seed")
def main(experiment, num_values, stats_key):
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
        for key in exp_dict["keys"]:
            print("{}/{}".format(key["prefix"], key["key"]))
            key_plot_dict = plot_dict.copy()
            key_plot_dict["labels"] = labels
            if key["key_label"]:
                key_plot_dict["ylabel"] = key["key_label"]
            else:
                key_plot_dict["ylabel"] = key["key"]
            key_plot_dict["output"] = os.path.join(
                savedir, "{}_{}.png".format(plot_name, key["filename"])
            )
            logs = load_runs(
                exp_paths, num_values, key["prefix"], key["key"], stats_key, filters_exp
            )
            key_plot_dict["logs"] = logs
            factor = 1
            max_value = False
            if "Success" in key["key"]:
                factor = 100
                max_value = True
            if "Success" in key["key"] or "Loss" in key["key"]:
                report_values(logs, factor, max_value)
            if key["kwargs"]:
                key_plot_dict.update(key["kwargs"])
            plot(**key_plot_dict)
        print("Plots saved in {}/{}_*.png".format(savedir, plot_name))


if __name__ == "__main__":
    main()
