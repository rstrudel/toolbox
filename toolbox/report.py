import click
import yaml
import os
import numpy as np

from toolbox.plot import plot
from toolbox.settings import BASE_DIR
from toolbox.logs_util import read_tensorboard


def map_nested_dicts(ob, key, func, func_filt):
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, k, func, func_filt) for k, v in ob.items()}
    else:
        return func(ob) if func_filt(key) else ob


def load_runs(logs_paths, num_values, prefix, keys, stats_key):
    steps = np.arange(num_values)
    logs = read_tensorboard(logs_paths, num_values, prefix, keys, stats_key)
    keys = list(logs.keys())
    # if vmin or vmax:
    #     logs = map_nested_dicts(
    #         logs, "", lambda x: np.clip(x, vmin, vmax), lambda k: "values" in k
    #     )
    return logs


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
    for exp_paths, plot_name, labels in zip(
        exp_dict["paths"], exp_dict["plot_name"], exp_dict["labels"]
    ):
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
            key_plot_dict["logs"] = load_runs(
                exp_paths,
                num_values,
                key["prefix"],
                key["key"],
                stats_key,
            )
            if key["kwargs"]:
                key_plot_dict.update(key["kwargs"])
            plot(**key_plot_dict)
        print("Plots saved in {}/{}_*.png".format(savedir, plot_name))


if __name__ == "__main__":
    main()
