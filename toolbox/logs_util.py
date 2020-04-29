import glob
import os
from itertools import groupby

import numpy as np

from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard(
    paths, num_scalars, filter_prefix, filter_key, stats_key=None, filters_exp=None
):

    counter = {}
    tf_size_guidance = {
        "compressedHistograms": 10,
        "images": 0,
        "scalars": num_scalars,
        "histograms": 1,
    }

    # gather all the logfiles in the subdirectories
    log_files_raw = []
    for path in paths:
        tb_files = glob.glob(os.path.join(path, "**/*tfevents*"), recursive=True)
        tb_files = [(tb_file, path) for tb_file in tb_files]
        log_files_raw += tb_files

    # filter and keep log files related to a given log key or exp name
    if filters_exp is None:
        filters_exp = []
    log_files = []
    for log_file in log_files_raw:
        has_prefix = filter_prefix in log_file[0]
        has_filter = len(filters_exp) == 0
        for filter_exp in filters_exp:
            if filter_exp in log_file[0]:
                has_filter = True
        if has_prefix and has_filter:
            log_files.append(log_file)

    log_files.sort(key=lambda x: x[0])

    logs = {}
    for log_file, path in log_files:
        log = {}
        event_acc = event_accumulator.EventAccumulator(log_file, tf_size_guidance)
        event_acc.Reload()

        tb_to_log_keys = {"wall_time": "timestamps", "step": "steps", "value": "values"}
        log = {"timestamps": [], "steps": [], "values": []}
        tags = event_acc.Tags()
        if filter_key not in tags["scalars"]:
            raise ValueError(
                "{} is not in the tensorboard logs {}. Available scalars are: {}".format(
                    filter_key, log_file, tags["scalars"]
                )
            )
        scalar_logs = event_acc.Scalars(filter_key)
        for scalar_log in scalar_logs:
            for tb_log_key, log_key in tb_to_log_keys.items():
                log[log_key].append(getattr(scalar_log, tb_log_key))
        for k, v in log.items():
            log[k] = np.array(v)

        exp_name = path
        if exp_name[-1] == "/":
            exp_name = exp_name[:-1]
        exp_name = exp_name.split("/")[-1]
        variant = log_file[len(path) :].split("/")[0]
        log_name = "{}/{}".format(exp_name, variant)
        logs[log_name] = log

    logs = compute_statistics(logs, num_scalars, stats_key)

    return logs


def compute_statistics(logs, num_scalars, stats_key):
    if stats_key is None:
        for k in logs.keys():
            values = logs[k]["values"]
            logs[k]["values"] = np.concatenate(
                (values[:, None], np.zeros_like(values)[:, None]), axis=1
            )
        return logs

    groups = [list(i) for j, i in groupby(logs.keys(), lambda a: a.split(stats_key)[0])]
    new_logs = {}
    # compute statistics by prefix of stats_key group
    for group in groups:
        # compute statistics only until the experiment with smallest
        # number of steps
        steps_max = []
        group_steps = []
        group_values = []
        for log_name in group:
            new_log_name = log_name.split(stats_key)[0]
            group_steps.append(logs[log_name]["steps"])
            group_values.append(logs[log_name]["values"])
        steps_min = max([s.min() for s in group_steps])
        steps_max = min([s.max() for s in group_steps])
        steps = np.linspace(steps_min, steps_max, num_scalars)
        # timestamps = np.interp(steps, steps_max, num_scalars)
        for i in range(len(group_values)):
            group_values[i] = np.interp(steps, group_steps[i], group_values[i])
        group_values = np.stack(group_values)

        new_logs[new_log_name] = {}
        new_logs[new_log_name]["steps"] = steps
        new_logs[new_log_name]["timestamps"] = logs[log_name]["timestamps"]
        new_logs[new_log_name]["values"] = np.concatenate(
            (group_values.mean(axis=0)[:, None], group_values.std(axis=0)[:, None]),
            axis=1,
        )
    return new_logs
