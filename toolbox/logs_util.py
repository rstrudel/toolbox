import glob
import os
from itertools import groupby
from pathlib import Path
from collections import defaultdict

import numpy as np

from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard(paths, log_keys, stats_key=None, filters_exp=None):
    num_scalars = 1000
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
        has_filter = len(filters_exp) == 0
        for filter_exp in filters_exp:
            if filter_exp in log_file[0]:
                has_filter = True
        if has_filter:
            log_files.append(log_file)

    log_files.sort(key=lambda x: x[0])

    logs = {}
    for log_file, root_path in log_files:
        logs[log_file] = {}
        event_acc = event_accumulator.EventAccumulator(log_file, tf_size_guidance)
        event_acc.Reload()

        tb_to_log_keys = {"wall_time": "timestamps", "step": "steps", "value": "values"}
        tags = event_acc.Tags()
        for log_key in log_keys:
            if log_key not in tags["scalars"]:
                print(f"WARNING: did not find {log_key} in logs.")
                continue
            scalar_logs = event_acc.Scalars(log_key)
            logs[log_file][log_key] = {"timestamps": [], "steps": [], "values": []}
            for scalar_log in scalar_logs:
                for tb_log_key, user_log_key in tb_to_log_keys.items():
                    logs[log_file][log_key][user_log_key].append(
                        getattr(scalar_log, tb_log_key)
                    )
            for k, v in logs[log_file][log_key].items():
                logs[log_file][log_key][k] = np.array(v).reshape(-1, 1)
        logs[log_file]["root"] = root_path

    logs_flattened = defaultdict(lambda: {})
    for exp_path, exp_log in logs.items():
        for log_key, log in exp_log.items():
            if log_key == "root":
                continue
            rp = Path(exp_log["root"])
            p = Path(exp_path)
            exp_name = p.parent.name
            if exp_name != rp.name:
                name = f"{rp.name}/{exp_name}"
            logs_flattened[log_key][name] = log

    # compute mean over different workers
    # worker_key = "/rank"
    # grouped_logs = group_by_key(logs, worker_key)
    # logs = align_logs_by_group(grouped_logs, num_scalars)
    # average_values(logs)

    # compute stats over different seeds
    # grouped_logs = group_by_key(logs, stats_key)
    # logs = align_logs_by_group(grouped_logs, num_scalars)
    # statistics_values(logs)

    return logs_flattened


def group_by_key(d, group_key):
    d_grouped = {}
    groups = [list(i) for j, i in groupby(d.keys(), lambda a: a.split(group_key)[0])]
    for group in groups:
        group_name = group[0].split(group_key)[0]
        for log_name in group:
            if group_name not in d_grouped:
                d_grouped[group_name] = []
            d_grouped[group_name].append(d[log_name])
    return d_grouped


def align_logs_by_group(grouped_logs, num_scalars):
    aligned_logs = {}
    for group_name, group in grouped_logs.items():
        steps_max = []
        group_steps = []
        group_values = []
        for log in group:
            group_steps.append(log["steps"])
            group_values.append(log["values"])
        steps_min = max([s.min() for s in group_steps])
        steps_max = min([s.max() for s in group_steps])
        steps = np.linspace(steps_min, steps_max, num_scalars)
        for i in range(len(group_values)):
            group_values[i] = np.interp(steps, group_steps[i], group_values[i])
        group_values = np.stack(group_values)

        aligned_logs[group_name] = {}
        aligned_logs[group_name]["steps"] = steps
        aligned_logs[group_name]["timestamps"] = log["timestamps"]
        aligned_logs[group_name]["values"] = group_values
    return aligned_logs


def average_values(grouped_logs):
    for group_name, group in grouped_logs.items():
        v_mean = group["values"].mean(axis=0)
        grouped_logs[group_name]["values"] = v_mean


# def statistics_values(grouped_logs):
#     for group_name, group in grouped_logs.items():
#         v_mean = group["values"].mean(axis=0)[:, None]
#         v_min = group["values"].min(axis=0)[:, None]
#         v_max = group["values"].max(axis=0)[:, None]
#         grouped_logs[group_name]["values"] = np.hstack((v_mean, v_min, v_max))
