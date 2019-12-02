import os
import glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard(paths, num_scalars, filter_prefix, filter_keys):
    log_files = []
    for path in paths:
        log_files += glob.glob(os.path.join(path, '**/*tfevents*'),
                               recursive=True)
    log_files = [log_file for log_file in log_files if filter_prefix in log_file]

    logs = {}
    counter = {}
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': num_scalars,
        'histograms': 1
    }
    for log_file in log_files:
        log = {}
        event_acc = event_accumulator.EventAccumulator(log_file,
                                                       tf_size_guidance)
        event_acc.Reload()

        tb_to_log_keys = {
            'wall_time': 'timestamps',
            'step': 'steps',
            'value': 'values'
        }
        for filter_key in filter_keys:
            log[filter_key] = {'timestamps': [], 'steps': [], 'values': []}
            scalar_logs = event_acc.Scalars(filter_key)
            for scalar_log in scalar_logs:
                for tb_log_key, log_key in tb_to_log_keys.items():
                    log[filter_key][log_key].append(
                        getattr(scalar_log, tb_log_key))
            for k, v in log[filter_key].items():
                log[filter_key][k] = np.array(v)

        for path in paths:
            if path in log_file:
                # TODO: change, not robust
                exp_name = os.path.split(path[:-1])[-1]
                variant = log_file.replace(path, '')[:-1]
                # TODO: find a better alternative, depends on os
                variant = variant.split('/')[0]
                log_name = '{}/{}'.format(exp_name, variant)
        logs[log_name] = log

    return logs
