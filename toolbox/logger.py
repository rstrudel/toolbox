import os
import datetime
import dateutil.tz
import yaml
import errno
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter


def mkdir_p(path, erase_path):
    try:
        os.makedirs(path)
    except OSError as exc:
        raise ValueError("Path {} already exists.".format(path))


def setup_logger(
    exp_prefix,
    variant,
    variant_log_file="variant.yml",
    tabular_log_file="progress.csv",
    snapshot_mode="last",
    snapshot_gap=1,
    log_dir=None,
):
    if variant is not None:
        logger.log("Variant:")
        logger.log(yaml.dump(variant))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    return log_dir


class Logger:
    def __init__(self):
        self._prefix_str = ""
        self._prefixes = []
        self._log_dict = {}
        self._tb_logs = {}

        self._snapshot_dir = None
        self._snapshot_mode = "all"
        self._snapshot_gap = 1

    def log(self, s, with_timestamp=True):
        out = s
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
            out = "{} | {}".format(timestamp, out)
            print(out)

    def record_entry(self, key, val):
        log_str = self._prefix_str + str(key)
        self._log_dict[log_str] = val

    def record_dict(self, d, global_step, prefix=None):
        if prefix is not None:
            self.record_tensorboard(d, global_step, prefix[:-1])
        if prefix is not None:
            self.push_prefix(prefix)
        for k, v in d.items():
            self.record_entry(k, v)
        if prefix is not None:
            self.pop_prefix()

    def record_tensorboard(self, d, global_step, prefix):
        if prefix not in self._tb_logs:
            self._tb_logs[prefix] = SummaryWriter(
                os.path.join(self._snapshot_dir, prefix)
            )

        tb_log = self._tb_logs[prefix]
        for k, v in d.items():
            if k in [
                "Average Returns",
                "Num Paths",
                "num_paths_total",
            ]:
                continue
            k = str(k)
            k = k.replace(" ", "_")
            idx = k.rfind("_")
            if idx >= 0:
                k = k[:idx] + "/" + k[idx + 1 :]
            tb_log.add_scalar(k, v, global_step=global_step)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = "".join(self._prefixes)

    def pop_prefix(self):
        del self._prefixes[-1]
        self._prefix_str = "".join(self._prefixes)

    def process_value(self, value):
        if isinstance(value, float):
            prc_value = "{:.5f}".format(value)
        else:
            prc_value = "{}".format(value)
        return prc_value

    def dump_log(self):
        for key, value in self._log_dict.items():
            print("{}: {}".format(key, self.process_value(value)))
        progress_name = osp.join(self._snapshot_dir, "progress.yml")
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self._log_dict["timestamp/"] = now.timestamp()
        with open(progress_name, "a") as f:
            a = yaml.dump(self._log_dict)
            f.write(yaml.dump(self._log_dict))
            f.write("§")
        self._log_dict = {}

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == "all":
                file_name = osp.join(self._snapshot_dir, "itr_{}.pth".format(itr))
                torch.save(params, file_name)
            elif self._snapshot_mode == "last":
                file_name = osp.join(self._snapshot_dir, "params.pth")
                torch.save(params, file_name)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, "itr_{}.pth".format(itr))
                    torch.save(params, file_name)
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, "itr_{}.pth".format(itr))
                    torch.save(params, file_name)
                file_name = osp.join(self._snapshot_dir, "params.pth")
                torch.save(params, file_name)
            elif self._snapshot_mode == "none":
                pass
            else:
                raise NotImplementedError

    def log_variant(self, log_file, variant_data, resume=False):
        mkdir_p(os.path.dirname(log_file), erase_path=not resume)
        with open(log_file, "w") as f:
            f.write(yaml.dump(variant_data))

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self,):
        return self._snapshot_dir

    def get_snapshot_mode(self,):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self,):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap


logger = Logger()
