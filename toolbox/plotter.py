import matplotlib.pyplot as plt
import numpy as np

from toolbox import colors


class Line:
    def __init__(self, ax, label):
        self.ax = ax
        self.line = None
        self.data = {"x": [], "y": []}
        self.label = label

    def reset(self):
        self.data = {"x": [], "y": []}

    def min(self):
        return (min(self.data["x"]), min(self.data["y"]))

    def max(self):
        return (max(self.data["x"]), max(self.data["y"]))

    def update(self, value):
        ax = self.ax
        self.data["y"] = self.data["y"] + [value]
        self.data["x"] = np.arange(len(self.data["y"]))
        if self.line is None:
            self.line = ax.plot(
                self.data["x"],
                self.data["y"],
                "r-",
                color=colors.pop(),
                label=self.label,
            )[0]
            ax.legend()
        else:
            self.line.set_data(self.data["x"], self.data["y"])


class Plotter:
    def __init__(self, rows, columns):
        self.fig, self.axs = plt.subplots(rows, columns)
        if rows == 1 and columns == 1:
            self.axs = np.array(self.axs).reshape(1, 1)
        elif columns == 1:
            self.axs = self.axs.reshape(-1, 1)
        elif rows == 1:
            self.axs = self.axs.reshape(1, -1)

        self.lines = {}
        self.ims = {}
        self.lim_factor = 1.1

    def reset(self):
        self.lines = {}
        self.ims = {}

    def plot_line(self, value, label, ax_i=0, ax_j=0, ymin=None, ymax=None):
        ax = self.axs[ax_i, ax_j]
        ax_key = f"{ax_i}{ax_j}"
        if ax_key not in self.lines:
            self.lines[ax_key] = {}
        if label not in self.lines[ax_key]:
            line = Line(ax, label)
            self.lines[ax_key][label] = line
        else:
            line = self.lines[ax_key][label]
        line.update(value)

        ax_lines = self.lines[ax_key].values()
        vmin = np.vstack([line.min() for line in ax_lines]).min(axis=0)
        vmax = np.vstack([line.max() for line in ax_lines]).max(axis=0)
        idx_scale = np.where(vmax <= vmin)
        vmax[idx_scale] = 1.1 * vmin[idx_scale] + 1e-5

        lf = self.lim_factor
        ax.set_xlim(lf * vmin[0], lf * vmax[0])
        # ax.set_ylim(lf * vmin[1], lf * vmax[1])
        if ymin is None:
            ymin = lf * vmin[1]
        if ymax is None:
            ymax = lf * vmax[1]
        ax.set_ylim(ymin, ymax)

    def plot_im(self, A, ax_i=0, ax_j=0, label="", **imshow_kwargs):
        ax = self.axs[ax_i, ax_j]
        ax_key = f"{ax_i}{ax_j}"
        if ax_key not in self.ims:
            self.ims[ax_key] = ax.imshow(A, **imshow_kwargs)
        else:
            im = self.ims[ax_key]
            im.set_data(A)

    def show(self, timeout=1e-5):
        plt.pause(timeout)
