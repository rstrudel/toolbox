import matplotlib.pyplot as plt
import numpy as np

from toolbox import colors


class Plotter:
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 1)
        self.data = []
        self.lines = []
        self.lim_factor = 1.1

    def reset(self):
        self.data = []

    def plot_lines(self, v):
        self.data.append(v)
        y = np.array(self.data)
        x = np.arange(len(y))
        if not self.lines:
            self.lines = []
            for i, column in enumerate(y.T):
                line = self.axs.plot(x, column, "r-", color=colors.pop(), label=f"{i}")[
                    0
                ]
                self.lines.append(line)
            plt.legend()
        else:
            for column, line in zip(y.T, self.lines):
                line.set_data(x, column)

        lf = self.lim_factor
        self.axs.set_xlim(lf * x.min(), lf * x.max())
        self.axs.set_ylim(lf * y.min(), lf * y.max())
        plt.pause(1e-5)
