import numpy as np
from itertools import cycle


class Lines:


    def __init__(self, resolution=20, smooth=None):
        self.COLORS = cycle([
            '#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
            '#a65628', '#f781bf'
        ])
        self.MARKERS = cycle('os^Dp>d<')
        self.LEGEND = dict(fontsize='medium', labelspacing=0, numpoints=1)
        self._resolution = resolution
        self._smooth_weight = smooth

    def __call__(self, ax, domains, lines, labels):
        assert len(domains) == len(lines) == len(labels)
        for index, (label, color,
                    marker) in enumerate(zip(labels, self.COLORS,
                                             self.MARKERS)):
            domain, line = domains[index], lines[index]
            line = self.smooth(line, self._smooth_weight)
            ax.plot(domain, line[:, 0], color=color, label=label)
            std_min = line[:, 0] - line[:, 1] / 2
            std_max = line[:, 0] + line[:, 1] / 2
            ax.fill_between(domain, std_min, std_max, color=color, alpha=0.2)
        self._plot_legend(ax, lines, labels)

    def _plot_legend(self, ax, lines, labels):
        scores = {
            label: -np.nanmedian(line)
            for label, line in zip(labels, lines)
        }
        handles, labels = ax.get_legend_handles_labels()
        # handles, labels = zip(*sorted(
        #     zip(handles, labels), key=lambda x: scores[x[1]]))
        legend = ax.legend(handles, labels, **self.LEGEND)
        legend.get_frame().set_edgecolor('white')
        for line in legend.get_lines():
            line.set_alpha(1)

    def smooth(self, scalars, weight):
        """
        weight in [0, 1]
        exponential moving average, same as tensorboard
        """
        assert weight >= 0 and weight <= 1
        last = scalars[0]
        smoothed = np.asarray(scalars)
        for i, point in enumerate(scalars):
            smoothed_val = last * weight + (1 - weight) * point
            smoothed[i] = smoothed_val
            last = smoothed_val

        return smoothed
