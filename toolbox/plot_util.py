import numpy as np
from itertools import cycle


class Lines:

    COLORS = cycle([
        '#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
        '#a65628', '#f781bf'
    ])
    MARKERS = cycle('os^Dp>d<')
    LEGEND = dict(fontsize='medium', labelspacing=0, numpoints=1)

    def __init__(self, resolution=20, smooth=None):
        self._resolution = resolution
        self._smooth = smooth

    def __call__(self, ax, domains, lines, labels):
        assert len(domains) == len(lines) == len(labels)
        for index, (label, color,
                    marker) in enumerate(zip(labels, self.COLORS,
                                             self.MARKERS)):
            domain, line = domains[index], lines[index]
            ax.plot(domain, line, c=color, label=label)
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

    def _smooth_line(self, values, steps, amount):
        # Step-aware smoothing (for irregular steps).
        # weights = (1 - 10 ** -amount) ** np.abs(steps[:, None] - steps[None, :])
        weights = amount**np.abs(
            np.arange(len(values))[:, None] - np.arange(len(values))[None, :])
        weights /= weights.sum(1)
        smooth_values = values.dot(weights)
        return smooth_values
