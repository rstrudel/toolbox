from itertools import cycle

COLORS = cycle(
    [
        "#377eb8",
        "#e41a1c",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]
)


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    values = (int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
    values = tuple(float(v) / 255 for v in values)
    return values


def pop():
    color = next(COLORS)
    color = hex_to_rgb(color)
    return color
