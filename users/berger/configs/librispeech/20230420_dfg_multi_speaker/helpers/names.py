"""
Helpers to create names for experiments
"""


def num2human(num):
    """
    Get string representation of number, e.g. 2k or 1.5M

    :param int|float num:
    :rtype: str
    """
    num = float("{:.3g}".format(num))  # print 3 decimal places
    if 0 < num < 1:
        return "{:.2e}".format(num)  # return as scientific notation
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude])


def any2str(x):
    """
    Convert any input to str for experiment name

    :param x:
    :rtype: str
    """
    if isinstance(x, str):
        return x
    elif type(x) in (int, float):
        return num2human(x)
    elif type(x) in (list, tuple):
        return "_".join([any2str(x_) for x_ in x])
    elif isinstance(x, dict):
        return "_".join(["{}_{}".format(any2str(k), any2str(v)) for k, v in x.items()])
    else:
        return str(x)


def get_experiment_name(**kwargs):
    """
    Create experiment name based on kwargs. Name is of form key1_value1-key2_value2-...

    :rtype: str
    """
    # noinspection SpellCheckingInspection
    map_keys = {
        "learning_rate": "lr",
        "learning_rates": "lr",
        "min_learning_rate": "minlr",
    }
    d = kwargs or {}
    d_out = {}
    for k, v in d.items():
        k_out = map_keys.get(k, k)
        d_out[k_out] = any2str(v)
    name = "-".join(["_".join([str(elem) for elem in pair]) for pair in d_out.items()])
    name = name.replace(" ", "")
    return name
