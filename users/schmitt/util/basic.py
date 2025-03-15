class FrozenDict(dict):
    """
    Frozen dict.
    (Copied from RETURNN.)
    """

    def __setitem__(self, key, value):
        raise ValueError("FrozenDict cannot be modified")

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def make_hashable(obj):
    """
    Sometimes you need hashable objects in some cases.
    This converts all objects as such, i.e. into immutable frozen types.
    (Copied from RETURNN.)

    :param T|dict|list|tuple obj:
    :rtype: T|FrozenDict|tuple
    """
    if isinstance(obj, dict):
        return FrozenDict([make_hashable(item) for item in obj.items()])
    if isinstance(obj, (list, tuple)):
        return tuple([make_hashable(item) for item in obj])
    if isinstance(obj, (str, float, int)):
        return obj
    if obj is None:
        return obj
    return obj
