import pathlib
import json


# From icefall/utils.py
class AttributeDict(dict):
    __slots__ = ("got_items",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.got_items = set()

    def __getitem__(self, item):
        res = super().__getitem__(item)
        self.got_items.add(item)
        return res

    def get(self, item, default=None):
        """
        :param str item:
        :param T default:
        :rtype: T|typing.Any|None
        """
        try:
            return self[item]
        except KeyError:
            return default

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        if key in ("got_items",):
            return super().__setattr__(key, value)
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        import torch

        tmp = {}
        for k, v in self.items():
            # PosixPath is ont JSON serializable
            if isinstance(v, pathlib.Path) or isinstance(v, torch.device):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)
