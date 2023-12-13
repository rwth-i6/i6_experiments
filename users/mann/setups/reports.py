from sisyphus import tk
from sisyphus.job_path import VariableNotSet
from sisyphus.delayed_ops import DelayedBase

import os
import tabulate as tab

class ReportMinimumError(Exception):
    pass

class ReportMinimum:
    """
    Class to wrap around a collection of values to later report their minimum.
    Useful for e.g. reporting only the minimum WER for recognition tuning.
    """
    def __init__(self, variables, filter_non_ready=False):
        self.vars = variables
        self.filter_non_ready = filter_non_ready
    
    def get(self):
        try:
            if self.filter_non_ready:
                return min(var.get() for var in self.vars if var.is_set())
            return min(var.get() for var in self.vars)
        except ValueError:
            raise ReportMinimumError("No minimum found")
        

def maybe_get(var):
    try:
        return var.get()
    except (VariableNotSet, FileNotFoundError, ReportMinimumError):
        return ""
    # return var.get() if var.is_set() else ""

eval_types = (DelayedBase, ReportMinimum)

def eval_tree(o, f=maybe_get, condition=lambda x: isinstance(x, eval_types)):
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    :param Any o: nested structure that may contain DelayedBase objects
    :return:
    """
    if condition(o):
        o = f(o)
    elif isinstance(o, list):
        for k in range(len(o)):
            o[k] = eval_tree(o[k], f, condition)
    elif isinstance(o, tuple):
        o = tuple(eval_tree(e, f, condition) for e in o)
    elif isinstance(o, dict):
        for k in o:
            o[k] = eval_tree(o[k], f, condition)
    return o

class SimpleValueReport:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return str(self.value)

class DescValueReport:
    def __init__(self, values: dict):
        self.values = values
    
    def __call__(self):
        keys = [str(key) for key in self.values]
        max_width = max(len(key) for key in keys)
        fmt = "{0:<%d}: {1}" % max_width
        return "\n".join(fmt.format(key, value) for key, value in self.values.items())

class TableReport:
    def __init__(self, data, floatfmt=None, tablefmt="presto"):
        self.data = data

        self.floatfmt = floatfmt
        self.tablefmt = tablefmt
    
    def __call__(self):
        kwargs = {}
        if (fmt := self.floatfmt):
            kwargs["floatfmt"] = fmt
        data = eval_tree(self.data)
        table = tab.tabulate(
            data,
            headers="keys",
            tablefmt=self.tablefmt,
            **kwargs,
        )
        return table

def print_report(fname, name, data, **kwargs):
    tk.register_report(
        os.path.join(fname, "summary", "{}.txt".format(name)),
        TableReport(data, **kwargs),
    )

class ReportSystem:
    def __init__(self, fname):
        self.fname = fname

    def print(self, name, data, **kwargs):
        print_report(self.fname, name, data, **kwargs)

class GenericReport:
    def __init__(self, cols, values, fmt_percent=False, add_latex=False):
        self.values = values
        self.cols = cols
        self.fmt_percent = fmt_percent
        self._latex = add_latex
    
    def __call__(self):
        from tabulate import tabulate
        values = self.values
        table = []
        cols = self.cols
        for cp, stat in values.items():
            row = [cp]
            try:
                for key in cols:
                    v = stat[key].get()
                    if self.fmt_percent:
                        v *= 100
                    row.append(round(v, 2))
            except TypeError as e:
                print(e)
                row += [""] * len(cols)
            table.append(row)
        if self.fmt_percent:
            cols = [c + " [%]" for c in cols]
        buffer = tabulate(table, headers=["corpus"] + cols, tablefmt="presto")
        if self._latex:
            buffer += "\n" * 2
            buffer += tabulate(table, headers=["corpus"] + cols, tablefmt="latex")
        return buffer