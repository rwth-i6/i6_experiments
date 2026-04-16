#!/usr/bin/env python3

__all__ = ["ProcessAedListJob"]

from sisyphus import tk

import ast
import gzip
import json
from typing import Any, Dict, List, Sequence, Tuple, Union
import math

Number = Union[int, float]
Pair = Union[Tuple[Any, Any], List[Any]]  # allow tuple or list pairs

def load_dict(path: tk.Path):
    
    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read()
    data = ast.literal_eval(text)
    
    return data

def normalize_sum(values: List[Number]):
    s = sum([math.exp(v) for v in values])
    if s == 0:
        return values[:]
    return [10000 * math.exp(v) / s for v in values]

def process_list(
    lst: Sequence[Pair],
    n: int,
):
    trimmed = list(lst[:n])

    firsts: List[Number] = []
    numeric_mask: List[bool] = []

    for item in trimmed:
        firsts.append(item[0])
        numeric_mask.append(True)

    normalized = normalize_sum([v for v, ok in zip(firsts, numeric_mask) if ok])

    norm_iter = iter(normalized)
    out: List[Pair] = []
    for item, ok in zip(trimmed, numeric_mask):
        if ok:
            new_first = next(norm_iter)
            
            if isinstance(item, tuple):
                out.append((new_first, *item[1:]))
            else:
                new_item = list(item)
                new_item[0] = new_first
                out.append(new_item)
        else:
            out.append(item)
    return out


def transform_dict(
    data: Dict[str, Any],
    n: int
):
    
    out: Dict[str, Any] = {}
    for k, v in data.items():
        out[k] = process_list(v, n=n)
    
    return out


def trim_norm(n_best_list: tk.Path, n: int = 100):

    data = load_dict(n_best_list)
    transformed = transform_dict(data, n=n)
    return transformed

def process_dict(data: dict, n: int):

    processed_data = {}
    for key, values in data.items():
        processed_data[key] = [values[n]]

    return processed_data

class ProcessAedListJob(tk.Job):
    def __init__(self, n_best_list: tk.Path, n: int = 100):
        """
        Obtains an N-best list in the form of a dict and processed it into fragmented dicts.

        :param tk.Path n_best_list: Path to the N-best list
        :param int n: The number of hypothesis fragment dicts to be obtained from the dict
        """

        self.n = n
        self.n_best_list = n_best_list
        self.out_dir = self.output_path("")

    def task(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        data = trim_norm(self.n_best_list, self.n)
        for i in range(self.n):
            with gzip.open(self.output_path(f'{i}.py.gz'), 'wt') as f:
                json.dump(process_dict(data, i), f,ensure_ascii=False, indent=2)



