#!/bin/python3
import sys
import importlib

from sisyphus.loader import config_manager
from sisyphus import toolkit as tk

# path to a sisyphus config file
function = sys.argv[1]
config_manager.load_configs([function])

jobs_in_graph = [j._sis_path() for j in sorted(tk.graph.graph.jobs(), key=lambda x: x._sis_path())]
print("Jobs in graph:")
print(jobs_in_graph)

mod = importlib.import_module(function[:-3].replace("/", "."))
jobs_in_test = mod.jobs

assert all([job in jobs_in_graph for job in mod.jobs])
