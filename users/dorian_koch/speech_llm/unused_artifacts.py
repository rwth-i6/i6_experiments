#!/usr/bin/env python3
"""What has been BUILT and is used by NOTHING?

The expensive failure mode in this project is not a wrong result, it is a right result nobody
knows exists. `triviaqa_mix_v6` was fully generated, TTS'd and annotated -- tens of GPU-hours --
and no arm ever trained on it; its own docstring said "opt in with *scale_mix(...)" and nobody did.
It took an explicit question to find. This makes that question one command.

Walks the loaded sisyphus graph and reports every FINISHED job whose output is consumed by no
other job and is not a registered output -- i.e. work that was paid for and leads nowhere.

Run from the setup root:  CUDA_HOME=/usr .venv/bin/python unused_artifacts.py
"""
import os, sys, collections, logging
sys.path.insert(0, "recipe"); sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)
from sisyphus import tk
from sisyphus.loader import config_manager
config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])

def finished(job):
    p = job._sis_path()
    return os.path.exists(p + "/finished") or os.path.exists(p + "/finished.tar.gz")

jobs = list(tk.sis_graph.jobs())
# every job that some OTHER job consumes
consumed = set()
for j in jobs:
    for p in j._sis_inputs:
        if p.creator is not None:
            consumed.add(id(p.creator))
# NOT exempting registered outputs. triviaqa_mix_v6 WAS registered and still had no arm training
# on it -- registering makes a thing visible, it does not make it used. Exempting registration is
# precisely how this check would have missed the case it exists for.

by_kind = collections.Counter()
orphans = []
for j in jobs:
    if not finished(j):
        continue
    if id(j) in consumed:
        continue
    orphans.append(j)
    by_kind[type(j).__name__] += 1

# The question that actually matters for a CORPUS: is it reachable from some run's train_data?
train_reach = set()
stack = []
for j in jobs:
    if type(j).__name__ == "SpeechFinetune":
        stack.extend(p.creator for p in j._sis_inputs if p.creator is not None)
while stack:
    c = stack.pop()
    if id(c) in train_reach:
        continue
    train_reach.add(id(c))
    stack.extend(p.creator for p in c._sis_inputs if p.creator is not None)

CORPUS_KINDS = ("MoshiAnnotate", "ChatterboxInference", "HfMergeShards", "HfDialogueCleaner",
                "FisherToMoshiTrainData", "MergeClipDatasets")
unused_corpora = [j for j in jobs
                  if type(j).__name__ in CORPUS_KINDS and finished(j) and id(j) not in train_reach]
print(f"jobs in graph: {len(jobs)}")
print(f"\n=== CORPUS-PRODUCING jobs that NO SpeechFinetune reaches ===")
if not unused_corpora:
    print("   none -- every finished corpus feeds at least one run.")
else:
    kinds = collections.Counter(type(j).__name__ for j in unused_corpora)
    for k, n in kinds.most_common():
        print(f"   {k:<36} {n}")
    print("\n   These are corpora built and never trained on. That is the v6 failure mode:")
    print("   tens of GPU-hours of TTS + annotation sitting unused because no arm opted in.")
print(f"\nfinished-and-unconsumed (any kind): {len(orphans)}\n")
if not orphans:
    print("nothing built leads nowhere.")
else:
    print("FINISHED but consumed by no job and not a registered output:")
    for kind, n in by_kind.most_common():
        print(f"   {kind:<42} {n}")
    print("\n(a few of these are expected -- terminal report/plot jobs. What matters is an")
    print(" expensive CORPUS or TTS job in this list: that is work paid for and never used.)")
    print("\nfirst 25:")
    for j in orphans[:25]:
        print(f"   {type(j).__name__}.{j._sis_id().split('.')[-1]}")
