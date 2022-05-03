import sys
import os
import yaml

import subprocess

PY="/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"

with open(sys.argv[1]) as _f:
    names = _f.read().splitlines()

out_map = {}
value_map = {}

prefix = sys.argv[2]

print_loading = False
for n in names:
    print(n + ": epoch: ", end="")
    cmd = ["ls", "alias/%s/%s/train.job/output/models" % (prefix, n), "|", "grep", "epoch", "|", "grep", "index"]
    #print(cmd)
    try:
        out = subprocess.check_output(" ".join(cmd), shell=True).decode('UTF-8')
    except:
        print("NONE")
        continue
    #print(out)
    print(out.split("\n")[-2].split(".")[1])