
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
    cmd = ["cat", "alias/%s/%s/train.job/submit_log.run" % (prefix, n)]
    try:
        out = subprocess.check_output(" ".join(cmd), shell=True).decode('UTF-8')
    except:
        print("Can't find the job %s" % n)
        continue


    cut = out[out.index("engine_info"): len(out) -2]
    cut2 = cut[cut.index("("): cut.index(")")]
    pid = int(cut2[cut2.index("'") +1: len(cut2) - 1])
    print("Killing: %s" % pid)
    try:
        out = subprocess.check_output("qdel %s" % pid, shell=True).decode('UTF-8')
    except Exception as e: print(e)
    print(out)