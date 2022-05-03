# returns a bunch of results
# input $1 = file with list of names
# input $2 = prefix
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
    call = PY + " get_wer.py --prefix " + sys.argv[2] + " --job_name " + n + " --print true"
    if print_loading:
        print(call)
    _call = call.split(" ") # We shall never have space in name!!!!

    out = subprocess.check_output(_call).decode('UTF-8')
    out_map[n] = out
    if print_loading:
        print(out)

    value_map[n] = {}

    for line in out.split("\n"):

        if line.startswith("num_params: "):
            params_string = line[line.index(":") + 1: line.index("mio")]
            m_params = float(params_string)
            if print_loading:
                print(m_params)
            value_map[n]["m_params"] = m_params

        if line.startswith("training time per subepoch: ") and line.endswith("h"):
            hours_str = line[line.index(":") + 1: len(line) - 1]
            h_flt = float(hours_str)
            if print_loading:
                print(h_flt)
            value_map[n]["h_train_ep"] = h_flt



    value_map[n]["recog1_wers"] = {}
    value_map[n]["lm_recog_wers"] = {}
    for data_set in ["dev-other", "dev-clean", "test-other", "test-clean"]:
        call = PY + " get_wer_for_set.py --prefix " + sys.argv[2] + " --job_name " + n + " --print true --set %s" % data_set
        if print_loading:
            print(call)
        _call = call.split(" ") # We shall never have space in name!!!!

        out = subprocess.check_output(_call).decode('UTF-8')

        for line in out.split("\n"):
            if line.startswith("recog: "):
                wer_string = line[line.index("{"): line.index("}")+1]
                if print_loading:
                    print(wer_string)
                recog_wers = eval(wer_string) #yaml.safe_load(wer_string.replace("'", "\""))
                value_map[n]["recog1_wers"][data_set] = recog_wers

            if line.startswith("optimized am lm scales:"):
                lm_wers_string = line[line.index("{"): line.index("}")+1]
                if print_loading:
                    print(lm_wers_string)
                lm_recog_wers = eval(lm_wers_string) #yaml.safe_load(wer_string.replace("'", "\""))
                value_map[n]["lm_recog_wers"][data_set] = lm_recog_wers

# Now print it in my preport style:

# V1
all_pendings = []
for n in names:
    print(n)
    indented = ""
    indented += "\n"
    indented_ep = {}

    # I know this aint efficient at all, but I was gona use my own setup, but now hat to switch so you'll only get them hacks

    for data_set in ["dev-other", "dev-clean", "test-other", "test-clean"]:
        if data_set in value_map[n]["recog1_wers"].keys():
            for ep in value_map[n]["recog1_wers"][data_set].keys():
                if not ep in indented_ep.keys():
                    indented_ep[ep] = ""
                indented_ep[ep] += "WER (%s): %s tuned 4gramLM: %s\n" % (data_set, value_map[n]["recog1_wers"][data_set][ep], value_map[n]["lm_recog_wers"][data_set][ep][2])
    
    if len(value_map[n]["recog1_wers"].keys()) == 0:
        indented += "PENDING\n"
        all_pendings.append(n)

    for ep in indented_ep.keys():
        indented += "epoch %s: time: ~ %6.2fh \n" % (ep, int(ep)* value_map[n]["h_train_ep"])
        indented += indented_ep[ep]
        indented += "\n"

    _ind = indented.split("\n")
    indented = "\n".join(["\t" + l for l in _ind])
    print(indented)


print_pending_info = False
pending_map = {}
if print_pending_info:
    print("ALL PENDING:")
    for n in all_pendings:
        pending_map[n] = {}
        print(n)
        print("infos:")
        out = subprocess.check_output(["cat", "alias/%s/%s/train.job/submit_log.run" % (prefix, n)]).decode("UTF-8")
        cut = out[out.index("engine_info"): len(out) -2]
        cut2 = cut[cut.index("("): cut.index(")")]
        print(cut2)
        pending_map[n]["pid"] = int(cut2[cut2.index("'") +1: len(cut2) - 1])
        print(out)
        print("loc:")
        out = subprocess.check_output(["readlink","alias/%s/%s/train.job" % (prefix, n)]).decode("UTF-8")
        pending_map[n]["link"] = out
        print(out)

    print("SUM:")
    for x in pending_map.keys():
        print("%s | %s \n%s" % (x, pending_map[x]["pid"], pending_map[x]["link"]))

