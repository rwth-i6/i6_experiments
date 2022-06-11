# This script allowes to run a eval on all stored models of a train
# i.e.: for all setups that trained without 'devtrain' initally
import os
import shutil
import time
import subprocess
import glob

import re

# Now what do we actually want to do? We want to run an eval task!

# e.g.:
# WE have this job: /work/asr3/zeyer/schupp/setups-data/ping_setup_refactor_tf23_work/i6_core/returnn/rasr_training/ReturnnRasrTrainingJob.0XNwa2tSw4Oq
# Then lets make a clone '/work/asr3/zeyer/schupp/setups-data/ping_setup_refactor_tf23_work/i6_core/returnn/rasr_training/ReturnnRasrTrainingJobCLONE.0XNwa2tSw4Oq'

# Then lets set in the config:
# eval_datasets = {...}
# task = "eval"
# And we need to add file: devtrain.rasr.config ( just steal form here /work/asr3/zeyer/schupp/setups-data/i6_exp_conformer_rtc_work/i6_experiments/users/schupp/hybrid_hmm_nn/helpers/returnn_helpers/ReturnnRasrTrainingJobDevtrain.HTLI4c4dJIPG )


RASR_FILES = [ # I'm pretty sure we can just reuse them as is
    f"/u/schupp/setups/data_god/post_train_devtrain/{f}" for f in [
        "rasr.devtrain.config",
        "rasr.dev.config",
        "rasr.train.config",
        "feature.flow",
        "class.labels",
        "dummy.flow",
        "rnn.sh"
    ]
]


# This will all the devtrain dataset
extra_code = """
eval_datasets = {
    "devtrain": {
        "class": "ExternSprintDataset",
        "partitionEpoch": 1,
        "sprintConfigStr": "--config=rasr.devtrain.config --*.LOGFILE=nn-trainer.devtrain.log --*.TASK=1",
        "sprintTrainerExecPath": "/work/asr3/zeyer/schupp/setups-data/custom_rasr/rasr/arch/linux-x86_64-generic_mkl/nn-trainer.linux-x86_64-generic_mkl",
    }
}
eval_output_file = "eval.out.log"\n
"""


float_or_int = r'[\d\.\d]+'

def create_eval_environment(config_path, work_dir_path, use_cpu=True):
    os.chdir(work_dir_path)

    # 1 - copy all rasr files to the workdir path
    for f in RASR_FILES:
        shutil.copy(f, ".") # copy with same name to current dir...

    # 2 - then read the config
    with open(config_path, "r") as config:
        data = config.read()

    # 3 - modify and save returnn.config
    lines = data.split("\n")
    num_epochs = None # We need to know because we want to make eval for the final epoch
    new_lines = []
    for l in lines:
        if "num_epochs" in l:
            num_epochs = int(re.findall(float_or_int, l)[0])
        elif "task = 'train'" in l: # This line shall be updated
            l = "task = 'eval'" # Can I just update this?
        elif use_cpu and "device = 'gpu'" in l:
            l = "device = 'cpu'"
        new_lines.append(l) # not efficient but easy

    modified = "\n".join(new_lines)
    modified += extra_code # 'devtrain' and eval output code
    modified += f"load_epoch = {num_epochs}"

    with open('returnn.config', "w") as config:
        config.write(modified)

    # Ok environment is set up

def submit_devtrain_eval(work_dir_path, sub_ex_name, use_cpu=True):
    os.chdir(work_dir_path)

    # 1 - submit to SGE
    gpu = 0 if use_cpu else 1
    call = f"qsub -cwd -N {sub_ex_name} -j y -o test.log -S /bin/bash -m n -l h_vmem=12G -l h_rss=12G -l h_fsize=50G -l gpu={gpu} -l num_proc=3 -l h_rt=57600 ./rnn.sh".split(" ")
    out = subprocess.check_output(call).decode('UTF-8')
    print(out) # finish ...

LIST_OF_EXPERIMENTS = [
    "/u/schupp/conformer_tf23_rasrV2/alias/conformer/blocks_amnt/conformer__conf__conv_amnt-blocks-14/train.job",
    "/u/schupp/conformer_tf23_rasrV2/alias/conformer/blocks_amnt/conformer__conf__conv_amnt-blocks-15/train.job"
]

EX_TO_IGNORE = [
    "stoch_depth_traincond",
    "test_new_experiments_struct",
    "tests",
    "batchnorm_ping",
    "lr_shed"
]

GLOBAL_WORK_DIR = "/u/schupp/setups/data_god/devtrain_eval/work"

BASE = "conformer"
all_existing_experiments = [s.split("/")[-1] for s in glob.glob(f"/u/schupp/conformer_tf23_rasrV2/alias/{BASE}/*") ]

sub_experiments = {k : [] for k in all_existing_experiments}
all_exp = []
for i, k in enumerate(all_existing_experiments):
    if k in EX_TO_IGNORE:
        continue
    sub_experiments[k] = glob.glob(f"/u/schupp/conformer_tf23_rasrV2/alias/conformer/{all_existing_experiments[i]}/*")
    for ex in sub_experiments[k]:
        if not "recog_" in ex:
            all_exp.append(ex + "/train.job")


if __name__ == "__main__":

    # 1 - update list of experiments
    #print(all_exp)
    #print(len(all_exp))

    LIST_OF_EXPERIMENTS += all_exp

    use_cpu = False
    only_first_X = 50
    i = 0

    # 2 - start all the seraches
    for experiment in LIST_OF_EXPERIMENTS:
        if i > only_first_X:
            break
        
        real_path = os.readlink(experiment)
        exp, sub_exp = (experiment.replace("/train.job", "")).split("/")[-2:]
        print(exp)
        print(sub_exp)

        # 0 - back to global work dir
        os.chdir(GLOBAL_WORK_DIR)

        # 1 - Setup experiment dirs
        if not os.path.exists(exp):
            os.mkdir(exp)
        os.chdir(exp)

        if os.path.exists(sub_exp):
            print("WARN: sub ex path exists, assuming already submitted skipping")
            continue

        os.mkdir(sub_exp)
        os.chdir(sub_exp)

        # 2 - setup work dir, copy and edit config ...
        create_eval_environment(f"{experiment}/output/returnn.config", os.getcwd(), use_cpu)

        # 3 - submit eval job
        submit_devtrain_eval(os.getcwd(), "devtrain_eval_" + exp + "_" + sub_exp, use_cpu)

        time.sleep(2)
        i += 1

    print(f"submitted first {only_first_X}")





