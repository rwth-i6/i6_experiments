#!/work/smt4/dorian.koch/.conda/envs/myenv2/bin/python

import os
import json


import argparse
import copy
import os.path
from pprint import pprint


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


important_metrics = [
    (
        "dev-clean",
        "eval/librispeech/dev-clean/dropout00p_specaugmentOFF",
    ),
    (
        "dev-other",
        "eval/librispeech/dev-other/dropout00p_specaugmentOFF",
    ),
]


def EpochData(learningRate, error):
    return {"learning_rate": learningRate, "error": error}


def get_epoch_data(filename, epoch):
    def _get_last_epoch(epoch_data):
        return max([ep for ep in epoch_data.keys() if epoch_data[ep]["error"]])

    if not os.path.exists(filename):
        return None
    if os.path.isdir(filename):
        if os.path.exists(os.path.join(filename, "work/learning_rates")):
            filename = os.path.join(filename, "work/learning_rates")
        else:
            return None
    with open(filename, "r") as f:
        data = eval(f.read())
    if epoch is None:
        return data
    last_epoch = _get_last_epoch(data)  # sorted(list(data.keys()))[-1]
    if epoch == "last":
        epoch = last_epoch
    else:
        try:
            epoch = int(epoch)
        except ValueError:
            raise ValueError('epoch must be "last" or int')
        epoch = min(epoch, last_epoch)
    progr = data[epoch]
    progr["epoch"] = epoch
    return progr


def find_intermediary(file):
    global important_metrics

    if not os.path.exists(file):
        # print("  |", "not found")
        return
    data = get_epoch_data(file, None)
    if data is None:
        print("  |", "no data")
        return
    print("  |", "intermediary ce:")
    i = 1
    while i in data:
        print("  -", i)
        err = data[i]["error"]
        for metric, key in important_metrics:
            kk = key + "_loss_ce"
            if kk in err:
                print("    |", metric, err[kk])
            else:
                print("    |", metric, "not found")
        if ":meta:effective_learning_rate" in err:
            print(
                "    |",
                "effective_learning_rate",
                "{:.2e}".format(err[":meta:effective_learning_rate"]),
            )
        i *= 2


def find_intermediary_wer(directory):
    global important_metrics
    if not os.path.exists(directory):
        print(
            "  |",
            "recog results directory does not exist! Have we started training?",
            directory,
        )
        return
    # list directory
    last_epoch = None
    for last_epoch in sorted(os.listdir(directory)):
        pass
    print("  |", "still training?, data from recent epoch:", last_epoch)
    filename = directory + last_epoch
    with open(filename, "r") as f:
        data = eval(f.read())
    for metric, key in important_metrics:
        if key in data:
            if data[key] <= baselines[metric]:
                print(
                    bcolors.OKGREEN + "    |",
                    metric,
                    data[key],
                    bcolors.ENDC,
                )
            else:
                print("    |", metric, data[key])
        else:
            print("    |", metric, "not found")


def print_rescored_wer(filename):
    global important_metrics
    # list directory
    if not os.path.exists(filename):
        print("  |", "no rescores exist...")
        return
    print("  |", "rescored:")
    with open(filename, "r") as f:
        data = eval(f.read())
    for metric, _ in important_metrics:
        if metric in data:
            if data[metric] <= baselines[metric]:
                print(
                    bcolors.OKGREEN + "    |",
                    metric,
                    data[metric],
                    bcolors.ENDC,
                )
            else:
                print("    |", metric, data[metric])
        else:
            print("    |", metric, "not found")


print("+++++++++++++++")
print("Baseline from ctc model:")
baselines = dict()
for metric, key in important_metrics:
    filename = f"output/2024-denoising-lm/hyps_from_model_spm10k_tts/eval/librispeech/{metric}/dropout00p_specaugmentOFF/hyps0_score_wer_against_ref"
    with open(filename, "r") as f:
        data = eval(f.read())
    print("  |", metric, data)
    baselines[metric] = data

print("+++++++++++++++")


dstart = "alias/2024-denoising-lm/error_correction_model/"
for dend in ["dataset_train-tiny_tts/", "dataset_train_tts/"]:
    print(dend)
    d = dstart + dend

    for trafo in sorted(os.listdir(d)):
        print("#", trafo)
        for filename in sorted(os.listdir(d + trafo)):
            print(" #", filename)
            summarize_file = d + trafo + "/" + filename + "/train-summarize"
            summarize_output = summarize_file + "/output/summary.json"
            # check if symlink is working
            if not os.path.exists(summarize_file) or not os.path.exists(
                summarize_output
            ):
                # job is not yet done, find intermediary results
                find_intermediary_wer(
                    "output"
                    + d[5:]
                    + trafo
                    + "/"
                    + filename
                    + "/recog_results_per_epoch/"
                )
                # find_intermediary(
                #    d + trafo + "/" + filename + "/train/work/learning_rates"
                # )
                continue

            with open(summarize_output, "r") as f:
                data = json.load(f)
                best_scores = data["best_scores"]

                for metric, key in important_metrics:
                    if key in best_scores:
                        if best_scores[key] <= baselines[metric]:
                            print(
                                bcolors.OKGREEN + "  |",
                                metric,
                                best_scores[key],
                                bcolors.ENDC,
                            )
                        else:
                            print("  |", metric, best_scores[key])
                    else:
                        print("  |", metric, "not found")
                # print with keys in alphabetical order
                # print(json.dumps(best_scores, indent=4, sort_keys=True))
            print_rescored_wer(
                "output"
                + d[5:]
                + trafo
                + "/"
                + filename
                + "/rescored-beam64-lmScale1.20-priorScaleRel0.00"
            )
