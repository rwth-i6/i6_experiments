#!/work/smt4/dorian.koch/.conda/envs/myenv2/bin/python

import os
import json


import argparse
import copy
import os.path
from pprint import pprint

import numpy as np
import sys
sys.path.append("./recipe")

from i6_experiments.users.dorian_koch.misc.automl import GaussianProcessOptimizer, plot_gaussian_process


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
    (
        "tts-trainlike",
        "dev_eval/dev/dropout00p_specaugmentOFF"
    )
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


def find_intermediary_wer(directory, fout=None):
    global important_metrics
    if not os.path.exists(directory):
        print(
            "  |",
            "recog results directory does not exist! Have we started training?",
            directory, file=fout
        )
        return
    # list directory
    last_epoch = None
    for last_epoch in sorted(os.listdir(directory)):
        pass
    print("  |", "still training?, data from recent epoch:", last_epoch, file=fout)
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
                    " + better than baseline",
                    bcolors.ENDC, file=fout
                )
            else:
                print("    |", metric, data[key], file=fout)
        else:
            print("    |", metric, "not found", file=fout)


def print_rescored_wer(filename):
    global important_metrics
    # list directory
    if not os.path.exists(filename):
        print("  |", "no rescores exist...")
        return {}
    print("  |", "rescored:")
    with open(filename, "r") as f:
        data = eval(f.read())
    retDict = {}
    for metric, _ in important_metrics:
        if metric in data:
            if data[metric] <= baselines[metric]:
                print(
                    bcolors.OKGREEN + "    |",
                    metric,
                    data[metric],
                    " + better than baseline",
                    bcolors.ENDC,
                )
            else:
                print("    |", metric, data[metric])
            retDict[metric] = data[metric]
        else:
            print("    |", metric, "not found")
    return retDict

def print_summarize(summarize_output, fout=None):
    ret = {}
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
                        " + better than baseline",
                        bcolors.ENDC,
                        file=fout,
                    )
                else:
                    print("  |", metric, best_scores[key], file=fout)
                ret[metric] = best_scores[key]
            else:
                print("  |", metric, "not found", file=fout)
    return ret


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filter",
    type=str,
    help="Filter model to extract values from",
    default=None,
    nargs="?",
)
parser.add_argument(
    "--rescore",
    action="store_true",
    help="Re-score results",
    default=False,
)
parser.add_argument(
    "--suggest",
    action="store_true",
    default=False
)
args = parser.parse_args()

print("+++++++++++++++")
print("Baseline from ctc model:")
baselines = dict()
for metric, key in important_metrics:
    filename = f"output/2024-denoising-lm/hyps_from_model_spm10k_tts/eval/librispeech/{metric}/dropout00p_specaugmentOFF/hyps0_score_wer_against_ref"
    if not os.path.exists(filename):
        baselines[metric] = 0
        continue
    with open(filename, "r") as f:
        data = eval(f.read())
    print("  |", metric, data)
    baselines[metric] = data

print("+++++++++++++++")

def trains_generator():
    dstart = "alias/2024-denoising-lm/error_correction_model/"
    for dataset_name in ["dataset_train_tts"]: # "dataset_train-tiny_tts", 
        print(dataset_name)
        d = dstart + dataset_name + "/"

        for trafo in sorted(os.listdir(d)):
            print("#", trafo)
            for filename in sorted(os.listdir(d + trafo)):
                yield (dataset_name, trafo, d + trafo + "/" + filename)


all_metrics = {}
#fout = sys.stdout
# devnull
fout = open(os.devnull, 'w')

for (dataset_name, trafo, filename) in trains_generator():
    if args.filter is not None and args.filter not in filename:
        continue
    print(" #", filename)
    summarize_file = filename + "/train-summarize"
    summarize_output = summarize_file + "/output/summary.json"
    # check if symlink is working
    if not os.path.exists(summarize_file) or not os.path.exists(
        summarize_output
    ):
        # job is not yet done, find intermediary results
        find_intermediary_wer(
            "output"
            + filename[5:]
            + "/recog_results_per_epoch/", fout=fout
        )
        # find_intermediary(
        #    d + trafo + "/" + filename + "/train/work/learning_rates"
        # )
        continue
    summary_metrics = print_summarize(summarize_output, fout=fout)

    cur_ds_mets = all_metrics.get(dataset_name, {})
    cur_ds_mets[filename] = summary_metrics
    all_metrics[dataset_name] = cur_ds_mets

    if args.rescore:
        print_rescored_wer(
            "output"
            + filename[5:]
            + "/rescored-beam64-lmScale1.20-priorScaleRel0.00"
        )

import re
import matplotlib.pyplot as plt
pattern = r"(trafo_ff_dropout00p|trafo_ffg_dropout10p)/doriank-ratio(\d+)p_dropout(\d+)p_specaugment(ON|OFF)_repeat(\d+)_numhyps(\d+)-lr((\d+e-?\d+))"

#print(all_metrics)
for dataset_name, metrics in all_metrics.items():
    if dataset_name == "dataset_train_tiny_tts":
        continue
    print("=======", dataset_name)
    # to array
    ratios = []
    arr = []
    for filename, metric in metrics.items():
        arr.append((filename, metric.get("dev-other", 999)+ metric.get("dev-clean", 999)))
        matches = re.findall(pattern, filename)
        if len(matches) == 1:
            matches = matches[0]
            ratios.append((int(matches[1]), metric, matches))

    # plot ratios
    ratios.sort(key=lambda x: x[0])
    print(ratios)

    plt.figure(figsize=(8, 5))
    for metric in ["dev-clean", "dev-other", "tts-trainlike"]:
        if any(metric not in r[1] for r in ratios):
            print(f"Skipping {metric} because its not in all ratios")
            continue
        x = [r[0] for r in ratios]
        y = [r[1][metric] for r in ratios]
        plt.plot(x, y, marker="o", linestyle="-", label=metric)

    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel("Librispeech mixing (%)")
    plt.ylabel("Metric Value")
    plt.title(f"Mixing vs. Metric ({dataset_name})")
    plt.ylim(0, 12.5)

    plt.grid(True)
    plt.legend()

    # save to file
    plt.savefig(dataset_name + ".png")

    # sort
    arr.sort(key=lambda x: x[1])
    #arr.sort(key=lambda x: x[0])
    for filename, metric in arr:
        print(filename, metric)

    if args.suggest:
        # from skopt import Optimizer
        # from skopt.acquisition import _gaussian_acquisition
        from skopt.space import Real, Integer, Categorical
        # from skopt.plots import _evenly_sample

        search_space = [
            Real(0.01, 10.0, name='learning_rate', prior='log-uniform'),
            Integer(0, 100 // 5, name='mixing_ratio_div_5'),
            Categorical([0, 10], name='data_dropout'),
            Categorical(["trafo_ff_dropout00p", "trafo_ffg_dropout10p"], name='trafo', transform="label")
        ]

        previous_results = []

        for ratio, metric, matches in ratios:
            trafo = matches[0]
            data_dropout = int(matches[2])
            data_specaugment = matches[3] == "ON"
            repeat = int(matches[4])
            numhyps = int(matches[5])
            lr = float(matches[6])

            if repeat > 1 or numhyps > 1 or data_specaugment or data_dropout > 10:
                continue # ignore these
            # print(f"{metric}: ratio={ratio}, data_dropout={data_dropout}, data_specaugment={data_specaugment}, repeat={repeat}, numhyps={numhyps}, lr={lr}")
            previous_results.append({
                "learning_rate": lr,
                "mixing_ratio_div_5": ratio // 5,
                "data_dropout": data_dropout,
                "trafo": trafo,
                "score": metric["dev-other"] + metric["dev-clean"]
            })

        print("previous results:", len(previous_results), "excluded:", len(ratios) - len(previous_results))
        assert len(previous_results) > 0
        optimizer = GaussianProcessOptimizer(search_space, previous_results)
        # clear fig
        import traceback
        datapoint_for_eval = [0.75, 20, 0, "trafo_ff_dropout00p"]
        for dim in range(3):
            try:
                plt.clf()

                plot_gaussian_process(optimizer.optimizer.get_result(), datapoint_for_eval, dim=dim)
                import os
                fname = f"automl_0_dim{dim}"
                i = 1
                while os.path.exists(fname + ".png"):
                    fname = f"automl_{i}_dim{dim}"
                    i += 1
                print(fname)
                plt.savefig(fname + ".png")
            except Exception as e:
                traceback.print_exc()
        # suggest new trials
        print("Suggested trials:")
        next_points = optimizer.suggest_next_points(n_points=16)
        # print("next_points:", next_points[0])
        well_formatted = []
        res = optimizer.optimizer.get_result()
        for p in next_points:
            lr = p["learning_rate"]
            mixing_ratio = p["mixing_ratio_div_5"] * 5
            data_dropout = p["data_dropout"]
            trafo = p["trafo"]
            # print learning rate as e-notation
            lr = "{:.2e}".format(lr)

            transformed_data = [[p["learning_rate"], p["mixing_ratio_div_5"], p["data_dropout"], p["trafo"]]]
            transformed_data = res.space.transform(transformed_data)
            y_pred = res.models[-1].predict(transformed_data)

            print(f"lr={lr}, mixing_ratio={mixing_ratio}, data_dropout={data_dropout}, trafo={trafo}; ypred={y_pred}")
            well_formatted.append(
                {
                    "learning_rate": lr,
                    "mixing_ratio": mixing_ratio,
                    "data_dropout": data_dropout,
                    "trafo": trafo
                }
            )
            
        print("copy:")
        print(well_formatted)
