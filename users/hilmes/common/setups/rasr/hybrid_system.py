__all__ = ["HybridSystem"]

import copy
import itertools
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)
from i6_core.util import MultiPath, MultiOutputPath
from i6_core.mm import CreateDummyMixturesJob
from i6_core.returnn import ReturnnComputePriorJobV2

from .nn_system import NnSystem, returnn_training

from .util import (
    RasrInitArgs,
    ReturnnRasrDataInput,
    HybridArgs,
    NnRecogArgs,
    RasrSteps,
    NnForcedAlignArgs,
    ReturnnTrainingJobArgs,
    AllowedReturnnTrainingDataInput,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------
from i6_core.report.report import _Report_Type
import numpy as np

def calc_stat(ls):
    avrg = np.average([float(x[1]) for x in ls])
    min = np.min([float(x[1]) for x in ls])
    max = np.max([float(x[1]) for x in ls])
    median = np.median([float(x[1]) for x in ls])
    std = np.std([float(x[1]) for x in ls])
    ex_str = f"Avrg: {avrg}, Min {min}, Max {max}, Median {median}, Std {std}, Values {len(ls)}     {avrg},{min},{max},{median},{std}"
    return ex_str


def hybrid_report_format(report: _Report_Type) -> str:
    quants = report.pop("quant")
    loss_tables = report.pop("loss_tables")
    loss_values = {}
    if len(loss_tables) > 0:
        for job in loss_tables:
            with open(loss_tables[job].out_files["loss_table"]) as f:
                for line in f:
                    loss_values[line.split(" ")[0]] = float(line.split(" ")[1].strip())
    extra_ls = ["iter", "filter", "quant_min_max", "quant_entropy", "quant_percentile", "rtf-intel", "loss_table"]
    out = [(recog, str(report[recog])) for recog in report if not any(extra in recog for extra in extra_ls)]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = []
    if len(out) > 0:
        best_ls.append(out[0])
    for extra in extra_ls:
        if extra == "iter":
            for quant, count in itertools.product(["min_max", "entropy", "percentile"], ["1", "2", "3", "4", "5", "10", "100", "500", "1000", "10000"]):
                others = ["seed", "avrg", "filter", "rtf"]
                out2 = [(recog, str(report[recog])) for recog in report if "iter" in recog and quant in recog and (recog.endswith(count) or recog.endswith(count + "-optlm")) and not any(x in report for x in others)]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + quant + "_" + count, ex_str))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
                # avg list
                out2 = [(recog, str(report[recog])) for recog in report if "iter" in recog and quant in recog and (
                            recog.endswith(count) or recog.endswith(
                        count + "-optlm")) and not "seed" in recog and "avg" in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_avrg" + "_" + quant + "_" + count, ex_str))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
                # different seeds
                for seed in ["24, 2005, 5"]:
                        out2 = [(recog, str(report[recog])) for recog in report if
                                "iter" in recog and quant in recog and (
                                            recog.endswith(count) or recog.endswith(count + "-optlm")) and f"seed_{seed}" in recog]
                        out2 = sorted(out2, key=lambda x: float(x[1]))
                        if len(out2) > 0:
                            ex_str = calc_stat(out2)
                            out.append((quant + "_seed_" + seed + "_" + extra + "-" + count, ex_str))
                            out.extend(out2[:5])
                            out.extend(out2[-5:])
                            best_ls.append(out2[0])
        elif extra == "filter":
            # max and min len filter methods
            for quant, count, mode, thresh in itertools.product(["min_max", "entropy", "percentile"], ["10", "500", "1000"], ["max_calib_len_", "min_calib_len_"], ["50", "100", "250", "400", "500", "1000" "1500"]):
                out2 = [(recog, str(report[recog])) for recog in report if "filter" in recog and quant in recog and (
                            recog.endswith(count) or recog.endswith(count + "-optlm")) and mode+thresh+"-" in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + mode + thresh + "_" + quant + "_" + count, ex_str))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
            # partition filter methods
            partitions = set()
            for recog in report:
                if "partition" in recog:
                    spl = recog.split("_")
                    for i, x in enumerate(spl):
                        if x == "partition":
                            partitions.add(spl[i+1].split("-")[0])  # add the number after the partition which gives identification
            for quant, count, thresh in itertools.product(["min_max", "entropy", "percentile"], ["10", "500", "1000"], partitions):
                mode = "partition_"
                out2 = [(recog, str(report[recog])) for recog in report if "filter" in recog and quant in recog and (
                            recog.endswith(count) or recog.endswith(count + "-optlm")) and mode+thresh in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + mode + thresh + "_" + quant + "_" + count, ex_str))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
            # budget filter methods
            budgets = set()
            for recog in report:
                if "budget" in recog:
                    spl = recog.split("_")
                    for i, x in enumerate(spl):
                        if x == "budget":
                            budgets.add(spl[i+1].split("-")[0])  # add the number after the partition which gives identification
            for quant, thresh, remaind in itertools.product(["min_max", "entropy", "percentile"], budgets, ["0.01", "0.005"]):
                mode = "budget_"
                out2 = [(recog, str(report[recog])) for recog in report if "filter" in recog and quant in recog and mode+thresh+"_"+remaind in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                tmp = []
                sequences = []
                losses = []
                wers = []
                for name, value in out2:
                    job_ls = [quants[x] for x in quants if name.split("/")[1].split("-")[0] == x.split("/")[-2] and quant in x and mode+thresh in x]
                    assert len(job_ls) == 1, (job_ls, out2, quants)
                    with open(job_ls[0].out_seq_info, "rt") as f:
                        loss_tmp = []
                        sequences_tmp = []
                        for line in f:
                            sequences_tmp.append(line.split(" ")[0].strip()[:-1])
                            loss_tmp.append(loss_values[line.split(" ")[0].strip()[:-1]])
                        losses.append(sum(loss_tmp))
                        sequences.append(sequences_tmp)
                        wers.append(float(value))
                    tmp.append((name, value, str(job_ls[0].out_num_seqs)))
                out2 = tmp
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + mode + thresh + "_" + remaind + "_" + quant, ex_str))
                    out.append(("Correlation between loss and drawn sequence", str(np.corrcoef(wers, losses)[0, 1])))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
            # range filter methods
            ranges = set()
            for recog in report:
                if "range_len" in recog:
                    spl = recog.split("_")
                    for i, x in enumerate(spl):
                        if x == "range":
                            ranges.add(f"{spl[i+2]}_{spl[i+3].split('-')[0]}")  # add the number after the partition which gives identification
            for quant, ran in itertools.product(["min_max", "entropy", "percentile"], ranges):
                mode = "range_len_"
                out2 = [(recog, str(report[recog])) for recog in report if "filter" in recog and quant in recog and mode+ran in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                tmp = []
                sequences = []
                losses = []
                wers = []
                for name, value in out2:
                    job_ls = [quants[x] for x in quants if name.split("/")[1].split("-")[0] == x.split("/")[-2] and quant in x and mode+ran in x]
                    assert len(job_ls) == 1, (name, job_ls, out2, quants)
                    with open(job_ls[0].out_seq_info, "rt") as f:
                        sequence = []
                        loss = []
                        for line in f:
                            sequences.append(line.split(" ")[0].strip())
                            wers.append(float(value))
                            loss.append(loss_values[line.split(" ")[0].strip()[:-1]])
                        losses.append(sum(loss))
                        sequences.append(sequence)
                        tmp.append((name.split("/")[1], value, " ".join(f.readlines()).strip()))
                out2 = tmp
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + mode + ran + "_" + quant, ex_str))
                    out.append(("Correlation between loss and drawn sequence", str(np.corrcoef(wers, losses)[0, 1])))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
            # single and unique tag filter methods
            for quant, count, mode in itertools.product(["min_max", "entropy", "percentile"], ["10", "20", "30", "50"], ["single_tag", "unique_tags"]):
                out2 = [(recog, str(report[recog])) for recog in report if "filter" in recog and quant in recog and (
                            recog.endswith(count) or recog.endswith(count + "-optlm")) and mode in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + "_" + mode + "_" + quant + "_" + count, ex_str))
                    out.extend(out2[:5])
                    out.extend(out2[-5:])
                    best_ls.append(out2[0])
        elif extra == "loss_table":
            # loss table
            for quant in ["min_max", "entropy", "percentile", "reverse"]:
                out2 = [(recog, str(report[recog])) for recog in report if extra in recog and quant in recog]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                tmp = []
                sequences = []
                losses = []
                wers = []
                for name, value in out2:
                    job_ls = [quants[x] for x in quants if
                              name.split("/")[1].split("-")[0] == x.split("/")[-2] and extra in x and x.endswith(name.split("/")[1].split("-")[1]) and name.split("/")[0].split("-")[-1] in x]
                    assert len(job_ls) == 1, (name, job_ls, len(out2), quants)
                    with open(job_ls[0].out_seq_info, "rt") as f:
                        for line in f:
                            sequences.append(line.split(" ")[0].strip())
                            wers.append(float(value))
                            losses.append(loss_values[line.split(" ")[0].strip()[:-1]])
                        tmp.append((name.split("/")[0].split("-")[-1]+"/"+name.split("/")[1], value, " ".join(f.readlines()).strip()))
                out2 = tmp
                if len(out2) > 0:
                    ex_str = calc_stat(out2)
                    out.append(('', ''))
                    out.append((extra + " " + quant, ex_str))
                    out.extend(out2)
                    out.extend(out2)
                    best_ls.append(out2[0])
            quant = "reverse"
            out2 = [(recog, str(report[recog])) for recog in report if extra in recog and quant not in recog]
            out2 = sorted(out2, key=lambda x: float(x[1]))
            tmp = []
            sequences = []
            losses = []
            wers = []
            for name, value in out2:
                job_ls = [quants[x] for x in quants if
                          name.split("/")[1].split("-")[0] == x.split("/")[-2] and extra in x and x.endswith(
                              name.split("/")[1].split("-")[1]) and name.split("/")[0].split("-")[-1] in x]
                assert len(job_ls) == 1, (name, job_ls, len(out2), quants)
                with open(job_ls[0].out_seq_info, "rt") as f:
                    for line in f:
                        sequences.append(line.split(" ")[0].strip())
                        wers.append(float(value))
                        losses.append(loss_values[line.split(" ")[0].strip()[:-1]])
                    tmp.append((name.split("/")[0].split("-")[-1] + "/" + name.split("/")[1], value,
                                " ".join(f.readlines()).strip()))
            out2 = tmp
            if len(out2) > 0:
                ex_str = calc_stat(out2)
                out.append(('', ''))
                out.append((extra + " no " + quant, ex_str))
                out.extend(out2)
                out.extend(out2)
                best_ls.append(out2[0])
        else:
            # mixed
            out2 = [(recog, str(report[recog])) for recog in report if extra in recog]
            out2 = sorted(out2, key=lambda x: float(x[1]))
            if len(out2) > 0:
                ex_str = calc_stat(out2)
                out.append(('', ''))
                out.append((extra, ex_str))
                out.extend(out2[:5])
                out.extend(out2[-5:])
                best_ls.append(out2[0])

    best_ls = sorted(best_ls, key=lambda x: float(x[1]))
    best_ls += [("Base Results", "")]
    out = best_ls + out
    out.insert(0, ("Best Results", ""))
    return "\n".join([f"{pair[0]}:  {str(' '.join(pair[1:]))}" for pair in out])



class HybridSystem(NnSystem):
    """
    - 5 corpora types: train, devtrain, cv, dev and test
        devtrain is a small split from the train set which is evaluated like
        the cv but not used for error calculating. Since we can have different
        datasubsets per subepoch, we do not caculate the tran score/error on
        a consistent datasubset
    - two training data settings: defined in returnn config or not
    - 3 different types of decoding: returnn, rasr, rasr-label-sync
    - 2 different lm: count, neural
    - cv is dev for returnn training
    - dev for lm param tuning
    - test corpora for final eval

    settings needed:
    - am
    - lm
    - lexicon
    - ce training
    - ce recognition
    - ce rescoring
    - smbr training
    - smbr recognition
    - smbr rescoring
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        super().__init__(
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            blas_lib=blas_lib,
        )

        self.tf_fwd_input_name = "tf-fwd-input"

        self.cv_corpora = []
        self.devtrain_corpora = []

        self.train_input_data = (
            None
        )  # type:Optional[Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]]]
        self.cv_input_data = (
            None
        )  # type:Optional[Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]]]
        self.devtrain_input_data = (
            None
        )  # type:Optional[Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]]]
        self.dev_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]
        self.test_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]

        self.train_cv_pairing = None

        self.datasets = {}

        self.oggzips = {}  # TODO remove?
        self.hdfs = {}  # TODO remove?
        self.extern_rasrs = {}  # TODO remove?

        self.nn_configs = {}
        self.nn_checkpoints = {}
        self.tf_flows = {}

    def _add_output_alias_for_train_job(
        self,
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
        train_corpus_key: str,
        cv_corpus_key: str,
        name: str,
    ):
        train_job.add_alias(f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_train")
        self.jobs[f"{train_corpus_key}_{cv_corpus_key}"][name] = train_job
        self.nn_checkpoints[f"{train_corpus_key}_{cv_corpus_key}"][name] = train_job.out_checkpoints
        self.nn_configs[f"{train_corpus_key}_{cv_corpus_key}"][name] = train_job.out_returnn_config_file
        tk.register_output(
            f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    # -------------------- Setup --------------------
    def init_system(
        self,
        rasr_init_args: RasrInitArgs,
        train_data: Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]],
        cv_data: Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]],
        devtrain_data: Optional[Dict[str, Union[ReturnnRasrDataInput, AllowedReturnnTrainingDataInput]]] = None,
        dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        train_cv_pairing: Optional[List[Tuple[str, ...]]] = None,  # List[Tuple[trn_c, cv_c, name, dvtr_c]]
    ):
        self.rasr_init_args = rasr_init_args

        self._init_am(**self.rasr_init_args.am_args)

        devtrain_data = devtrain_data if devtrain_data is not None else {}
        dev_data = dev_data if dev_data is not None else {}
        test_data = test_data if test_data is not None else {}

        self._assert_corpus_name_unique(train_data, cv_data, devtrain_data, dev_data, test_data)

        self.train_input_data = train_data
        self.cv_input_data = cv_data
        self.devtrain_input_data = devtrain_data
        self.dev_input_data = dev_data
        self.test_input_data = test_data

        self.train_corpora.extend(list(train_data.keys()))
        self.cv_corpora.extend(list(cv_data.keys()))
        self.devtrain_corpora.extend(list(devtrain_data.keys()))
        self.dev_corpora.extend(list(dev_data.keys()))
        self.test_corpora.extend(list(test_data.keys()))

        self._set_eval_data(dev_data)
        self._set_eval_data(test_data)

        self.train_cv_pairing = (
            list(itertools.product(self.train_corpora, self.cv_corpora))
            if train_cv_pairing is None
            else train_cv_pairing
        )

        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]

            corpus_pair_name = f"{trn_c}_{cv_c}"

            self.jobs[corpus_pair_name] = {}
            self.nn_models[corpus_pair_name] = {}
            self.nn_checkpoints[corpus_pair_name] = {}
            self.nn_configs[corpus_pair_name] = {}

    def _set_eval_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.jobs[c_key] = {}
            self.ctm_files[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.crp[c_key].set_executables(rasr_binary_path=self.rasr_binary_path, rasr_arch=self.rasr_arch)
            self.feature_flows[c_key] = c_data.feature_flow
            self.feature_scorers[c_key] = {}
            if c_data.stm is not None:
                self.stm_files[c_key] = c_data.stm
            if c_data.glm is not None:
                self.glm_files[c_key] = c_data.glm

    def prepare_data(self, raw_sampling_rate: int, feature_sampling_rate: int):
        for name in self.train_corpora + self.devtrain_corpora + self.cv_corpora:
            self.jobs[name]["ogg_zip"] = j = returnn.BlissToOggZipJob(
                bliss_corpus=self.crp[name].corpus_config.corpus_file,
                segments=self.crp[name].segment_path,
                rasr_cache=self.feature_flows[name]["init"],
                raw_sample_rate=raw_sampling_rate,
                feat_sample_rate=feature_sampling_rate,
            )
            self.oggzips[name] = j.out_ogg_zip
            j.add_alias(f"oggzip/{name}")

            # TODO self.jobs[name]["hdf_full"] = j = returnn.ReturnnDumpHDFJob()

    def generate_lattices(self):
        pass

    # -------------------- Training --------------------

    def returnn_training(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        nn_train_args: Union[Dict, ReturnnTrainingJobArgs],
        train_corpus_key,
        cv_corpus_key,
        devtrain_corpus_key=None,
    ) -> returnn.ReturnnTrainingJob:
        if nn_train_args.returnn_root is None:
            nn_train_args.returnn_root = self.returnn_root
        if nn_train_args.returnn_python_exe is None:
            nn_train_args.returnn_python_exe = self.returnn_python_exe

        train_job = returnn_training(
            name=name,
            returnn_config=returnn_config,
            training_args=nn_train_args,
            train_data=self.train_input_data[train_corpus_key],
            cv_data=self.cv_input_data[cv_corpus_key],
            additional_data={"devtrain": self.devtrain_input_data[devtrain_corpus_key]}
            if devtrain_corpus_key is not None
            else None,
            register_output=False,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    def _get_feature_flow(self, feature_flow_key: str, data_input: ReturnnRasrDataInput):
        """
        Select the appropriate feature flow from the data input object.

        If no flows are defined, tries to create the flow based on the features
        cache directly

        :param feature_flow_key: key identifier, e.g. "gt" or "mfcc40" etc...
        :param data_input: Data input object containing the flows
        :return: training feature flow
        """
        if isinstance(data_input.feature_flow, Dict):
            feature_flow = data_input.feature_flow[feature_flow_key]
        elif isinstance(data_input.feature_flow, rasr.FlowNetwork):
            feature_flow = data_input.feature_flow
        else:
            if isinstance(data_input.features, rasr.FlagDependentFlowAttribute):
                feature_path = data_input.features
            elif isinstance(data_input.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": data_input.features,
                    },
                )
            elif isinstance(data_input.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": data_input.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(data_input.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        return feature_flow

    def returnn_rasr_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
        feature_flow_key: str = "gt",
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        train_crp.set_executables(rasr_binary_path=self.rasr_binary_path, rasr_arch=self.rasr_arch)
        dev_crp = dev_data.get_crp()
        dev_crp.set_executables(rasr_binary_path=self.rasr_binary_path, rasr_arch=self.rasr_arch)

        assert train_data.feature_flow == dev_data.feature_flow
        assert train_data.features == dev_data.features
        assert train_data.alignments == dev_data.alignments

        feature_flow = self._get_feature_flow(feature_flow_key, train_data)

        if isinstance(train_data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(train_data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(train_data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(train_data.alignments, tk.Path):
            alignments = train_data.alignments
        else:
            raise NotImplementedError

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnRasrTrainingJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flow=feature_flow,
            alignment=alignments,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    # -------------------- Recognition --------------------

    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.Checkpoint],
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
        prior_scales: List[float],
        pronunciation_scales: List[float],
        lm_scales: List[float],
        optimize_am_lm_scale: bool,
        recognition_corpus_key: str,
        feature_flow_key: str,
        search_parameters: Dict,
        lattice_to_ctm_kwargs: Dict,
        parallelize_conversion: bool,
        rtf: int,
        mem: int,
        epochs: Optional[List[int]] = None,
        use_epoch_for_compile=False,
        forward_output_layer="output",
        native_ops: Optional[List[str]] = None,
        acoustic_mixture_path: Optional[tk.Path] = None,
        **kwargs,
    ):
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            native_op_paths = self.get_native_ops(op_names=native_ops)

            tf_graph = None
            if not use_epoch_for_compile:
                tf_graph = self.nn_compile_graph(name, returnn_config)

            feature_flow = self.feature_flows[recognition_corpus_key]
            if isinstance(feature_flow, Dict):
                feature_flow = feature_flow[feature_flow_key]
            assert isinstance(
                feature_flow, rasr.FlowNetwork
            ), f"type incorrect: {recognition_corpus_key} {type(feature_flow)}"

            epochs = epochs if epochs is not None else list(checkpoints.keys())

            for pron, lm, prior, epoch in itertools.product(pronunciation_scales, lm_scales, prior_scales, epochs):
                assert epoch in checkpoints.keys()
                acoustic_mixture_path = CreateDummyMixturesJob(
                    num_mixtures=returnn_config.config["extern_data"]["classes"]["dim"],
                    num_features=returnn_config.config["extern_data"]["data"]["dim"],
                ).out_mixtures
                lmgc_scorer = rasr.GMMFeatureScorer(acoustic_mixture_path)
                prior_job = ReturnnComputePriorJobV2(
                    model_checkpoint=checkpoints[epoch],
                    returnn_config=train_job.returnn_config,
                    returnn_python_exe=train_job.returnn_python_exe,
                    returnn_root=train_job.returnn_root,
                    log_verbosity=train_job.returnn_config.post_config["log_verbosity"],
                )

                prior_job.add_alias("extract_nn_prior/" + name)
                prior_file = prior_job.out_prior_xml_file
                assert prior_file is not None
                scorer = rasr.PrecomputedHybridFeatureScorer(
                    prior_mixtures=acoustic_mixture_path,
                    priori_scale=prior,
                    prior_file=prior_file,
                )
                assert acoustic_mixture_path is not None

                if use_epoch_for_compile:
                    tf_graph = self.nn_compile_graph(name, returnn_config, epoch=epoch)

                tf_flow = make_precomputed_hybrid_tf_feature_flow(
                    tf_checkpoint=checkpoints[epoch],
                    tf_graph=tf_graph,
                    native_ops=native_op_paths,
                    output_layer_name=forward_output_layer,
                )
                flow = add_tf_flow_to_base_flow(feature_flow, tf_flow)

                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}"] = scorer
                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-tf-{epoch:03d}"] = flow

                recog_name = f"e{epoch:03d}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
                recog_func(
                    name=f"{name}-{recognition_corpus_key}-{recog_name}",
                    prefix=f"nn_recog/{name}/",
                    corpus=recognition_corpus_key,
                    flow=flow,
                    feature_scorer=scorer,
                    pronunciation_scale=pron,
                    lm_scale=lm,
                    search_parameters=search_parameters,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    parallelize_conversion=parallelize_conversion,
                    rtf=rtf,
                    mem=mem,
                    lmgc_alias=f"lmgc/{name}/{recognition_corpus_key}-{recog_name}",
                    lmgc_scorer=lmgc_scorer,
                    **kwargs,
                )

    def nn_recog(
        self,
        train_name: str,
        train_corpus_key: str,
        returnn_config: Path,
        checkpoints: Dict[int, returnn.Checkpoint],
        step_args: HybridArgs,
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
    ):
        for recog_name, recog_args in step_args.recognition_args.items():
            recog_args = copy.deepcopy(recog_args)
            whitelist = recog_args.pop("training_whitelist", None)
            if whitelist:
                if train_name not in whitelist:
                    continue
            for dev_c in self.dev_corpora:
                self.nn_recognition(
                    name=f"{train_corpus_key}-{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    train_job=train_job,
                    recognition_corpus_key=dev_c,
                    acoustic_mixture_path=None if isinstance(self.train_input_data[train_corpus_key], dict) else self.train_input_data[train_corpus_key].acoustic_mixtures, # fix for distill inputs
                    **recog_args,
                )

            for tst_c in self.test_corpora:
                r_args = copy.deepcopy(recog_args)
                if step_args.test_recognition_args is None or recog_name not in step_args.test_recognition_args.keys():
                    break
                r_args.update(step_args.test_recognition_args[recog_name])
                r_args["optimize_am_lm_scale"] = False
                self.nn_recognition(
                    name=f"{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    train_job=train_job,
                    recognition_corpus_key=tst_c,
                    acoustic_mixture_path=None if isinstance(self.train_input_data[train_corpus_key], dict) else self.train_input_data[train_corpus_key].acoustic_mixtures, # fix for distill inputs
                    **r_args,
                )

    def nn_compile_graph(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        epoch: Optional[int] = None,
    ):
        """
        graph compile helper including alias

        :param name: name for the alias
        :param returnn_config: ReturnnConfig that defines the graph
        :param epoch: optionally a specific epoch to compile when using
            e.g. `def get_network(epoch=...)` in the config
        :return: the TF graph
        """
        # TODO remove, temporary hack
        cfg = returnn_config
        if "pretrain" in cfg.config.keys():
            del cfg.config["pretrain"]
        graph_compile_job = returnn.CompileTFGraphJob(
            cfg,
            epoch=epoch,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
        )
        graph_compile_job.add_alias(f"nn_recog/graph/{name}")
        return graph_compile_job.out_graph

    # -------------------- Rescoring  --------------------

    def nn_rescoring(self):
        # TODO calls rescoring setup
        raise NotImplementedError

    # -------------------- run functions  --------------------

    def run_data_preparation_step(self, step_args):
        # TODO here be ogg zip generation for training or lattice generation for SDT
        raise NotImplementedError

    def run_nn_step(self, step_name: str, step_args: HybridArgs):
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = [pairing[2]] if len(pairing) >= 3 else list(step_args.returnn_training_configs.keys())
            dvtr_c_list = [pairing[3]] if len(pairing) >= 4 else self.devtrain_corpora
            dvtr_c_list = [None] if len(dvtr_c_list) == 0 else dvtr_c_list
            import time
            for name, dvtr_c in itertools.product(name_list, dvtr_c_list):
                start = time.time()
                if isinstance(self.train_input_data[trn_c], ReturnnRasrDataInput):
                    returnn_train_job = self.returnn_rasr_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                    )
                elif isinstance(self.train_input_data[trn_c], AllowedReturnnTrainingDataInput):
                    returnn_train_job = self.returnn_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                        devtrain_corpus_key=dvtr_c,
                    )
                else:
                    raise NotImplementedError

                returnn_recog_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                print(f"NN Train Iteration {time.time() - start}")
                start = time.time()
                self.nn_recog(
                    train_name=name,
                    train_corpus_key=trn_c,
                    returnn_config=returnn_recog_config,
                    checkpoints=returnn_train_job.out_checkpoints,
                    step_args=step_args,
                    train_job=returnn_train_job,
                )
                print(f"NN Recog Iteration {time.time() - start}")
                for recog_name, _ in step_args.recognition_args.items():
                    results = {}
                    from i6_core.report import GenerateReportStringJob, MailJob
                    for c in self.dev_corpora + self.test_corpora:
                        for job_name in self.jobs[c]:
                            if "scorer" not in job_name:
                                continue
                            if not name == job_name.split("-")[1]:
                                continue
                            if not f"{recog_name}-{c}" in job_name:  # e.g. "quant_multile-dev
                                continue
                            scorer = self.jobs[c][job_name]
                            if scorer.out_wer:
                                results[job_name] = scorer.out_wer
                    tk.register_report(f"reports/{name.replace('/', '_')}/{recog_name}", values=results)
                    quants = {}
                    for c in self.dev_corpora + self.test_corpora:
                        for job_name in self.jobs[c]:
                            #if "quantize_static" in job_name and "budget" in job_name and f"{recog_name}" in job_name:
                            if "quantize_static" in job_name and f"{recog_name}" in job_name:
                                quants[job_name] = self.jobs[c][job_name]
                    results["quant"] = quants
                    loss_tables = {}
                    for c in self.dev_corpora + self.test_corpora:
                        for job_name in self.jobs[c]:
                            if "calculate_loss" in job_name and f"{recog_name}" in job_name:
                                loss_tables[job_name] = self.jobs[c][job_name]
                    results["loss_tables"] = loss_tables
                    report = GenerateReportStringJob(report_values=results, report_template=hybrid_report_format)
                    report.add_alias(f"report/report_{name}_{recog_name}")
                    mail = MailJob(report.out_report, send_contents=True, subject=name + " " + recog_name)
                    mail.add_alias(f"report/mail_{name}_{recog_name}")
                    tk.register_output("mail/" + name + "_" + recog_name, mail.out_status)


    def run_nn_recog_step(self, step_args: NnRecogArgs):
        for eval_c in self.dev_corpora + self.test_corpora:
            self.nn_recognition(recognition_corpus_key=eval_c, **asdict(step_args))

    def run_rescoring_step(self, step_args):
        for dev_c in self.dev_corpora:
            raise NotImplementedError

        for tst_c in self.test_corpora:
            raise NotImplementedError

    def run_realign_step(self, step_args):
        for trn_c in self.train_corpora:
            for devtrv_c in self.devtrain_corpora[trn_c]:
                raise NotImplementedError
            for cv_c in self.cv_corpora[trn_c]:
                raise NotImplementedError

    def run_forced_align_step(self, step_args: NnForcedAlignArgs):
        for tc_key in step_args["target_corpus_keys"]:
            featurer_scorer_corpus_key = step_args["feature_scorer_corpus_key"]
            scorer_model_key = step_args["scorer_model_key"]
            epoch = step_args["epoch"]
            base_flow = self.feature_flows[tc_key][step_args["base_flow_key"]]
            tf_flow = self.tf_flows[step_args["tf_flow_key"]]
            feature_flow = self.add_tf_flow_to_base_flow(base_flow, tf_flow)

            self.forced_align(
                name=step_args["name"],
                target_corpus_key=tc_key,
                flow=feature_flow,
                feature_scorer_corpus_key=featurer_scorer_corpus_key,
                feature_scorer=scorer_model_key,
                dump_alignment=step_args["dump_alignment"],
            )

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            print("init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs")
            sys.exit(-1)

        self.prepare_scoring()

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    corpus_list = (
                        self.train_corpora
                        + self.cv_corpora
                        + self.devtrain_corpora
                        + self.dev_corpora
                        + self.test_corpora
                    )
                    step_args = self.rasr_init_args.feature_extraction_args
                else:
                    corpus_list = step_args.pop("corpus_list")

                for all_c in corpus_list:
                    if all_c not in self.feature_caches.keys():
                        self.feature_caches[all_c] = {}
                    if all_c not in self.feature_bundles.keys():
                        self.feature_bundles[all_c] = {}
                    if all_c not in self.feature_flows.keys():
                        self.feature_flows[all_c] = {}
                self.extract_features(step_args, corpus_list=corpus_list)

            # ---------- Prepare data ----------
            if step_name.startswith("data"):
                self.run_data_preparation_step(step_args)

            # ---------- NN Training ----------
            if step_name.startswith("nn"):
                import time
                start = time.time()
                self.run_nn_step(step_name, step_args)
                print(f"NN Time {time.time() - start}")

            if step_name.startswith("recog"):
                self.run_nn_recog_step(step_args)

            # ---------- Rescoring ----------
            if step_name.startswith("rescor"):
                self.run_rescoring_step(step_args)

            # ---------- Realign ----------
            if step_name.startswith("realign"):
                self.run_realign_step(step_args)

            # ---------- Forced Alignment ----------
            if step_name.startswith("forced") or step_name.startswith("align"):
                self.run_forced_align_step(step_args)
