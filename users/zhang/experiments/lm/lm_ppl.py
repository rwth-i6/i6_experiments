import copy
from typing import Union, Optional, List

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from returnn_common.datasets_old_2022_10.interface import VocabConfig
from sisyphus import *

from i6_core.returnn.config import ReturnnConfig
from i6_core.util import instanciate_delayed

from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.model_interfaces.model import ModelDef, ModelDefWithCfg, serialize_model_def
from i6_experiments.users.zeyer.datasets.librispeech import LibrispeechLmDataset

from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model
from i6_experiments.users.zeyer.recog import _v2_forward_out_filename
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from i6_experiments.users.zhang.datasets.vocab import GetSubwordRatioJob, ApplyBPEToTextJob
from i6_experiments.users.zhang.datasets.librispeech import get_test_corpus_text
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_lm_eval_text

from returnn_common import nn

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict


def lm_forward_def(*, model: rf.Module, targets: Tensor, targets_spatial_dim: Dim, same_seq:bool,**_other) -> Tensor:
    # noinspection PyTypeChecker
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    targets_w_bos, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)

    if same_seq: # Trafo lm
        logits, _ = model(
            targets_w_bos,
            spatial_dim=targets_w_eos_spatial_dim,
            state=model.default_initial_state(batch_dims=batch_dims),
        )
        #import pdb;pdb.set_trace()
        # logits, pack_dim = rf.pack_padded(
        #     logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        # )
        #  We need to mask out the padding..? Yeah, but before feeds to the model..? Isnt it already specified in w_eos_spatial_dim?
        # Transformerdecoder seems already use causal attention by default.


    else:
        logits, _ = model(
            targets,
            spatial_dim=targets_spatial_dim,
            out_spatial_dim=targets_w_eos_spatial_dim,
            state=model.default_initial_state(batch_dims=batch_dims),
        )
    # import pdb; pdb.set_trace()
    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(log_prob, indices=targets_w_eos, axis=model.vocab_dim) # Why before it is indices = targets_w_eos?
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)

    return targets_w_eos, log_prob_targets_seq


def _returnn_forward_step(*, model: rf.Module, extern_data: TensorDict, **kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    target = extern_data[default_target_key]
    data_spatial_dim = data.get_time_dim_tag()
    forward_def = config.typed_value("_forward_def")
    forward_out = forward_def(model=model, targets=target, targets_spatial_dim=data_spatial_dim)

    hyps, scores = forward_out
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, hyps.dims[1]])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim])


def _returnn_forward_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.data = {}

        def init(self, *, model):
            pass

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            hyps: Tensor = outputs["hyps"]  # [out_spatial]
            scores: Tensor = outputs["scores"]  # []
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[0].dyn_size_ext is not None, f"hyps {hyps} do not define seq lengths"
            hyps_len = hyps.dims[0].dyn_size_ext
            self.data[seq_tag] = [hyps_len.raw_tensor.item(), scores.raw_tensor.item()]
            import torch
            torch.cuda.empty_cache()

        def finish(self):
            import json
            import gzip

            out_file_fp = gzip.open(_v2_forward_out_filename, "wt", encoding="utf-8")
            json.dump(self.data, out_file_fp)

    return _ReturnnRecogV2ForwardCallbackIface()


def _returnn_ppl_config(model_def: ModelDef, dataset: LibrispeechLmDataset, dataset_key: str, same_seq:bool=False, batch_size:int=80_000) -> ReturnnConfig:
    from i6_experiments.users.zeyer.utils.sis_setup import get_base_module

    unhashed_package_root_model_def, setup_base_name_model_def = get_base_module(
        model_def.model_def if isinstance(model_def, ModelDefWithCfg) else model_def
    )

    returnn_config = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=dataset.get_dataset(dataset_key),
        batch_size=batch_size,#100_000,
    )

    if isinstance(model_def, ModelDefWithCfg):
        # TODO:
        #  somehow the activation func is not being serialized correctly for forwarding but works fine in training
        #  for now we will just convert it to a dict

        assert "_model_def_dict" in model_def.config
        model_def_config = copy.deepcopy(model_def.config)
        if "activation_func" in model_def_config["_model_def_dict"] and not isinstance(
            model_def_config["_model_def_dict"]["activation_func"], dict
        ):
            model_def_config["_model_def_dict"]["activation_func"] = rf.build_dict(
                model_def_config["_model_def_dict"]["activation_func"]
            )

        returnn_config = dict_update_deep(returnn_config, model_def_config)

    extern_data_raw = instanciate_delayed(dataset.get_extern_data())

    serial_collection = [
        serialization.NonhashedCode(get_import_py_code()),
        serialization.NonhashedCode(
            nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
        ),
        *serialize_model_def(model_def, unhashed_package_root=unhashed_package_root_model_def),
        serialization.Import(_returnn_v2_get_model, import_as="get_model"),
        serialization.PartialImport(
            code_object_path=lm_forward_def,
            unhashed_package_root=None,
            hashed_arguments={"same_seq": same_seq},
            unhashed_arguments={},
            import_as="_forward_def",
            ignore_import_as_for_hash=True),
        #serialization.Import(lm_forward_def, import_as="_forward_def", ignore_import_as_for_hash=True),
        serialization.Import(_returnn_forward_step, import_as="forward_step"),
        serialization.Import(_returnn_forward_callback, import_as="forward_callback"),
        serialization.ExplicitHash({"version": "5"}),
        serialization.PythonEnlargeStackWorkaroundNonhashedCode,
        serialization.PythonCacheManagerFunctionNonhashedCode,
        serialization.PythonModelineNonhashedCode,
    ]

    returnn_config = ReturnnConfig(config=returnn_config, python_epilog=serialization.Collection(serial_collection))

    return returnn_config


def compute_ppl(*, prefix_name, model_with_checkpoints, dataset, dataset_keys: Union[str, List[str]], epochs:List[int]=[], word_ppl: bool = False, same_seq:bool=False, batch_size:int=80_000, vocab:[str | VocabConfig] = "bpe128", task_name:str = "LBS",**kwargs_unused):
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths
    if isinstance(dataset_keys, str):
        dataset_keys = [dataset_keys]
    ppls = dict()
    for epoch in model_with_checkpoints.fixed_epochs:
        ppls[f"epoch{epoch}"] = dict()
    for dataset_key in dataset_keys:
        if task_name == "LBS":
            text_data = get_test_corpus_text(keys=[dataset_key[len("transcriptions-") :] if dataset_key.startswith("transcriptions-") else dataset_key])
        elif task_name == "ES":
            text_data = get_lm_eval_text(key = dataset_key)
        else:
            raise ValueError("task_name must be 'LBS' or 'ES'")
        if isinstance(vocab, SentencePieceModel) or "spm" in vocab:
            apply_job = ApplySentencepieceToTextJob
        elif isinstance(vocab, Bpe) or "bpe" in vocab:
            apply_job = ApplyBPEToTextJob
        else:
            raise ValueError("vocab must be SentencePieceModel or SentencePieceModel")
        ratio = GetSubwordRatioJob(text_data, vocab, get_returnn_subword_nmt(),apply_job=apply_job).out_ratio
        tk.register_output(f"{task_name}_{dataset_key}_{vocab}_ratio", ratio)
        ratio = 1.0 if not word_ppl else ratio

        returnn_config = _returnn_ppl_config(model_with_checkpoints.definition, dataset, dataset_key, same_seq, batch_size=batch_size)

        for epoch in model_with_checkpoints.fixed_epochs:
            if len(epochs) > 0:
                if epoch not in epochs:
                    continue
            res = ReturnnForwardJobV2(
                model_checkpoint=model_with_checkpoints.get_epoch(epoch).checkpoint,
                returnn_config=returnn_config,
                output_files=[_v2_forward_out_filename],
                returnn_python_exe=tools_paths.get_returnn_python_exe(),
                returnn_root=tools_paths.get_returnn_root(),
            )
            if kwargs_unused.get("rqmt",None):
                res.rqmt.update(kwargs_unused["rqmt"])
            # print(f"Exponent:{exponent.get()}")
            ppl_job = ComputePerplexityJob(scores_and_lens_file=res.out_files[_v2_forward_out_filename],exponent=ratio)
            dataset_key_ = (
                dataset_key[len("transcriptions-") :] if dataset_key.startswith("transcriptions-") else dataset_key
            ) if task_name == "LBS" else dataset_key
            #print(f"Will compute ppl:ppl/{prefix_name}/{epoch}/{dataset_key_}_ppl")
            res.add_alias(f"ppl/{prefix_name}/{epoch}/{dataset_key_}_ppl")
            tk.register_output(f"ppl/{prefix_name}/{epoch}/{dataset_key_}_ppl", ppl_job.out_ppl)
            ppls[f"epoch{epoch}"][dataset_key_] = ppl_job.out_ppl
            # if prefix_name == "ES/trafo-n32-d1280-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k":
            #     print(f"Add PPLs on {dataset_key_} for epoch {epoch}:\n -> {ppls}\n")
            # if dataset_key_ == "test-other":
            #     ppls[f"epoch{epoch}"]["test-other"] = ppl_job.out_ppl

    return ppls

def compute_ppl_single_epoch(*, prefix_name, model_with_checkpoint, epoch, dataset, dataset_keys: Union[str, List[str]], word_ppl: bool = False, same_seq:bool=False, batch_size:int=80_000, vocab: [str | VocabConfig] = "bpe128", task_name: str = "LBS", **kwargs_unused):
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths

    if isinstance(dataset_keys, str):
        dataset_keys = [dataset_keys]
    ppls = dict()
    for dataset_key in dataset_keys:
        if task_name == "LBS":
            text_data = get_test_corpus_text(keys=[dataset_key[len("transcriptions-") :] if dataset_key.startswith("transcriptions-") else dataset_key])
        elif task_name == "ES":
            text_data = get_lm_eval_text(key = dataset_key)
        else:
            raise ValueError("task_name must be 'LBS' or 'ES'")
        if isinstance(vocab, SentencePieceModel) or "spm" in vocab:
            apply_job = ApplySentencepieceToTextJob
        elif isinstance(vocab, Bpe) or "bpe" in vocab:
            apply_job = ApplyBPEToTextJob
        else:
            raise ValueError("vocab must be SentencePieceModel or SentencePieceModel")
        ratio = GetSubwordRatioJob(text_data, vocab, get_returnn_subword_nmt(),apply_job=apply_job).out_ratio
        tk.register_output(f"{task_name}_{dataset_key}_{vocab}_ratio", ratio)
        ratio = 1.0 if not word_ppl else ratio

        returnn_config = _returnn_ppl_config(model_with_checkpoint.definition, dataset, dataset_key, same_seq, batch_size=batch_size)

        res = ReturnnForwardJobV2(
            model_checkpoint=model_with_checkpoint.checkpoint,
            returnn_config=returnn_config,
            output_files=[_v2_forward_out_filename],
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
        )
        if kwargs_unused.get("rqmt", None):
            res.rqmt.update(kwargs_unused["rqmt"])
        ppl_job = ComputePerplexityJob(scores_and_lens_file=res.out_files[_v2_forward_out_filename], exponent=ratio)

        dataset_key_ = (
            dataset_key[len("transcriptions-"):] if dataset_key.startswith("transcriptions-") else dataset_key
        )
        exponent_raw = ratio.get() if isinstance(ratio, tk.Variable) else float(ratio)
        res.add_alias(f"ppl/{prefix_name}/{epoch}/{dataset_key_}_ppl")
        tk.register_output(f"ppl/{prefix_name}/{epoch}/{dataset_key_}_{'word' if exponent_raw > 1 else 'subword'}_ppl", ppl_job.out_ppl)
        ppls[dataset_key_] = ppl_job.out_ppl
    return ppls

class ComputePerplexityJob(Job):
    def __init__(self, scores_and_lens_file: Optional[Path], exponent:Union[float,tk.Variable] = 1.0, version:int=1):
        self.scores_and_lens_file = scores_and_lens_file

        self.out_ppl = self.output_path("ppl")
        self.exponent = exponent.get() if isinstance(exponent, tk.Variable) else exponent

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        import math
        import json

        fpath = self.scores_and_lens_file.get_path()
        if fpath.endswith(".gz"):
            import gzip

            open_func = gzip.open
        else:
            open_func = open

        with open_func(fpath, "rt") as f:
            d = json.load(f)

        scores = 0.0
        lens = 0
        for v in d.values():
            scores += v[1]
            lens += v[0]

        ppl = math.exp(-1.0 * scores / lens)

        with open(self.out_ppl.get_path(), "w+") as f:
            if self.exponent > 1:
                f.write("Original level: %f ratio: %f \n" % (ppl, self.exponent))
                print(f"ori_ppl:{ppl}, exponent:{self.exponent}")
                ppl = math.pow(ppl, self.exponent)
            f.write("Perplexity: %f" % ppl)
