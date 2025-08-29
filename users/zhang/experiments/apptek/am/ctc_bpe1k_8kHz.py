import torch
import copy
import os
from i6_experiments.users.wu.systems.functors.rasr_base import RecognitionScoringType
from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from sisyphus import gs, tk, Job, Task

import i6_core.rasr as rasr
from i6_core.lib.lexicon import Lexicon, Lemma
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs, SummaryKey
from i6_experiments.users.wu.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from apptek_asr.users.hwu.corpus.ES_f8kHz.ctc_data import get_es_8khz_data_bpe
from apptek_asr.meta.nn import NNSystemV2
from apptek_asr.meta.rasr import PyrasrVenvBuilder, BpeRasrFsaExporterConfigBuilder
from apptek_asr.meta.state_tying import compute_num_states
from apptek_asr.recognition.scoring import ApptekScorer
from apptek_asr.artefacts import ArtefactSpecification
from apptek_asr.artefacts.factory import AbstractArtefactRepository
from i6_experiments.users.wu.util import pyrasr_tools
#from i6_experiments.users.wu.util import seq_concat_pyrasr_tools


from i6_core.lib.corpus import Corpus, Recording, Segment, lexicon
class FilterCorpusByDurationTargetsRatioJob(Job):
    """
    Filter segments based on time/words ratio
    """

    def __init__(self, bliss_corpus, lexicon_path, frame_rate=0.04, compressed=True):
        super().__init__()
        self.bliss_corpus = bliss_corpus
        self.frame_rate = frame_rate
        self.lexicon_path = lexicon_path

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if compressed else ""))
        self.out_segment_list = self.output_path("segment_list")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        def filter_by_ratio(corpus: Corpus, recording: Recording, segment: Segment) -> bool:
            import math
            """
            returns True if T > S, T is number of feature frames, S is number of target tokens
            """
            seg_duration = segment.end - segment.start
            target_indices = []
            for word in segment.orth.strip().split():
                target_indices += self.lookup_dict[word]
            valid = math.floor(seg_duration / self.frame_rate) > len(target_indices)
            if not valid:
                print(seg_duration)
                print(segment.orth)
                print(len(target_indices))
            return valid

        lex = lexicon.Lexicon()
        lex.load(self.lexicon_path)
        # Mapping from phoneme symbol to target index. E.g. {"[SILENCE]": 0, "a": 1, "b": 2, ...}
        phoneme_indices = dict(zip(lex.phonemes.keys(), range(len(lex.phonemes))))

        # build lookup dict of word to target sequence
        self.lookup_dict = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if not orth:
                    continue
                if len(lemma.phon) > 0:
                    phon = lemma.phon[0]
                else:
                    phon = ""
                self.lookup_dict[orth] = [phoneme_indices[p] for p in phon.split()]

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        c.filter_segments(filter_by_ratio)
        segment_list = []
        for seg in c.segments():
            segment_list.append(seg.fullname())
        c.dump(self.out_corpus.get_path())
        with open(self.out_segment_list.get(), "w") as f:
            for seg in segment_list:
                f.write(seg + "\n")
# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

# Atanas BPE setup
bpe_size = 1000
num_outputs = 1104
tools = copy.deepcopy(pyrasr_tools)


def returnn_config_generator(
    variant: ConfigVariant, num_subepochs: int, **kwargs
) -> ReturnnConfig:
    # ********** Return Config generators **********
    ### General ###
    # ISO language code of the system (e.g. "EN_US")
    # used to determine the namespace of most artefacts
    lang_code_es = "ES"
    lang_code_es_us = "ES_US"
    lang_code_es_es = "ES_ES"

    # Frequency of the audio data: either "16kHz" or "8kHz"
    sampling_rate = "8kHz"
    sampling_rate_int = int(sampling_rate[:-len("kHz")]) * 1000

    # Name for the experiment. This will name the output folder
    experiment_name = "CTC-40ms-BPE-nativeLoss"

    ### Training ###
    # Specify lists of training HDF artefact names from the appropriate namespace here
    # as Dict[str, List[str]]. These will be concatenated and used for training
    #
    # feature + alignment pairs can either be stored separately in two different hdf
    # files (then set both train_corpora_alignment and train_corpora_features) or
    # together in a single file (in this case only set train_corpora_alignment).
    feature_hdf_ns_es = f"hdf.features.{lang_code_es}.f{sampling_rate}"
    feature_hdf_ns_es_us = f"hdf.features.{lang_code_es_us}.f{sampling_rate}"
    feature_hdf_ns_es_es = f"hdf.features.{lang_code_es_es}.f{sampling_rate}"

    train_corpora_features = {
        feature_hdf_ns_es: [
            f"{corpus_name}-batch.{batch_name}-hdf-raw_wav.{sampling_rate}.split-{split}"
            for corpus_name, batch_name, split in [("Fisher", "1.train.v2", 17)]
        ],
        feature_hdf_ns_es_es: [
            f"{corpus_name}-batch.{batch_name}-hdf-raw_wav.{sampling_rate}.split-{split}"
            for corpus_name, batch_name, split in [("Appen", "1.v1", 10)]
        ],
        feature_hdf_ns_es_us: [
            f"{corpus_name}-batch.{batch_name}-hdf-raw_wav.{sampling_rate}.split-{split}"
            for corpus_name, batch_name, split in [
                ("AAA", "1.v1", 1),
                ("CallFriend", "1.v1", 5),
                ("CollinCounty", "1.v1", 9),
                ("Ignite.ATNT", "1.v1", 6),
                ("Ignite.HomeShopping", "1.v1", 32),
                ("ListenTrust", "1.v1", 40),
                ("ListenTrust", "2.v1", 40),
                ("ListenTrust", "3.capital.v2", 3),
                ("ListenTrust", "4.funeral.v2", 1),
                ("ListenTrust", "5.immigration.v2", 4),
                ("ListenTrust", "6.real.v2", 26),
                ("NameAddr", "1.v1", 1),
                ("CallHome", "1.train.v1", 1),
            ]
        ],
    }
    dev_hdf_artefact_specs = ArtefactSpecification(
        feature_hdf_ns_es,
        f"Fisher-batch.2.dev.v2-hdf-raw_wav.{sampling_rate}.split-1",
    )

    # collection of bliss corpus files to be used for fsa training
    corpus_ns_es = f"corpus.{lang_code_es}.f{sampling_rate}"
    corpus_ns_es_us = f"corpus.{lang_code_es_us}.f{sampling_rate}"
    corpus_ns_es_es = f"corpus.{lang_code_es_es}.f{sampling_rate}"
    corpora_bliss = {
        corpus_ns_es: [
            f"{corpus_name}-batch.{batch_name}"
            for corpus_name, batch_name, split in [
                ("Fisher", "1.train.v2", 17),
                ("Fisher", "2.dev.v2", 1),  # only for dev
            ]
        ],
        corpus_ns_es_es: [
            f"{corpus_name}-batch.{batch_name}"
            for corpus_name, batch_name, split in [
                ("Appen", "1.v1", 10)
            ]
        ],
        corpus_ns_es_us: [
            f"{corpus_name}-batch.{batch_name}"
            for corpus_name, batch_name, split in [
                ("AAA", "1.v1", 1),
                ("CallFriend", "1.v1", 5),
                ("CollinCounty", "1.v2", 9),
                ("Ignite.ATNT", "1.v2", 6),
                ("Ignite.HomeShopping", "1.v2", 32),
                ("ListenTrust", "1.v2", 40),
                ("ListenTrust", "2.v2", 40),
                ("ListenTrust", "3.capital.v3", 3),
                ("ListenTrust", "4.funeral.v3", 1),
                ("ListenTrust", "5.immigration.v3", 4),
                ("ListenTrust", "6.real.v3", 26),
                ("NameAddr", "1.v1", 1),
                ("CallHome", "1.train.v1", 1),
            ]
        ],
    }
    # Total data: 250h

    # BPE codes and vocab
    bpe_ns = "subword_units.subword_nmt.ES"
    bpe_name = "2024-12-bpe_1102-telephony-research_task"

    # subword nmt
    subword_nmt_ns = "software.subword_nmt"
    subword_nmt_name = "2023-03-subword-nmt-returnn"

    fsa_topology_config = rasr.RasrConfig()
    fsa_topology_config.lib_rasr.alignment_fsa_exporter.allophone_state_graph_builder.topology="ctc"
    fsa_topology_config.lib_rasr.alignment_fsa_exporter.model_combination.acoustic_model.phonology.history_length=0
    fsa_topology_config.lib_rasr.alignment_fsa_exporter.model_combination.acoustic_model.phonology.future_length=0
    fsa_exporter_kwargs = {"silence_phoneme": "", "extra_fsa_exporter_post_config": fsa_topology_config}

    # Extra options for the HDF dataset.
    # Relevant: partition_epoch splits one epoch into N subepochs.
    #           This generates checkpoints faster but has implications for LR scheduling.
    dataset_extra_opts = {
        "partition_epoch": 40,
        "local_caching": True,
        "num_workers": 1,
        "data_dtype": "int16",
        "cv_hdf_artefact_specs": {"dev": dev_hdf_artefact_specs},
    }
    if kwargs.get("seq_concat", False):
        data_postprocessing_ns = "returnn_config.data_postprocessing"
        data_postprocessing_name = "concat-seqs-uniform-3-laplace-1000"
        data_postprocessing_kwargs = {
            "max_seq_len": kwargs.get("max_seq_len", 30) * 8000,
            "max_num_seqs": 2,
            "classes_key": None,
            "num_features_per_class": None,
        }
        dataset_extra_opts.update(
            {
                "postprocessing_artefact_spec": ArtefactSpecification(
                    data_postprocessing_ns, data_postprocessing_name, **data_postprocessing_kwargs
                )
            }
        )
    else:
        dataset_extra_opts["train_set_sorting"] = "laplace:.1000"  # if seq concat, use random sort in dataset, sorting handled by Postprocessing


    # Network config artefact.
    # Most conformer nets are too deep an need a special python_prolog:
    # "config_params": {"python_prolog": "import sys\nsys.setrecursionlimit(10000)"},
    network_config_ns = "network_config.pt_conformer_rel_pos"
    network_config_name = "2024-09-conformer_rel_pos-ctc-1102_bpe-8kHz_logmel"
    network_config_kwargs = {"num_heads": kwargs.get("num_heads", 8), "num_outputs": num_outputs}

    # training returnn config components
    training_config_ns = "returnn_config.training"
    training_config_name = "modular-training-pt-600-v1"
    training_config_kwargs = {
        "num_epochs": num_subepochs,
        "training_args": {
            "mem_rqmt": 24,
            "horovod_num_processes": 4,  # num gpus to use
            "distributed_launch_cmd": "torchrun",
        },
        "config_params": {
            "config": {
                "use_horovod": False,
                "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
                "torch_dataloader_opts": {"num_workers": 1},
                # important for usage of the simple HDFDataset
                "cache_size": "0",
                "torch_log_memory_usage": True,
                "max_seq_length": {"data": kwargs.get("max_seq_len", 30) * 8000},  # 30s covers 99% seqs
            },
        },
    }
    if kwargs.get("use_amp", True):
        training_config_kwargs["config_params"]["config"].update(
            {"torch_amp": {"dtype": "bfloat16", "grad_scaler": None},}
        )

    returnn_config_batch_size_ns = "returnn_config.batch_size"
    returnn_config_batch_size_name = "raw-wav-8kHz-batch_size-2080"
    returnn_config_batch_size_kwargs = {"batch_size": kwargs.get("batch_size", 60_000)}

    returnn_config_learning_rates_ns = "returnn_config.learning_rates"
    returnn_config_learning_rates_name = "oclr-classic-300"
    returnn_config_learning_rates_kwargs = {
        "training_num_epochs": num_subepochs,  # portion of oclr phase given by ramp_epochs, training_num_epochs scales this up
        "learning_rate_points": [kwargs.get("initial_lr", 3e-5), kwargs.get("peak_lr", 1e-3), kwargs.get("decay_lr", 3e-5), kwargs.get("final_lr", kwargs.get("decay_lr", 3e-5)/10)],
    }

    # Training chunking setting (not compatible with fullsum yet)
    returnn_config_chunking_ns = "returnn_config.chunking"
    returnn_config_chunking_name = "no-chunking"
    returnn_config_chunking_kwargs = {}

    returnn_config_optimizer_ns = "returnn_config.optimizer"
    returnn_config_optimizer_name = "adam"
    returnn_config_optimizer_kwargs = {"class": "nadam", "weight_decay": kwargs.get("l2", 0)}  # in torch, l2 is implemented as weight decay


    # Torch train step function
    train_step_func_ns = "returnn_config.pytorch"
    train_step_func_name = "fullsum-max-likelihood-loss"
    train_step_func_kwargs = {"label_posterior_scale": 1.0, "transition_scale": 1.0, "split_batch_on_oom": False}

    # acoustic_model_config artefact
    training_acoustic_model_config_ns = "am_config"
    training_acoustic_model_config_name = "e2e-monophone-am-config-v1"
    training_acoustic_model_config_kwargs = {}#{"allophone_history_future_length": (0, 0)}

    ### Recognition ###
    # prior computation torch forward step function
    prior_forward_step_func_ns = "returnn_config.pytorch"
    prior_forward_step_func_name = "recognition-log-softmax"
    # prior computation returnn callback function
    prior_callback_ns = "returnn_config.pytorch"
    prior_callback_name = "prior-callback"
    # recognition torch forward step function
    forward_step_func_ns = "returnn_config.pytorch"
    forward_step_func_name = "recognition-log-softmax"


    ### Software ###
    # RASR artefact and the runtime used to run RASR
    # usually the most recent version available should be used
    asrmon_name = "asrmon-2024-10-17"
    rasr_name = "streaming-rasr-2024-11-06"
    returnn_name = "returnn-2024-09-10"
    runtime_name = "ApptekCluster-ubuntu2204-tf2.15.1-pt2.3.0-2024-04-24"
    sctk_name = "sctk-2022-09-08"

    aar = AbstractArtefactRepository()
    
    navigation = ["config_params", "post_config", "cleanup_old_models"]
    nested_dict = training_config_kwargs
    for param in navigation:
       nested_dict.setdefault(param, {})
       nested_dict = nested_dict[param]
    nested_dict["keep"] = sorted(
       list(
           set(
               nested_dict.get("keep", [])
               + [50, 100, 250, 400, 450, 500] 
           )
       )
    )
    
    training_args, training_config = (
        ArtefactSpecification(
            training_config_ns, training_config_name, **training_config_kwargs
        )
        .get_factory(aar)
        .build()
    )

    training_config.update(
        ArtefactSpecification(
            network_config_ns, network_config_name, **network_config_kwargs
        )
        .get_factory(aar)
        .build()
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_batch_size_ns,
            returnn_config_batch_size_name,
            **returnn_config_batch_size_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_chunking_ns,
            returnn_config_chunking_name,
            **returnn_config_chunking_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_learning_rates_ns,
            returnn_config_learning_rates_name,
            **returnn_config_learning_rates_kwargs,
        ).build(aar)
    )
    training_config.update(
        ArtefactSpecification(
            returnn_config_optimizer_ns, 
            returnn_config_optimizer_name,
            optimizer_kwargs=returnn_config_optimizer_kwargs,
            gradient_clip_value=50,  # 2 seems too small
        ).build(aar)
    )

    prior_config = copy.deepcopy(training_config)
    prior_config.config.pop("torch_distributed", None)
    prior_config.update(ArtefactSpecification(prior_forward_step_func_ns, prior_forward_step_func_name).build(aar))
    prior_config.update(ArtefactSpecification(prior_callback_ns, prior_callback_name).build(aar))

    recog_network_config_kwargs = copy.deepcopy(network_config_kwargs)
    recog_network_config_kwargs.update({"mode": "recognition"})
    recognition_config = ArtefactSpecification(
        network_config_ns, network_config_name, **recog_network_config_kwargs
    ).build(aar)
    recognition_config.update(ArtefactSpecification(forward_step_func_ns, forward_step_func_name).build(aar))


    returnn_root = aar.get_artefact_factory("software.returnn", returnn_name).build()["returnn_root"]
    returnn_exe = tk.Path("/usr/bin/python3")

    artefacts = {
        "runtime_spec": ArtefactSpecification("runtime", runtime_name),
        "rasr_spec": ArtefactSpecification("rasr", rasr_name),
    }
    
    # Prepare specific SCTK version used for scoring
    sctk = aar.get_artefact_factory("software.sctk", sctk_name).build()
    scorer_kwargs = {"sctk_binary_path": sctk["sctk_binary_path"]}

    feature_hdf_artefact_specs = {
        hdf_ns_ + corpus: ArtefactSpecification(hdf_ns_, corpus)
        for hdf_ns_, corpora_ in train_corpora_features.items()
        for corpus in corpora_
    }

    # Prepare training dataset
    dataset_opts = {
        "dataset_type": "HDFDataset",
        "hdf_artefact_specs": feature_hdf_artefact_specs,
        "has_classes": False,
        **dataset_extra_opts,
    }

    corpus_specs = {
        f"{ns}.{name}": ArtefactSpecification(ns, name)
        for ns, corpus_list in corpora_bliss.items()
        for name in corpus_list
    }
    bpe_spec = ArtefactSpecification(bpe_ns, bpe_name)
    subword_nmt_spec = ArtefactSpecification(subword_nmt_ns, subword_nmt_name)
    training_am_config_spec = ArtefactSpecification(
        training_acoustic_model_config_ns, training_acoustic_model_config_name, **training_acoustic_model_config_kwargs
    )

    nonword_lex = Lexicon()
    for phon in ["[SILENCE]", "[NOISE]", "[MUSIC]"]:
        nonword_lex.add_phoneme(phon, variation="none")
    nonword_lex.add_lemma(Lemma(orth=["<blank>"], phon=["[SILENCE]"], special="blank"))
    nonword_lex.add_lemma(Lemma(orth=["[silence]"], phon=["[SILENCE]"], special="silence"))
    nonword_lex.add_lemma(Lemma(orth=["[noise]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(Lemma(orth=["[vocalized-noise]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(Lemma(orth=["[vocalized-unknown]"], phon=["[NOISE]"]))
    nonword_lex.add_lemma(Lemma(orth=["[unknown]"], phon=["[NOISE]", "[MUSIC]"], special="unknown"))


    fsa_exporter_opts = {
        "corpus_specs": corpus_specs,
        "bpe_spec": bpe_spec,
        "subword_nmt_spec": subword_nmt_spec,
        "am_config_spec": training_am_config_spec,
        "meta_lexicon_extra_kwargs": {"nonword_lex": nonword_lex,},
        **fsa_exporter_kwargs,
    }
    fsa_exporter_config_builder = BpeRasrFsaExporterConfigBuilder(aar=aar, **fsa_exporter_opts)
    fsa_exporter_config_path = fsa_exporter_config_builder.build()
    lexicon_path = fsa_exporter_config_builder.corpus_merger.out["lexicon"].get_path()
    
    segment_list = FilterCorpusByDurationTargetsRatioJob(
        fsa_exporter_config_builder.corpus_merger.out["filtered_corpus"],
        lexicon_path,
    ).out_segment_list if kwargs.get("filter_invalid_seq", False) else None
    if segment_list is not None:
        dataset_opts["segment_list"] = segment_list#fsa_exporter_config_builder.get_filtered_segments(segment_list)
    else:
        dataset_opts["segment_list"] = fsa_exporter_config_builder.out_segments

    py_rasr_arg = copy.deepcopy(artefacts)
    pyrasr_venv_builder = PyrasrVenvBuilder(aar, **py_rasr_arg)
    returnn_exe = pyrasr_venv_builder.build(tk.Path("/usr/bin/python3"))
    rasr_exe = pyrasr_venv_builder.jobs["compile_rasr"]
    
    train_step_config = ArtefactSpecification(
        train_step_func_ns,
        train_step_func_name,
        fsa_exporter_config_path=fsa_exporter_config_path,
        **train_step_func_kwargs,
    ).build(aar)

    training_config.update(train_step_config)

    # Run training
    nn_system = NNSystemV2(
        aar,
        model_type=NNSystemV2.ModelType.onnx_from_pt,
        training_args=training_args,
        training_config=training_config,
        recognition_config=recognition_config,
        prior_config=prior_config,
        dataset_options=dataset_opts,
        exp_name=experiment_name,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        **artefacts,
    )
    nn_system.set_data_config()

    if variant == ConfigVariant.RECOG:
        recognition_config = copy.deepcopy(nn_system.recognition_config)
        recognition_config.config["extern_data"] = {
            "data": {"dim": 60, "dtype": "float32"},
        }
        recognition_config.config["model_outputs"] = {
            "output": {
                "dim": num_outputs,
            }
        }
        if kwargs.get("use_amp", True):
            recognition_config.config["torch_amp_options"] = {"dtype": "bfloat16"}
        return recognition_config
    if variant == ConfigVariant.PRIOR:
        prior_config = copy.deepcopy(nn_system.prior_config)
        prior_config.config["model_outputs"] = {
            "output": {
                "dim": num_outputs,
            }
        }
        if kwargs.get("seq_concat", False):
            prior_config.config["forward_data"] = prior_config.config["forward_data"]["dataset"]
            prior_config.config["train"] = prior_config.config["train"]["dataset"]
        #prior_config.config["batch_size"] = 20_000 * 80 # run comfortably on 11-g gpu 
        del prior_config.config["max_seq_length"]
        return prior_config
    return nn_system.returnn_config

def get_returnn_config_collection(
    num_subepochs: int,
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            variant=ConfigVariant.TRAIN,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        prior_config=returnn_config_generator(
            variant=ConfigVariant.PRIOR,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        recog_configs={
            "recog": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                num_subepochs=num_subepochs,
                **kwargs,
            )
        },
    )


def run_exp():
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    summary_reports = []
    for initial_lr, peak_lr, decay_lr, final_lr in [(6e-5, 2e-3, 6e-5, 6e-6)]: 
        for l2 in [5e-6]:
            use_amp = True
            seq_concat = False
            filter_invalid_seq = False
            max_seq_len = 30
            # now needed only for recognition
            data = get_es_8khz_data_bpe(
                bpe_size=bpe_size,
                target_size=num_outputs,
                returnn_root=tools.returnn_root,
                returnn_python_exe=tools.returnn_python_exe,
                rasr_binary_path=tools.rasr_binary_path,
                feature_type=FeatureType.SAMPLES,
                #segmenter_settings=["apptek-default-seg"],
                #segmenter_settings=["zoltan-e2e-seg"],
                segmenter_settings=["ref-seg"],
                recog_lexicon=tk.Path("/home/hwu/setups/2024-07-15_torch_transducer_es_8kHz/output/restricted_lexicon_1104_noSynt.xml"),
                align_lexicon=tk.Path("/home/hwu/setups/2024-07-15_torch_transducer_es_8kHz/work/apptek_asr/lexicon/validate_lexicon/ValidateLexiconJob.YbNFaicqGQiq/output/validated_lexicon.xml.gz"),
            )

            for data_input in data.data_inputs.values():
                data_input.create_lm_images(tools.rasr_binary_path)

            # ********** Step args **********
            train_args = exp_args.get_ctc_train_step_args(
                num_epochs=500, horovod_num_processes=4, distributed_launch_cmd="torchrun"
            )
            recog_args = exp_args.get_ctc_recog_step_args(
                num_classes=num_outputs,
                epochs=[50, 100, 250, 400, 450, 500],
                reduction_factor=4,
                reduction_subtrahend=3, # floor pooling is default in Apptek conformer
                prior_scales=[0.3],
                lm_scales=[0.5],
                feature_type=FeatureType.LOGMEL_8K,
                recognition_scoring_type=RecognitionScoringType.LatticeUpsample,
                search_stats=True,
                seq2seq_v2=True,
                rqmt_update={'time': 20, 'mem': 48, 'cpu': 3},
            )
            align_args = exp_args.get_ctc_align_step_args(
                num_classes=num_outputs,
                reduction_factor=4,
                reduction_subtrahend=3,
                feature_type=FeatureType.LOGMEL_8K,
                prior_scale=0.3,
                epoch=500,
            )

            # ********** System **********

            system = ReturnnSeq2SeqSystem(
                tools,
                summary_keys=[
                    SummaryKey.TRAIN_NAME,
                    SummaryKey.RECOG_NAME,
                    SummaryKey.CORPUS,
                    SummaryKey.EPOCH,
                    SummaryKey.PRIOR,
                    SummaryKey.LM,
                    SummaryKey.WER,
                    SummaryKey.SUB,
                    SummaryKey.DEL,
                    SummaryKey.INS,
                    SummaryKey.ERR,
                    SummaryKey.RTF,
                ],
                summary_sort_keys=[SummaryKey.ERR, SummaryKey.CORPUS],
            )

            system.init_corpora(
                dev_keys=data.dev_keys,
                test_keys=data.test_keys,
                align_keys=data.align_keys,
                #test_keys=["test_set.ES_US.f8kHz_eval_callcenter_lt-v5_2022-09-bcn-arpa-local-or-s3_ref-seg"],
                corpus_data=data.data_inputs,
            )
            system.setup_scoring(
                scorer_type=ApptekScorer,
                stm_kwargs={"non_speech_tokens": ["[MUSIC]", "[NOISE]"]},
                score_kwargs={
                    "sctk_binary_path": tk.Path("/home/hwu/repositories/SCTK/bin"),
                },
            )

            # ********** Returnn Configs **********
            filter_suffix = "_filterInvalidSeq" if filter_invalid_seq else ""
            amp_suffix = "_noAmp" if not use_amp else ""
            concat_suffix = "_seqConcat" if seq_concat else ""
            system.add_experiment_configs(
                f"Conformer_CTC_4-gpu_initialLR-{initial_lr}_peakLR-{peak_lr}_decayLR-{decay_lr}_finalLR-{final_lr:.1g}_l2-{l2:.1g}{filter_suffix}{amp_suffix}{concat_suffix}_maxSeqLen{max_seq_len}_globalClip50_newData",
                get_returnn_config_collection(
                    500,  # 500 sub-epoch, 40 partition epoch, 4 gpu -> 50 full epochs
                    initial_lr=initial_lr,
                    peak_lr=peak_lr,
                    decay_lr=decay_lr,
                    final_lr=final_lr,
                    l2=l2,
                    filter_invalid_seq=filter_invalid_seq,
                    use_amp=use_amp,
                    max_seq_len=max_seq_len,
                    seq_concat=seq_concat,
                ),
            )

            system.run_train_step(**train_args)
            #system.run_dev_recog_step(**recog_args)
            recog_args["epochs"] = [500]
            system.run_test_recog_step(**recog_args)

            align_data = next(iter(system.run_align_step(**align_args).values()))

            system.get_train_job().update_rqmt("run", {"gpu_mem": 48})

            assert system.summary_report
            summary_reports.append(system.summary_report)
    
    return summary_reports, align_data


def py():
    filename_handle = os.path.splitext(os.path.basename(__file__))[0]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    srs, align_data = run_exp()
    for sr in srs:
        summary_report.merge_report(sr, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, align_data
