"""
Pipeline parts to create the necessary jobs for training / forwarding / search etc...
"""
import copy
import enum
from dataclasses import dataclass, asdict
import os.path
from typing import Any, Dict, List, Optional, Tuple

from sisyphus import tk, Job, Task

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob, AverageTorchCheckpointsJob, GetBestPtCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datasets import Dataset

from .config import get_forward_config, get_training_config, get_prior_config, TrainingDatasets
from .default_tools import SCTK_BINARY_PATH, RETURNN_EXE, MINI_RETURNN_ROOT


@dataclass
class ASRModel:
    checkpoint: tk.Path
    net_args: Dict[str, Any]
    network_module: str
    prior_file: Optional[tk.Path]
    prefix_name: Optional[str]


@dataclass
class NeuralLM:
    checkpoint: tk.Path
    net_args: Dict[str, Any]
    network_module: str
    prefix_name: Optional[str]
    bpe_vocab: Optional[tk.Path] = None
    bpe_codes: Optional[tk.Path] = None
    phon_vocab: Optional[tk.Path] = None
    onnx_state_initializer: Optional[tk.Path] = None
    onnx_state_updater: Optional[tk.Path] = None
    onnx_scorer: Optional[tk.Path] = None


def search_single(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    recognition_dataset: Dataset,
    recognition_bliss_corpus: tk.Path,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: float = 16,
    use_gpu: bool = False,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param recognition_dataset: Dataset to perform recognition on
    :param recognition_bliss_corpus: path to bliss file used as Sclite evaluation reference
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: some search jobs might need more memory
    :param use_gpu: if to do GPU decoding
    """
    returnn_config = copy.deepcopy(returnn_config)
    # real RETURNN reads the forward dataset from the `forward_data` config key
    # (MiniReturnn used `forward`); see __main__.execute_main_task.
    returnn_config.config["forward_data"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="gpu" if use_gpu else "cpu",
        cpu_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=search_job.out_files["search_out.py"],
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job


@tk.block()
def search(
    prefix_name: str,
    forward_config: Dict[str, Any],
    asr_model: ASRModel,
    decoder_module: str,
    decoder_args: Dict[str, Any],
    test_dataset_tuples: Dict[str, Tuple[Dataset, tk.Path]],
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    unhashed_decoder_args: Optional[Dict[str, Any]] = None,
    use_gpu: bool = False,
    mem_rqmt: float = 16,
    debug: bool = False,
):
    """
    Run search over multiple datasets and collect statistics

    :param prefix_name: prefix folder path for alias and output files
    :param forward_config: returnn config parameter for the forward job
    :param asr_model: the ASRModel from the training
    :param decoder_module: path to the file containing the decoder definition
    :param decoder_args: arguments for the decoding forward_init_hook
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param unhashed_decoder_args: decoder arguments for the decoding forward_init_hook, but not hashed
    :param use_gpu: run search with GPU
    """
    if asr_model.prior_file is not None:
        decoder_args["config"]["prior_file"] = asr_model.prior_file

    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        unhashed_decoder_args=unhashed_decoder_args,
        decoder=decoder_module,
        debug=debug,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_name = prefix_name + "/%s" % key
        wers[search_name], search_job = search_single(
            search_name,
            returnn_search_config,
            asr_model.checkpoint,
            test_dataset,
            test_dataset_reference,
            returnn_exe,
            returnn_root,
            use_gpu=use_gpu,
            mem_rqmt=mem_rqmt,
        )
        search_jobs.append(search_job)

    return search_jobs, wers


class PhonemizeSearchWordsJob(Job):
    """
    Replace each word in a RETURNN search-output dict (``search_out.py``) by its first phoneme
    pronunciation from a bliss lexicon, producing a phoneme-level search-output dict.

    Used for the auxiliary PER of the lexicon-constrained search: that search emits a WORD
    hypothesis, which we map to phonemes via the same EOW lexicon used to phonemize the reference,
    so it can be scored against the phoneme reference with sclite. Tokens absent from the lexicon
    (should not occur for a lexicon-constrained hypothesis other than bracketed tags, which the
    CTM job filters anyway) are passed through unchanged.
    """

    def __init__(self, recog_words_file: tk.Path, bliss_lexicon: tk.Path):
        self.recog_words_file = recog_words_file
        self.bliss_lexicon = bliss_lexicon
        self.out_search_results = self.output_path("search_out.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from i6_core.lib.lexicon import Lexicon
        from i6_core.util import uopen

        lex = Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        # orth -> first pronunciation (mirrors ApplyLexiconToCorpusJob's PICK_FIRST strategy).
        lookup = {}
        for lemma in lex.lemmata:
            if not lemma.phon:
                continue
            for orth in lemma.orth:
                if orth and orth not in lookup:
                    lookup[orth] = lemma.phon[0]

        with uopen(self.recog_words_file.get_path(), "rt") as f:
            d = eval(f.read())
        assert isinstance(d, dict), "search_out.py must be a dict of seq_tag -> hypothesis"

        with uopen(self.out_search_results.get_path(), "wt") as f:
            f.write("{\n")
            for seq_tag, hyp in d.items():
                toks = [lookup.get(w, w) for w in hyp.split()]
                phon_hyp = " ".join(" ".join(toks).split())
                f.write("%r: %r,\n" % (seq_tag, phon_hyp))
            f.write("}\n")


class PhonemizeCorpusJob(Job):
    """
    Replace every word in a bliss corpus's segment orthography by its first phoneme pronunciation
    from a bliss lexicon, producing a phoneme-level corpus (used to build the phoneme reference STM).

    Like i6_core's :class:`ApplyLexiconToCorpusJob` (PICK_FIRST) but does NOT raise on
    out-of-vocabulary words: an OOV word is kept verbatim as a single token. OOVs are rare proper
    nouns absent from the LibriSpeech lexicon (~0.2-0.4% of dev words, e.g. "MAINHALL"); keeping the
    word makes it an always-wrong reference token (it can never match a phoneme hypothesis), which
    is the correct effect for an unscorable word and is identical across the lexicon-free and
    lexicon-constrained hypotheses (so it does not bias their comparison).
    """

    def __init__(self, bliss_corpus: tk.Path, bliss_lexicon: tk.Path):
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from i6_core.lib import corpus as _corpus
        from i6_core.lib.lexicon import Lexicon

        lex = Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        lookup = {}
        for lemma in lex.lemmata:
            if not lemma.phon:
                continue
            for orth in lemma.orth:
                if orth and orth not in lookup:
                    lookup[orth] = lemma.phon[0]

        c = _corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        for segment in c.segments():
            toks = [lookup.get(w, w) for w in segment.orth.split()]
            segment.orth = " ".join(" ".join(toks).split())
        c.dump(self.out_corpus.get_path())


class PhonemeDurationStatsJob(Job):
    """
    Aggregate average phoneme / silence **durations** (ms) from a RASR-produced phoneme CTM.

    Input is the ``durations.ctm`` written by ``ctm_lexfree_ngram_v1`` -- one row
    ``<seg> 1 <start_s> <dur_s> <label>`` per LibRASR traceback item (phonemes plus the index-0
    blank / silence lemma), in real seconds. Rows are grouped per segment and ordered by start time;
    each row is one occurrence of its label (the time-synchronous search already merged self-loops
    and blank / silence runs into one item).

    Categories (only the applicable ones are emitted as output variables):

    * phonemes -- split on the trailing EOW ``#`` into ``phon_eow`` / ``phon_non_eow`` when
      ``use_eow`` is set, else a single ``phon``;
    * silence -- when ``fold_blank_into_phoneme`` is False (pHMM: the index-0 label is real
      ``[SILENCE]``), each silence occurrence is classified by its position in the segment into
      ``sil_leading`` (before the first phoneme), ``sil_trailing`` (after the last phoneme) or
      ``sil_between`` (everything in between);
    * CTC blank -- when ``fold_blank_into_phoneme`` is True (the index-0 label is ``[BLANK]``,
      not silence), every blank occurrence's duration is added to the **preceding** phoneme (a
      leading blank with no preceding phoneme is added to the following one), so the blank tail
      belongs to its phoneme; no silence categories are emitted.

    Each category's value is the micro-average duration in ms (total duration / occurrence count).
    ``out_vars[category]`` holds the value (None if the category never occurred); ``out_stats`` is a
    JSON dump with per-category ``avg_ms`` / ``count`` / ``total_s`` for inspection.
    """

    def __init__(
        self,
        ctm_file: tk.Path,
        *,
        silence_label: str,
        use_eow: bool,
        fold_blank_into_phoneme: bool,
    ):
        self.ctm_file = ctm_file
        self.silence_label = silence_label
        self.use_eow = use_eow
        self.fold_blank_into_phoneme = fold_blank_into_phoneme

        phon_categories = ["phon_eow", "phon_non_eow"] if use_eow else ["phon"]
        sil_categories = [] if fold_blank_into_phoneme else ["sil_leading", "sil_trailing", "sil_between"]
        self.categories = phon_categories + sil_categories

        self.out_stats = self.output_path("durations.json")
        self.out_vars: Dict[str, tk.Variable] = {c: self.output_var(f"{c}_ms") for c in self.categories}

    def tasks(self):
        yield Task("run", mini_task=True)

    def _phon_category(self, label: str) -> str:
        if not self.use_eow:
            return "phon"
        return "phon_eow" if label.endswith("#") else "phon_non_eow"

    def _fold_blank(self, occurrences: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        """Fold each blank occurrence's duration into the preceding phoneme (leading -> following)."""
        folded: List[Tuple[float, str]] = []
        pending = 0.0  # leading-blank duration waiting for the first phoneme
        for dur, label in occurrences:
            if label == self.silence_label:  # CTC blank
                if folded:
                    folded[-1] = (folded[-1][0] + dur, folded[-1][1])
                else:
                    pending += dur
            else:
                folded.append((dur + pending, label))
                pending = 0.0
        return folded

    def run(self):
        import json
        from collections import OrderedDict, defaultdict

        segments: "OrderedDict[str, List[Tuple[float, float, str]]]" = OrderedDict()
        with open(self.ctm_file.get_path(), "rt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                rec, _ch, start, dur, label = parts[0], parts[1], float(parts[2]), float(parts[3]), parts[4]
                segments.setdefault(rec, []).append((start, dur, label))

        sums: "defaultdict[str, float]" = defaultdict(float)
        counts: "defaultdict[str, int]" = defaultdict(int)
        for items in segments.values():
            items.sort(key=lambda x: x[0])
            occurrences = [(dur, label) for (_start, dur, label) in items]

            if self.fold_blank_into_phoneme:
                for dur, label in self._fold_blank(occurrences):
                    cat = self._phon_category(label)
                    sums[cat] += dur
                    counts[cat] += 1
                continue

            phon_positions = [i for i, (_d, lab) in enumerate(occurrences) if lab != self.silence_label]
            first_phon = phon_positions[0] if phon_positions else None
            last_phon = phon_positions[-1] if phon_positions else None
            for i, (dur, label) in enumerate(occurrences):
                if label == self.silence_label:
                    if first_phon is None or i < first_phon:
                        cat = "sil_leading"
                    elif i > last_phon:
                        cat = "sil_trailing"
                    else:
                        cat = "sil_between"
                else:
                    cat = self._phon_category(label)
                sums[cat] += dur
                counts[cat] += 1

        stats = {}
        for cat in self.categories:
            avg_ms = (sums[cat] / counts[cat] * 1000.0) if counts[cat] > 0 else None
            stats[cat] = {"avg_ms": avg_ms, "count": counts[cat], "total_s": sums[cat]}
            self.out_vars[cat].set(avg_ms)
        with open(self.out_stats.get_path(), "wt") as f:
            json.dump(stats, f, indent=2, sort_keys=True)


def forward_durations_ctm(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    recognition_dataset: Dataset,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: float = 16,
    use_gpu: bool = False,
) -> Tuple[tk.Path, ReturnnForwardJobV2]:
    """
    Run a forward pass that writes ``durations.ctm`` (the CTM-emitting lexfree decoder) for one
    dataset. Mirrors :func:`search_single` but collects the CTM instead of ``search_out.py`` and runs
    no sclite. Returns ``(durations.ctm path, forward job)``.
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward_data"] = recognition_dataset.as_returnn_opts()
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="gpu" if use_gpu else "cpu",
        cpu_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["durations.ctm"],
    )
    forward_job.add_alias(prefix_name + "/forward_durations")
    return forward_job.out_files["durations.ctm"], forward_job


def compute_per(
    prefix_name: str,
    search_jobs: List[ReturnnForwardJobV2],
    test_dataset_tuples: Dict[str, Tuple[Dataset, tk.Path]],
    phoneme_lexicon: tk.Path,
    *,
    hyp_is_phonemes: bool,
) -> Dict[str, tk.Variable]:
    """
    Phoneme Error Rate (PER) for each dataset of a finished :func:`search` call.

    * Reference: phonemize the word reference corpus with ``phoneme_lexicon``
      (:class:`PhonemizeCorpusJob`, first pronunciation) -> phoneme STM (:class:`CorpusToStmJob`).
    * Hypothesis: the lexicon-free search already emits a phoneme sequence (``hyp_is_phonemes=True``),
      turned into a CTM directly; the lexicon-constrained search emits words (``hyp_is_phonemes=False``),
      which are phonemized with the same lexicon first (:class:`PhonemizeSearchWordsJob`).

    ``sclite`` then scores the two phoneme streams. Both reference and hypothesis use the same
    EOW-phoneme alphabet as the model's labels (the ``#`` end-of-word markers are kept, so this is
    the error rate over the model's actual phoneme inventory). The reference STM jobs depend only on
    (corpus, lexicon) and are shared across all epochs/scales by content hash.

    :param prefix_name: same prefix passed to :func:`search` (per-dataset jobs live at ``prefix/<dataset>``).
    :param search_jobs: the list returned by :func:`search` (aligned with ``test_dataset_tuples`` order).
    :param test_dataset_tuples: ``{dataset: (Dataset, reference_bliss_corpus)}``.
    :param phoneme_lexicon: bliss lexicon mapping words -> EOW-phoneme pronunciations (e.g. the CTC lexicon).
    :param hyp_is_phonemes: ``True`` for lexicon-free (phoneme) search, ``False`` for word search.
    :return: ``{"<prefix>/<dataset>": PER tk.Variable}`` (same key shape as :func:`search`'s ``wers``).
    """
    pers: Dict[str, tk.Variable] = {}
    for (key, (_test_dataset, ref_corpus)), search_job in zip(test_dataset_tuples.items(), search_jobs):
        search_name = prefix_name + "/%s" % key

        recog_words = search_job.out_files["search_out.py"]
        if not hyp_is_phonemes:
            recog_words = PhonemizeSearchWordsJob(recog_words, phoneme_lexicon).out_search_results
        hyp_ctm = SearchWordsToCTMJob(recog_words_file=recog_words, bliss_corpus=ref_corpus).out_ctm_file

        phon_corpus = PhonemizeCorpusJob(ref_corpus, phoneme_lexicon).out_corpus
        phon_stm = CorpusToStmJob(bliss_corpus=phon_corpus).out_stm_path

        sclite_job = ScliteJob(ref=phon_stm, hyp=hyp_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
        tk.register_output(search_name + "/sclite_per/per", sclite_job.out_wer)
        tk.register_output(search_name + "/sclite_per/report", sclite_job.out_report_dir)
        pers[search_name] = sclite_job.out_wer
    return pers


@tk.block()
def compute_prior(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: int = 16,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
    """
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=8,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    # Pin to gpu_24gb. The gpu_11gb pool is the Pascal (GTX 1080 Ti, sm_61) partition and the librasr
    # venv's torch (2.11+cu130, CUDA 13.0) has NO Pascal kernels -> "no kernel image for device". CUDA
    # 13.0 also needs a recent driver, so the one stale gpu_24gb node (cn-504, old driver -> "driver
    # too old") is --exclude'd in settings.py. rqmt (incl. gpu_mem) is not hashed. See memory
    # gpu-11gb-pool-cuinit-failure.
    search_job.rqmt["gpu_mem"] = 24
    search_job.add_alias(prefix_name + "/prior_job")
    return search_job.out_files["prior.txt"]


def training(
    training_name,
    datasets,
    train_args,
    num_epochs,
    returnn_exe,
    returnn_root,
    num_processes=None,
    distributed_launch_cmd="torchrun",
):
    """
    :param training_name:
    :param datasets:
    :param train_args:
    :param num_epochs:
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param num_processes: if set, run single-node multi-GPU data-parallel training with this many
        processes (= number of GPUs). i6_core scales the cpu/gpu/mem rqmt by this factor, and the
        training config must set ``torch_distributed`` accordingly. None => single-GPU (the hash is
        then identical to the non-distributed path).
    :param distributed_launch_cmd: "torchrun" (torch backend) or "mpirun"; only passed through when
        num_processes is set. Not part of the job hash, so it can be changed without rehashing.
    """
    returnn_config = get_training_config(training_datasets=datasets, **train_args)
    default_rqmt = {
        "mem_rqmt": 24,
        "time_rqmt": 168,
        "cpu_rqmt": 6,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }
    distributed_rqmt = {}
    if num_processes is not None:
        # horovod_num_processes IS hashed (=> new experiment per GPU count); i6_core then multiplies
        # the cpu/gpu/mem rqmt by it (single node). distributed_launch_cmd is not hashed.
        distributed_rqmt = {
            "horovod_num_processes": num_processes,
            "distributed_launch_cmd": distributed_launch_cmd,
        }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt, **distributed_rqmt
    )
    train_job.add_alias(training_name + "/training")
    tk.register_output(training_name + "/learning_rates", train_job.out_learning_rates)
    return train_job


def prepare_asr_model(
    training_name,
    train_job,
    train_args,
    with_prior,
    datasets: Optional[TrainingDatasets] = None,
    get_specific_checkpoint: Optional[int] = None,
    get_best_averaged_checkpoint: Optional[Tuple[int, str]] = None,
    get_last_averaged_checkpoint: Optional[int] = None,
    prior_config: Optional[Dict[str, Any]] = None,
):
    """
    :param training_name:
    :param train_job: output of training
    :param train_args: same args as for training
    :param with_prior: If prior should be used (yes for CTC, no for RNN-T)
    :param datasets: Needed if with_prior == True
    :param get_specific_checkpoint: return a specific epoch (set one get_*)
    :param get_best_averaged_checkpoint: return the average with (n checkpoints, loss-key), n checkpoints can be 1
    :param get_last_averaged_checkpoint: return the average of the last n checkpoints
    :param prior_config: if with_prior is true, can be used to add Returnn config parameters for the prior compute job
    :return:
    """

    params = [get_specific_checkpoint, get_last_averaged_checkpoint, get_best_averaged_checkpoint]
    assert sum([p is not None for p in params]) == 1
    assert not with_prior or datasets is not None

    if get_best_averaged_checkpoint is not None:
        num_checkpoints, loss_key = get_best_averaged_checkpoint
        checkpoints = []
        for index in range(num_checkpoints):
            best_job = GetBestPtCheckpointJob(
                train_job.out_model_dir,
                train_job.out_learning_rates,
                key=loss_key,
                index=index,
            )
            best_job.add_alias(training_name + f"/get_best_job_{index}")
            checkpoints.append(best_job.out_checkpoint)
        if num_checkpoints > 1:
            # perform averaging
            avg = AverageTorchCheckpointsJob(
                checkpoints=checkpoints, returnn_python_exe=RETURNN_EXE, returnn_root=MINI_RETURNN_ROOT
            )
            checkpoint = avg.out_checkpoint
            training_name = training_name + "/avg_best_%i_cpkt" % num_checkpoints
        else:
            # we only have one
            checkpoint = checkpoints[0]
            training_name = training_name + "/best_cpkt"
    elif get_last_averaged_checkpoint is not None:
        assert get_last_averaged_checkpoint >= 2, "For the single last checkpoint use get_specific_checkpoint instead"
        num_checkpoints = len(train_job.out_checkpoints)
        avg = AverageTorchCheckpointsJob(
            checkpoints=[train_job.out_checkpoints[num_checkpoints - i] for i in range(get_last_averaged_checkpoint)],
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        checkpoint = avg.out_checkpoint
        training_name = training_name + "/avg_last_%i_cpkt" % num_checkpoints
    else:
        checkpoint = train_job.out_checkpoints[get_specific_checkpoint]
        training_name = training_name + "/ep_%i_cpkt" % get_specific_checkpoint

    prior_file = None
    if with_prior:
        returnn_config = get_prior_config(
            training_datasets=datasets,
            network_module=train_args["network_module"],
            config=prior_config if prior_config is not None else {},
            net_args=train_args["net_args"],
            unhashed_net_args=train_args.get("unhashed_net_args", None),
            debug=train_args.get("debug", False),
        )
        prior_file = compute_prior(
            training_name,
            returnn_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(training_name + "/prior.txt", prior_file)
    else:
        if prior_config is not None:
            raise ValueError("prior_config can only be set if with_prior is True")

    asr_model = ASRModel(
        checkpoint=checkpoint,
        network_module=train_args["network_module"],
        net_args=train_args["net_args"],
        prior_file=prior_file,
        prefix_name=training_name,
    )

    return asr_model
