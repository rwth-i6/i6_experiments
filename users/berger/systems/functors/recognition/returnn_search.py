import copy
from enum import Enum, auto
from typing import Any, Dict, List

from i6_core import returnn
from i6_core.lexicon.modification import itertools
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from sisyphus import tk

from i6_experiments.users.berger.helpers.kenlm import arpa_to_kenlm_bin
from i6_experiments.users.berger.recipe.lexicon.conversion import BlissLexiconToWordLexicon
from i6_experiments.users.berger.recipe.returnn.training import (
    GetBestCheckpointJob,
    GetBestEpochJob,
    get_backend,
)

from ... import dataclasses, types
from ..base import RecognitionFunctor


class LexiconType(Enum):
    NONE = auto()
    BLISS = auto()
    FLASHLIGHT = auto()


class VocabType(Enum):
    NONE = auto()
    RETURNN = auto()


class LmType(Enum):
    NONE = auto()
    ARPA_FILE = auto()


class ReturnnSearchFunctor(RecognitionFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
    def __init__(
        self,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
    ) -> None:
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe

    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: returnn.ReturnnConfig,
        recog_config: dataclasses.NamedConfig[returnn.ReturnnConfig],
        recog_corpus: dataclasses.NamedCorpusInfo,
        epochs: List[types.EpochType],
        lm_scales: List[float] = [0.0],
        prior_scales: List[float] = [0.0],
        lexicon_type: LexiconType = LexiconType.NONE,
        vocab_type: VocabType = VocabType.NONE,
        lm_type: LmType = LmType.NONE,
        convert_bpe_results: bool = False,
        **_,
    ) -> List[Dict]:
        assert recog_corpus.corpus_info.scorer is not None
        crp = recog_corpus.corpus_info.crp

        recog_results = []

        backend = get_backend(recog_config.config)

        for lm_scale, prior_scale, epoch in itertools.product(lm_scales, prior_scales, epochs):
            if epoch == "best":
                checkpoint = GetBestCheckpointJob(
                    model_dir=train_job.job.out_model_dir,
                    learning_rates=train_job.job.out_learning_rates,
                    backend=backend,
                ).out_checkpoint
            else:
                checkpoint = train_job.job.out_checkpoints[epoch]

            config_update: Dict[str, Any] = {"lm_scale": lm_scale, "prior_scale": prior_scale}

            if prior_scale:
                prior_job = returnn.ReturnnForwardJobV2(
                    model_checkpoint=checkpoint,
                    returnn_config=prior_config,
                    returnn_root=self.returnn_root,
                    returnn_python_exe=self.returnn_python_exe,
                    output_files=["prior.txt"],
                    mem_rqmt=8,
                )

                prior_file = prior_job.out_files["prior.txt"]
            else:
                prior_file = None

            config_update["prior_file"] = prior_file

            if lexicon_type == LexiconType.NONE:
                lexicon_file = None
            elif lexicon_type == LexiconType.BLISS:
                assert crp.lexicon_config is not None
                lexicon_file = crp.lexicon_config.file
            elif lexicon_type == LexiconType.FLASHLIGHT:
                assert crp.lexicon_config is not None
                lexicon_file = BlissLexiconToWordLexicon(crp.lexicon_config.file).out_lexicon
            else:
                raise NotImplementedError

            config_update["lexicon_file"] = lexicon_file

            if vocab_type == VocabType.NONE:
                vocab_file = None
            elif vocab_type == VocabType.RETURNN:
                assert crp.lexicon_config is not None
                vocab_file = ReturnnVocabFromPhonemeInventory(crp.lexicon_config.file).out_vocab
            else:
                raise NotImplementedError

            config_update["vocab_file"] = vocab_file

            if lm_type == LmType.NONE:
                lm_file = None
            elif lm_type == LmType.ARPA_FILE:
                assert crp.language_model_config is not None
                assert crp.language_model_config.type == "ARPA"
                lm_file = arpa_to_kenlm_bin(crp.language_model_config.file)
            else:
                raise NotImplementedError

            config_update["lm_file"] = lm_file

            updated_recog_config = copy.deepcopy(recog_config.config)
            updated_recog_config.update(returnn.ReturnnConfig(config=config_update))

            forward_job = returnn.ReturnnForwardJobV2(
                model_checkpoint=checkpoint,
                returnn_config=updated_recog_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                output_files=["search_out.py"],
                mem_rqmt=8,
            )

            if isinstance(epoch, str):
                epoch_str = epoch
            else:
                epoch_str = f"{epoch:03d}"

            exp_full = f"{recog_config.name}_e-{epoch_str}"
            if prior_scale:
                exp_full += f"_prior-{prior_scale:02.2f}"
            if lm_scale:
                exp_full += f"_lm-{lm_scale:02.2f}"

            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/{exp_full}"

            forward_job.set_vis_name(f"Recog {path}")
            forward_job.add_alias(path)

            search_output = forward_job.out_files["search_out.py"]
            if convert_bpe_results:
                search_output = returnn.SearchBPEtoWordsJob(search_output).out_word_search_results
            word2ctm = returnn.SearchWordsToCTMJob(
                search_output, recog_corpus.corpus_info.data.corpus_object.corpus_file
            ).out_ctm_file

            scorer_job = recog_corpus.corpus_info.scorer.get_score_job(word2ctm)

            tk.register_output(
                f"{path}.reports",
                scorer_job.out_report_dir,
            )

            if epoch == "best":
                epoch_value = GetBestEpochJob(train_job.job.out_learning_rates).out_epoch
            else:
                epoch_value = epoch

            recog_results.append(
                {
                    dataclasses.SummaryKey.TRAIN_NAME.value: train_job.name,
                    dataclasses.SummaryKey.RECOG_NAME.value: recog_config.name,
                    dataclasses.SummaryKey.CORPUS.value: recog_corpus.name,
                    dataclasses.SummaryKey.EPOCH.value: epoch_value,
                    dataclasses.SummaryKey.LM.value: lm_scale,
                    dataclasses.SummaryKey.PRIOR.value: prior_scale,
                    dataclasses.SummaryKey.WER.value: scorer_job.out_wer,
                    dataclasses.SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                    dataclasses.SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                    dataclasses.SummaryKey.INS.value: scorer_job.out_percent_insertions,
                    dataclasses.SummaryKey.ERR.value: scorer_job.out_num_errors,
                }
            )

        return recog_results
