import copy
from typing import Any, Dict, List, Optional

from i6_core.returnn import SearchBPEtoWordsJob, SearchWordsToCTMJob
from i6_experiments.users.berger.recipe import returnn
from i6_core.lexicon.modification import itertools
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from sisyphus import tk

from i6_experiments.users.berger.helpers.kenlm import arpa_to_kenlm_bin
from i6_experiments.users.berger.helpers.rasr_lm_config import ArpaLMData
from i6_experiments.users.berger.recipe.lexicon.conversion import BlissLexiconToWordLexicon
from i6_experiments.users.berger.recipe.returnn.training import (
    Backend,
)
from i6_experiments.users.berger.corpus.general.hdf import build_feature_hdf_dataset_config
from i6_experiments.users.berger.corpus.general.ogg import build_oggzip_dataset_config

from ... import dataclasses
from ..base import RecognitionFunctor
from .returnn_search import LexiconType, VocabType, LmType


class OptunaReturnnSearchFunctor(RecognitionFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig]):
    def __init__(
        self,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
        rasr_binary_path: tk.Path,
    ) -> None:
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe
        self.rasr_binary_path = rasr_binary_path

    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        recog_config: dataclasses.NamedConfig[returnn.OptunaReturnnConfig],
        recog_corpus: dataclasses.NamedRasrDataInput,
        feature_type: dataclasses.FeatureType,
        epochs: List[int],
        trial_nums: List[int],
        lm_scales: List[float] = [0.0],
        prior_scales: List[float] = [0.0],
        lexicon_type: LexiconType = LexiconType.NONE,
        vocab_type: VocabType = VocabType.NONE,
        lm_type: LmType = LmType.NONE,
        convert_bpe_results: bool = False,
        ogg_dataset: bool = False,
        extra_audio_config: Optional[dict] = None,
        backend: Backend = Backend.PYTORCH,
        **kwargs,
    ) -> List[Dict]:
        assert recog_corpus.data.scorer is not None

        recog_results = []

        out_scores = {trial_num: [] for trial_num in trial_nums}

        recog_config_forward = copy.deepcopy(recog_config.config)

        if ogg_dataset:
            recog_config_forward.update(
                returnn.ReturnnConfig(
                    config={
                        "forward": build_oggzip_dataset_config(
                            data_inputs=[recog_corpus.data],
                            returnn_root=self.returnn_root,
                            returnn_python_exe=self.returnn_python_exe,
                            audio_config={
                                "features": "raw",
                                "peak_normalization": True,
                                **(extra_audio_config or {}),
                            },
                            extra_config={
                                "partition_epoch": 1,
                                "seq_ordering": "sorted",
                            },
                        )
                    }
                )
            )
        else:
            recog_config_forward.update(
                returnn.ReturnnConfig(
                    config={
                        "forward": build_feature_hdf_dataset_config(
                            data_inputs=[recog_corpus.data],
                            feature_type=feature_type,
                            returnn_root=self.returnn_root,
                            returnn_python_exe=self.returnn_python_exe,
                            rasr_binary_path=self.rasr_binary_path,
                            single_hdf=True,
                            extra_config={
                                "partition_epoch": 1,
                                "seq_ordering": "sorted",
                            },
                        )
                    }
                )
            )

        for lm_scale, prior_scale, epoch, trial_num in itertools.product(lm_scales, prior_scales, epochs, trial_nums):
            updated_recog_config = copy.deepcopy(recog_config_forward)

            trial = train_job.job.out_trials[trial_num]
            if epoch == "best":
                checkpoint = returnn.GetBestCheckpointJob(
                    model_dir=train_job.job.out_trial_model_dir[trial_num],
                    learning_rates=train_job.job.out_trial_learning_rates[trial_num],
                    backend=backend,
                ).out_checkpoint
            else:
                checkpoint = train_job.job.out_trial_checkpoints[trial_num][epoch]

            config_update: Dict[str, Any] = {"lm_scale": lm_scale, "prior_scale": prior_scale}

            if prior_scale:
                prior_job = returnn.OptunaReturnnForwardJob(
                    model_checkpoint=checkpoint,
                    optuna_returnn_config=prior_config,
                    trial=trial,
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
                lexicon_file = recog_corpus.data.lexicon.filename
            elif lexicon_type == LexiconType.FLASHLIGHT:
                lexicon_file = recog_corpus.data.lexicon.filename
                lexicon_file = BlissLexiconToWordLexicon(lexicon_file).out_lexicon
            else:
                raise NotImplementedError

            config_update["lexicon_file"] = lexicon_file

            if vocab_type == VocabType.NONE:
                vocab_file = None
            elif vocab_type == VocabType.LEXICON_INVENTORY:
                vocab_file = ReturnnVocabFromPhonemeInventory(recog_corpus.data.lexicon.filename).out_vocab
            else:
                raise NotImplementedError

            config_update["vocab_file"] = vocab_file

            if lm_type == LmType.NONE:
                lm_file = None
            elif lm_type == LmType.ARPA_FILE:
                assert isinstance(recog_corpus.data.lm, ArpaLMData)
                lm_file = arpa_to_kenlm_bin(recog_corpus.data.lm.filename)
            else:
                raise NotImplementedError

            config_update["lm_file"] = lm_file

            updated_recog_config.update(returnn.ReturnnConfig(config=config_update))

            forward_job = returnn.OptunaReturnnForwardJob(
                model_checkpoint=checkpoint,
                trial=trial,
                optuna_returnn_config=updated_recog_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                output_files=["search_out.py"],
                **kwargs,
            )

            exp_full = f"{recog_config.name}_e-{epoch:03d}"
            if prior_scale:
                exp_full += f"_prior-{prior_scale:02.2f}"
            if lm_scale:
                exp_full += f"_lm-{lm_scale:02.2f}"

            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/{exp_full}"

            forward_job.set_vis_name(f"Recog {path}")
            forward_job.add_alias(path)

            search_output = forward_job.out_files["search_out.py"]
            if convert_bpe_results:
                search_output = SearchBPEtoWordsJob(search_output).out_word_search_results
            word2ctm = SearchWordsToCTMJob(search_output, recog_corpus.data.corpus_object.corpus_file).out_ctm_file

            scorer_job = recog_corpus.data.scorer.get_score_job(word2ctm)

            tk.register_output(
                f"{path}.reports",
                scorer_job.out_report_dir,
            )

            out_scores[trial_num].append(
                returnn.OptunaReportIntermediateScoreJob(
                    trial_num=trial_num,
                    step=epoch,
                    score=scorer_job.out_wer,
                    study_name=train_job.job.study_name,
                    study_storage=train_job.job.study_storage,
                ).out_reported_score
            )

            recog_results.append(
                {
                    dataclasses.SummaryKey.TRAIN_NAME.value: train_job.name,
                    dataclasses.SummaryKey.RECOG_NAME.value: recog_config.name,
                    dataclasses.SummaryKey.CORPUS.value: recog_corpus.name,
                    dataclasses.SummaryKey.TRIAL.value: trial_num,
                    dataclasses.SummaryKey.EPOCH.value: epoch,
                    dataclasses.SummaryKey.LM.value: lm_scale,
                    dataclasses.SummaryKey.PRIOR.value: prior_scale,
                    dataclasses.SummaryKey.WER.value: scorer_job.out_wer,
                    dataclasses.SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                    dataclasses.SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                    dataclasses.SummaryKey.INS.value: scorer_job.out_percent_insertions,
                    dataclasses.SummaryKey.ERR.value: scorer_job.out_num_errors,
                }
            )

        for trial_num in trial_nums:
            final_score = returnn.OptunaReportFinalScoreJob(
                trial_num=trial_num,
                scores=out_scores[trial_num],
                study_name=train_job.job.study_name,
                study_storage=train_job.job.study_storage,
            ).out_reported_score
            tk.register_output(
                f"optuna/{recog_corpus.name}/{train_job.name}/trial-{trial_num:03d}/best_wer",
                value=final_score,
            )

        return recog_results
