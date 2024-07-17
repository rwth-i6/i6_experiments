from typing import Dict, List

from i6_core import returnn
from sisyphus import tk

from i6_experiments.users.berger.recipe.returnn.training import (
    GetBestCheckpointJob,
    GetBestEpochJob,
    get_backend,
)

from ... import dataclasses, types
from ..base import RecognitionFunctor


class ReturnnBpeSearchFunctor(RecognitionFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
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
        recog_config: dataclasses.NamedConfig[returnn.ReturnnConfig],
        recog_corpus: dataclasses.NamedRasrDataInput,
        epochs: List[types.EpochType],
        **_,
    ) -> List[Dict]:
        assert recog_corpus.data.scorer is not None

        recog_results = []

        backend = get_backend(recog_config.config)

        for epoch in epochs:
            if epoch == "best":
                checkpoint = GetBestCheckpointJob(
                    model_dir=train_job.job.out_model_dir,
                    learning_rates=train_job.job.out_learning_rates,
                    backend=backend,
                ).out_checkpoint
            else:
                checkpoint = train_job.job.out_checkpoints[epoch]

            forward_job = returnn.ReturnnForwardJobV2(
                model_checkpoint=checkpoint,
                returnn_config=recog_config.config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                output_files=["search_out.py"],
            )

            if isinstance(epoch, str):
                epoch_str = epoch
            else:
                epoch_str = f"{epoch:03d}"

            exp_full = f"{recog_config.name}_e-{epoch_str}"

            path = f"nn_recog/{recog_corpus.name}/{train_job.name}/{exp_full}"

            forward_job.set_vis_name(f"Recog {path}")
            forward_job.add_alias(path)

            bpe2word = returnn.SearchBPEtoWordsJob(forward_job.out_files["search_out.py"]).out_word_search_results
            word2ctm = returnn.SearchWordsToCTMJob(bpe2word, recog_corpus.data.corpus_object.corpus_file).out_ctm_file

            scorer_job = recog_corpus.data.scorer.get_score_job(word2ctm)

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
                    dataclasses.SummaryKey.WER.value: scorer_job.out_wer,
                    dataclasses.SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                    dataclasses.SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                    dataclasses.SummaryKey.INS.value: scorer_job.out_percent_insertions,
                    dataclasses.SummaryKey.ERR.value: scorer_job.out_num_errors,
                }
            )

        return recog_results
