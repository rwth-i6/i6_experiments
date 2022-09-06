from sisyphus import tk
from typing import Dict
from copy import deepcopy
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_tts_data_from_rasr_alignment, get_ls_100_f0_hdf, extend_meta_datasets_with_f0, get_ls_100_energy_hdf,
    extend_meta_datasets_with_energy
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
    get_training_config,
    tts_training,
    synthesize_with_splits,
    build_speaker_embedding_dataset,
    gl_swer,
    get_forward_config
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
    get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_inference_dataset_old, TTSForwardData
)


def gmm_duration_cheat(alignments: Dict, rasr_allophones):
    """
    :param alignments
    Experiments with duration predictor cheating
    :return:
    """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository
    synthetic_data_dict = {}
    job_splits = 10
    reference_corpus = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )["train-clean-100"]
    default_vocoder = get_default_vocoder(
        name="experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/vocoder/")

    for align_name, alignment in alignments.items():
        name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/{align_name}"
        (
            training_datasets,
            vocoder_data,
            new_corpus,
            durations_hdf,
        ) = get_tts_data_from_rasr_alignment(
            name + "/datasets",
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            rasr_alignment=alignment,
            rasr_allophones=rasr_allophones,
        )

        for upsampling in ["repeat", "gauss"]:
          exp_name = name + f"/{upsampling}"
          train_config = get_training_config(
              returnn_common_root=returnn_common_root,
              training_datasets=training_datasets,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
          )
          if upsampling == "gauss":
              train_config.config["learning_rates"] = [0.0001, 0.001]
          train_job = tts_training(
              config=train_config,
              returnn_exe=returnn_exe,
              returnn_root=returnn_root,
              prefix=exp_name,
              num_epochs=200,
          )
          forward_config = get_forward_config(
              returnn_common_root=returnn_common_root,
              forward_dataset=TTSForwardData(
                  dataset=training_datasets.cv, datastreams=training_datasets.datastreams
              ),
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              calc_speaker_embedding=True,
          )
          gl_swer(
              name=exp_name + "/gl_swer",
              vocoder=default_vocoder,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              checkpoint=train_job.out_checkpoints[200],
              config=forward_config,
          )
          # synthesis

          speaker_embedding_hdf = build_speaker_embedding_dataset(
              returnn_common_root=returnn_common_root,
              returnn_exe=returnn_exe,
              returnn_root=returnn_root,
              datasets=training_datasets,
              prefix=exp_name,
              train_job=train_job
          )
          for dur_pred in ["pred", "cheat"]:
            synth_dataset = get_inference_dataset_old(
                new_corpus,
                returnn_root=returnn_root,
                returnn_exe=returnn_exe,
                datastreams=training_datasets.datastreams,
                speaker_embedding_hdf=speaker_embedding_hdf,
                durations=durations_hdf if dur_pred == "cheat" else None,
                process_corpus=False,
            )

            synth_corpus = synthesize_with_splits(
                name=exp_name + f"/{dur_pred}",
                reference_corpus=reference_corpus.corpus_file,
                corpus_name="train-clean-100",
                job_splits=job_splits,
                datasets=synth_dataset,
                returnn_root=returnn_root,
                returnn_exe=returnn_exe,
                returnn_common_root=returnn_common_root,
                checkpoint=train_job.out_checkpoints[200],
                vocoder=default_vocoder,
                embedding_size=256,
                speaker_embedding_size=256,
                gauss_up=(upsampling == "gauss"),
                use_true_durations=(dur_pred == "cheat"),
            )
            synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}"] = synth_corpus

            if upsampling == "repeat" and align_name == "tts_align_sat" and dur_pred == "pred":
                for add in [0.5]:
                    synth_corpus = synthesize_with_splits(
                        name=exp_name + f"/{dur_pred}_add_{add}",
                        reference_corpus=reference_corpus.corpus_file,
                        corpus_name="train-clean-100",
                        job_splits=job_splits,
                        datasets=synth_dataset,
                        returnn_root=returnn_root,
                        returnn_exe=returnn_exe,
                        returnn_common_root=returnn_common_root,
                        checkpoint=train_job.out_checkpoints[200],
                        vocoder=default_vocoder,
                        embedding_size=256,
                        speaker_embedding_size=256,
                        gauss_up=(upsampling == "gauss"),
                        use_true_durations=(dur_pred == "cheat"),
                        duration_add=add
                    )
                    synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_add_{add}"] = synth_corpus

    for align_name in ["tts_align_sat"]:
      alignment = alignments[align_name]
      name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
      (
        training_datasets,
        vocoder_data,
        new_corpus,
        durations_hdf,
      ) = get_tts_data_from_rasr_alignment(
        name + "/datasets",
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        rasr_alignment=alignment,
        rasr_allophones=rasr_allophones,
      )
      returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
        checkout_folder_name="returnn_common",
      ).out_repository
      returnn_root_job = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="ce4366ff0caafc2e4b349fd2a189870f3c76f630",  # fixes shape error in normal
      )
      returnn_root = returnn_root_job.out_repository
      #for variance in ["f0", "energy", "f0_energy"]:
      for variance in ["f0"]:
        for upsampling in ["repeat", "gauss"]:
          exp_name = name + f"/{variance}/{upsampling}"
          var_training_datasets = deepcopy(training_datasets)
          if "f0" in variance:
            f0_hdf = get_ls_100_f0_hdf(durations=durations_hdf, returnn_exe=returnn_exe, returnn_root=returnn_root,
              prefix=exp_name)
            var_training_datasets = extend_meta_datasets_with_f0(datasets=training_datasets, f0_dataset=f0_hdf)
          if "energy" in variance:
            energy_hdf = get_ls_100_energy_hdf(returnn_exe=returnn_exe,
              returnn_root=returnn_root, prefix=exp_name)
            var_training_datasets = extend_meta_datasets_with_energy(var_training_datasets, energy_dataset=energy_hdf)
          train_config = get_training_config(
              returnn_common_root=returnn_common_root,
              training_datasets=var_training_datasets,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_pitch_pred=("f0" in variance),
              use_energy_pred=("energy" in variance)
          )
          if upsampling == "gauss":
              train_config.config["learning_rates"] = [0.0001, 0.001]
          train_job = tts_training(
            config=train_config,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
            num_epochs=200,
          )
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=TTSForwardData(
              dataset=var_training_datasets.cv, datastreams=var_training_datasets.datastreams
            ),
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            calc_speaker_embedding=True,
            use_pitch_pred=("f0" in variance),
            use_energy_pred=("energy" in variance)
          )
          gl_swer(
              name=exp_name + "/gl_swer",
              vocoder=default_vocoder,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              checkpoint=train_job.out_checkpoints[200],
              config=forward_config,
          )

          speaker_embedding_hdf = build_speaker_embedding_dataset(
            returnn_common_root=returnn_common_root,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            datasets=var_training_datasets,
            prefix=exp_name,
            train_job=train_job
          )
          
          for dur_pred in ["pred", "cheat"]:
            synth_dataset = get_inference_dataset_old(
              new_corpus,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              datastreams=var_training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations_hdf if dur_pred == "cheat" else None,
              process_corpus=False,
            )

            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=(dur_pred == "cheat"),
              use_pitch_pred=("f0" in variance),
              use_energy_pred=("energy" in variance)
            )
            synthetic_data_dict[f"{align_name}_{variance}_{upsampling}_{dur_pred}"] = synth_corpus

    return synthetic_data_dict

