from sisyphus import tk
from typing import Dict
from copy import deepcopy
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.datasets.librispeech import (
  get_corpus_object_dict,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  get_tts_data_from_rasr_alignment,
  get_ls_100_f0_hdf,
  extend_meta_datasets_with_f0,
  get_ls_100_energy_hdf,
  extend_meta_datasets_with_energy,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
  get_training_config,
  tts_training,
  synthesize_with_splits,
  build_speaker_embedding_dataset,
  gl_swer,
  get_forward_config,
  build_vae_speaker_prior_dataset,
  tts_forward,
  calculate_feature_variance,
  get_average_checkpoint_v2,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
  get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  get_inference_dataset_old,
  TTSForwardData,
  get_inference_dataset,
)
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import AddSpeakerTagsFromMappingJob
from i6_experiments.users.hilmes.tools.tts.analysis import CalculateVarianceFromFeaturesJob


def gmm_duration_cheat(
  alignments: Dict,
  rasr_allophones,
  full_graph=False,
  return_trainings=False,
  basic_trainings=False,
  skip_extensions=False,
  silence_prep=True,
):
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
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(
    name="experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/vocoder/"
  )
  trainings = {}
  for align_name, alignment in alignments.items():
    # if not full_graph and not return_trainings:
    #  continue
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/{align_name}"
    (training_datasets, vocoder_data, new_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
      name + "/datasets",
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      rasr_alignment=alignment,
      rasr_allophones=rasr_allophones,
      silence_prep=silence_prep,
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
      if upsampling == "repeat" and align_name == "tts_align_sat" and return_trainings:
        trainings["tts_align_sat/repeat/baseline"] = train_job
      if upsampling == "gauss" and align_name == "tts_align_sat" and return_trainings:
        trainings["tts_align_sat/gauss/baseline"] = train_job
      if return_trainings:
        continue
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
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
      if upsampling == "gauss" and align_name == "tts_align_sat" and basic_trainings:
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          returnn_root=returnn_root,
          returnn_exe=returnn_exe,
          prefix=name,
        )
        forward_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output("test/gauss_pred_sat.hdf", forward_hdf)
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=(upsampling == "gauss"),
        calc_speaker_embedding=True,
        use_true_durations=True,
      )
      gl_swer(
        name=exp_name + "/gl_swer_true_durations",
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
      )
      # synthesis
      if upsampling == "gauss" and (align_name == "tts_align_sat" or align_name == "tts_align_mono"):
        returnn_common_root_local = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn_common",
          commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
          checkout_folder_name="returnn_common",
        ).out_repository
        returnn_root_local = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
        ).out_repository
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          datasets=training_datasets,
          prefix=exp_name,
          train_job=train_job,
        )

        synth_dataset = get_inference_dataset(
          new_corpus,
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
          datastreams=training_datasets.datastreams,
          speaker_embedding_hdf=speaker_embedding_hdf,
          durations=None,
          process_corpus=False,
        )
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root if "sat" in align_name else returnn_common_root_local,
          forward_dataset=synth_dataset,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=True,
          dump_durations=True,
        )
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          prefix=exp_name + "/dump_dur",
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
        )
        forward_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output(exp_name + "/dump_dur/durations.hdf", forward_hdf)

        # extract dev features and durations for covariance analysis
        # no speaker shuffling for now
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root if "sat" in align_name else returnn_common_root_local,
          forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=True,
        )
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          prefix=exp_name + "/cov_analysis/dev/features",
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
        )
        forward_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output(exp_name + "/cov_analysis/dev/features.hdf", forward_hdf)
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root if "sat" in align_name else returnn_common_root_local,
          forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=True,
          dump_durations=True,
          dump_round_durations=True,
        )
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          prefix=exp_name + "/cov_analysis/dev/durations",
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
        )
        forward_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output(exp_name + "/cov_analysis/dev/durations.hdf", forward_hdf)

        # extract full features and durations
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          datasets=training_datasets,
          prefix=exp_name + "_full_var",
          train_job=train_job,
        )

        synth_dataset = get_inference_dataset(
          new_corpus,
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
          datastreams=training_datasets.datastreams,
          speaker_embedding_hdf=speaker_embedding_hdf,
          durations=None,
          process_corpus=False,
          shuffle_info=False,
        )
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root if "sat" in align_name else returnn_common_root_local,
          forward_dataset=synth_dataset,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=True,
        )
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          prefix=exp_name + "/cov_analysis/full/features",
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
        )
        forward_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output(exp_name + "/cov_analysis/full/features.hdf", forward_hdf)
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root if "sat" in align_name else returnn_common_root_local,
          forward_dataset=synth_dataset,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=True,
          dump_round_durations=True,
          dump_durations=True,
        )
        forward_job = tts_forward(
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
          prefix=exp_name + "/cov_analysis/full/durations",
          returnn_root=returnn_root if "sat" in align_name else returnn_root_local,
          returnn_exe=returnn_exe,
        )
        dur_hdf = forward_job.out_hdf_files["output.hdf"]
        tk.register_output(exp_name + "/cov_analysis/full/durations.hdf", dur_hdf)
        var_job = CalculateVarianceFromFeaturesJob(feature_hdf=forward_hdf, duration_hdf=dur_hdf, bliss=new_corpus)
        tk.register_output(exp_name + "/cov_analysis/full/variance", var_job.out_variance)

      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job,
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
        """ removed due to space
        if upsampling == "gauss" and align_name == "tts_align_sat":
          returnn_common_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
            checkout_folder_name="returnn_common",
          ).out_repository
          returnn_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
          ).out_repository
          avrg_check = get_average_checkpoint_v2(train_job, returnn_exe, returnn_root)
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_average",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            checkpoint=avrg_check,
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_average"] = synth_corpus
        """
        var_root = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn",
          commit="5cbec73032aeba2eec269624eb755fc5766f3c98",
        ).out_repository
        var_common_root = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn_common",
          commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
          checkout_folder_name="returnn_common",
        ).out_repository
        returnn_common_root_local = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn_common",
          commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
          checkout_folder_name="returnn_common",
        ).out_repository
        returnn_root_local = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
        ).out_repository
        calculate_feature_variance(
          train_job=train_job,
          corpus=new_corpus,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          returnn_common_root=returnn_common_root_local,
          prefix=exp_name + f"/{dur_pred}",
          training_datasets=training_datasets,
          gauss_up=(upsampling == "gauss"),
          embedding_size=256,
          speaker_embedding_size=256,
          use_true_durations=(dur_pred == "cheat"),
          durations=durations_hdf if dur_pred == "cheat" else None,
          original_durations=durations_hdf,
        )
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root_local,
          forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=True,
          use_true_durations=(dur_pred == "cheat"),
        )
        gl_swer(
          name=exp_name + f"/gl_swer_{dur_pred}",
          vocoder=default_vocoder,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
        )
        if upsampling == "gauss" and dur_pred == "pred" and align_name == "tts_align_sat":
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
            name=exp_name + f"/{dur_pred}_no_round",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            checkpoint=train_job.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            round_durations=False,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_no_round"] = synth_corpus
          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root_local if not "no_sil_p" in align_name else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local if not "no_sil_p" in align_name else returnn_common_root_local,
            prefix=exp_name + f"/{dur_pred}_no_round",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=256,
            speaker_embedding_size=256,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            round_durations=False,
          )
        if upsampling == "gauss" and dur_pred == "pred" and align_name == "tts_align_sat":
          synth_dataset = get_inference_dataset_old(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
          )
          for random_duration_std in [0.025, 0.0125, 0.0375, 0.05, 0.075]:
            forward_config = get_forward_config(
              returnn_common_root=returnn_common_root_local,
              forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              calc_speaker_embedding=True,
              use_true_durations=(dur_pred == "cheat"),
              random_duration_scaling=[1.05, random_duration_std, 0.9, 1.2],
            )
            gl_swer(
              name=exp_name + f"/gl_swer_1.05_{random_duration_std}",
              vocoder=default_vocoder,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              checkpoint=train_job.out_checkpoints[200],
              config=forward_config,
            )
            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}_{random_duration_std}",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_local,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=(dur_pred == "cheat"),
              random_duration_scaling=[1.05, random_duration_std, 0.9, 1.2],
              redo=True
            )
            synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_{str(random_duration_std).replace('.','_')}"] = synth_corpus
            calculate_feature_variance(
              train_job=train_job,
              corpus=new_corpus,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_local,
              prefix=exp_name + f"/{dur_pred}_{random_duration_std}",
              training_datasets=training_datasets,
              gauss_up=(upsampling == "gauss"),
              embedding_size=256,
              speaker_embedding_size=256,
              use_true_durations=(dur_pred == "cheat"),
              durations=durations_hdf if dur_pred == "cheat" else None,
              random_duration_scaling=[1.05, random_duration_std, 0.9, 1.2],
              original_durations=durations_hdf,
              redo=True,
            )
            if True:
                _, scale_mapping = get_inference_dataset(
                    new_corpus,
                    returnn_root=returnn_root,
                    returnn_exe=returnn_exe,
                    datastreams=training_datasets.datastreams,
                    speaker_embedding_hdf=speaker_embedding_hdf,
                    durations=durations_hdf if dur_pred == "cheat" else None,
                    process_corpus=False,
                    return_mapping=True
                )
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping)
                tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_1.05_{random_duration_std}" + "/add_tags")
                tk.register_output(exp_name + "/add_tags/corpus.xml.gz", tag_corpus_job.out_corpus)
                tk.register_output(f"paper_nick/21_02_23_ls100/random_1.05_{random_duration_std}_real_tags_corpus.xml.gz",
                    tag_corpus_job.out_corpus)
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping,
                    prefix="synth_")
                tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_1.05_{random_duration_std}" + "/add_tags_synth")
                tk.register_output(
                    f"paper_nick/21_02_23_ls100/random_1.05_{random_duration_std}_synth_pref_tags_corpus.xml.gz",
                    tag_corpus_job.out_corpus)
          for mean in [1.0]:
            for random_duration_std in [0.025, 0.0125, 0.03, 0.035, 0.02, 0.0249, 0.0375, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,0.075,0.08,0.09,0.1, 1.0, 0.2, 0.3, 0.15, 0.5, 0.75]:
              synth_dataset, scale_mapping = get_inference_dataset(
                new_corpus,
                returnn_root=returnn_root,
                returnn_exe=returnn_exe,
                datastreams=training_datasets.datastreams,
                speaker_embedding_hdf=speaker_embedding_hdf,
                durations=durations_hdf if dur_pred == "cheat" else None,
                process_corpus=False,
                return_mapping=True
              )
              if True:
                  calculate_feature_variance(
                      train_job=train_job,
                      corpus=new_corpus,
                      returnn_root=returnn_root_local,
                      returnn_exe=returnn_exe,
                      returnn_common_root=returnn_common_root_local,
                      prefix=exp_name + f"/{dur_pred}_{random_duration_std}_{mean}",
                      training_datasets=training_datasets,
                      gauss_up=(upsampling == "gauss"),
                      embedding_size=256,
                      speaker_embedding_size=256,
                      use_true_durations=(dur_pred == "cheat"),
                      durations=durations_hdf if dur_pred == "cheat" else None,
                      random_duration_scaling=[mean, random_duration_std, 0.9, 1.2],
                      original_durations=durations_hdf,
                      just_kl=True,
                  )
                  calculate_feature_variance(
                      train_job=train_job,
                      corpus=new_corpus,
                      returnn_root=returnn_root_local,
                      returnn_exe=returnn_exe,
                      returnn_common_root=returnn_common_root_local,
                      prefix=exp_name + f"/{dur_pred}_{random_duration_std}_{mean}_min_0.8",
                      training_datasets=training_datasets,
                      gauss_up=(upsampling == "gauss"),
                      embedding_size=256,
                      speaker_embedding_size=256,
                      use_true_durations=(dur_pred == "cheat"),
                      durations=durations_hdf if dur_pred == "cheat" else None,
                      random_duration_scaling=[mean, random_duration_std, 0.8, 1.2],
                      original_durations=durations_hdf,
                      just_kl=True,
                  )
              if random_duration_std not in [0.0125, 0.025, 0.0375, 0.05, 0.075]:
                  continue
              forward_config = get_forward_config(
                returnn_common_root=returnn_common_root_local,
                forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
                embedding_size=256,
                speaker_embedding_size=256,
                gauss_up=(upsampling == "gauss"),
                calc_speaker_embedding=True,
                use_true_durations=(dur_pred == "cheat"),
                random_duration_scaling=[mean, random_duration_std, 0.9, 1.2],
              )
              gl_swer(
                name=exp_name + f"/gl_swer_{mean}_{random_duration_std}",
                vocoder=default_vocoder,
                returnn_root=returnn_root_local,
                returnn_exe=returnn_exe,
                checkpoint=train_job.out_checkpoints[200],
                config=forward_config,
              )
              synth_corpus = synthesize_with_splits(
                name=exp_name + f"/{dur_pred}_{random_duration_std}_{mean}",
                reference_corpus=reference_corpus.corpus_file,
                corpus_name="train-clean-100",
                job_splits=job_splits,
                datasets=synth_dataset,
                returnn_root=returnn_root_local,
                returnn_exe=returnn_exe,
                returnn_common_root=returnn_common_root_local,
                checkpoint=train_job.out_checkpoints[200],
                vocoder=default_vocoder,
                embedding_size=256,
                speaker_embedding_size=256,
                gauss_up=(upsampling == "gauss"),
                use_true_durations=(dur_pred == "cheat"),
                random_duration_scaling=[mean, random_duration_std, 0.9, 1.2],
                redo=True,
              )
              synthetic_data_dict[
                f"{align_name}_{upsampling}_{dur_pred}_{str(mean).replace('.', '_')}_{str(random_duration_std).replace('.', '_')}"] = synth_corpus
              calculate_feature_variance(
                train_job=train_job,
                corpus=new_corpus,
                returnn_root=returnn_root_local,
                returnn_exe=returnn_exe,
                returnn_common_root=returnn_common_root_local,
                prefix=exp_name + f"/{dur_pred}_{random_duration_std}_{mean}",
                training_datasets=training_datasets,
                gauss_up=(upsampling == "gauss"),
                embedding_size=256,
                speaker_embedding_size=256,
                use_true_durations=(dur_pred == "cheat"),
                durations=durations_hdf if dur_pred == "cheat" else None,
                random_duration_scaling=[mean, random_duration_std, 0.9, 1.2],
                original_durations=durations_hdf,
                redo=True,
              )
              if True:
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping)
                tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_{mean}_{random_duration_std}" + "/add_tags")
                tk.register_output(exp_name + "/add_tags/corpus.xml.gz", tag_corpus_job.out_corpus)
                tk.register_output(f"paper_nick/21_02_23_ls100/random_{mean}_{random_duration_std}_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping, prefix="synth_")
                tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_{mean}_{random_duration_std}" + "/add_tags_synth")
                tk.register_output(f"paper_nick/21_02_23_ls100/random_{mean}_{random_duration_std}_synth_pref_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
        if upsampling == "gauss" and dur_pred == "pred" and align_name == "tts_align_sat":
          corpora = []
          for seed in range(10):
            synth_dataset = get_inference_dataset(
              new_corpus,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              datastreams=training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations_hdf if dur_pred == "cheat" else None,
              process_corpus=False,
              seed=seed
            )

            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}_1000h_seed_{seed}",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100_" + str(seed),
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_local,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=(dur_pred == "cheat"),
              round_durations=False,
            )
            corpora.append(synth_corpus)
          from i6_core.corpus import MergeCorporaJob

          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_1000h"] = MergeCorporaJob(corpora, name="Pseudo-1000").out_merged_corpus
          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root_local if not "no_sil_p" in align_name else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local if not "no_sil_p" in align_name else returnn_common_root_local,
            prefix=exp_name + f"/{dur_pred}_1000h",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=256,
            speaker_embedding_size=256,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            round_durations=False,
          )
        # for duration_scale in [0.9, 1.1, 1.2, 1.3, 2.0, 1.15, 1.25, 1.75]:, REMOVED because of space
        # for duration_scale in [1.1, 1.2, 1.3, 1.15, 1.25]:, REMOVED due to space
        for duration_scale in [0.8, 0.9, 2.0, 1.2, 1.1, 1.15, 1.05, 1.25, 1.3]:
          # TODO for duration_scale in [1.1, 1.2, 1.3, 1.15, 1.25]: once finished use this + maybe remove 1.3, 1.25
          if upsampling != "gauss" or dur_pred != "pred" or align_name != "tts_align_sat":
            continue
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root_local,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            calc_speaker_embedding=True,
            use_true_durations=(dur_pred == "cheat"),
            duration_scale=duration_scale,
          )
          gl_swer(
            name=exp_name + f"/gl_swer_{dur_pred}_scale{duration_scale}",
            vocoder=default_vocoder,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            checkpoint=train_job.out_checkpoints[200],
            config=forward_config,
          )
          if duration_scale in [1.25, 1.3]:
            continue
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_scale_{duration_scale}",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root if duration_scale not in [0.8, 1.2, 0.9, 2.0, 1.05] else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root if duration_scale not in [0.8, 1.2, 0.9, 2.0, 1.05] else returnn_common_root_local,
            checkpoint=train_job.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            round_durations=True,
            duration_scale=duration_scale,
          )
          var_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="5cbec73032aeba2eec269624eb755fc5766f3c98",
          ).out_repository
          var_common_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
            checkout_folder_name="returnn_common",
          ).out_repository

          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root if duration_scale not in [0.8, 1.2, 0.9, 2.0, 1.1, 1.05, 1.15] else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root if duration_scale not in [0.8, 1.2, 0.9, 2.0, 1.1, 1.05, 1.15] else returnn_common_root_local,
            prefix=exp_name + f"/{dur_pred}_scale{duration_scale}",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=256,
            speaker_embedding_size=256,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            duration_scale=duration_scale,
            round_durations=True,
            original_durations=durations_hdf,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_scale_{str(duration_scale).replace('.', '_')}"] = synth_corpus
          if duration_scale == 1.1 or duration_scale == 1.05:
            synth_dataset, scale_mapping = get_inference_dataset(
              new_corpus,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              datastreams=training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations_hdf if dur_pred == "cheat" else None,
              process_corpus=False,
              return_mapping=True
            )
            tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping)
            tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_scale{duration_scale}" + "/add_tags")
            tk.register_output(exp_name + "/add_tags/corpus.xml.gz", tag_corpus_job.out_corpus)
            tk.register_output(f"paper_nick/21_02_23_ls100/scale_{duration_scale}_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
            tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=scale_mapping, prefix="synth_")
            tag_corpus_job.add_alias(exp_name + f"/{dur_pred}_scale{duration_scale}" + "/add_synth_tags")
            tk.register_output(f"paper_nick/21_02_23_ls100/scale_{duration_scale}_synth_prefix_tags_corpus.xml.gz", tag_corpus_job.out_corpus)

          #if duration_scale in [1.1, 1.2, 1.15]:, REMOVED due to space
          if duration_scale in [1.1, 1.15]:
            synth_dataset = get_inference_dataset(
              new_corpus,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              datastreams=training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations_hdf if dur_pred == "cheat" else None,
              process_corpus=False,
            )

            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}_scale_{duration_scale}_no_round",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root if not duration_scale == 1.15 else returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root if not duration_scale == 1.15 else returnn_common_root_local,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=(dur_pred == "cheat"),
              round_durations=False,
              duration_scale=duration_scale,
            )
            synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_scale_{str(duration_scale).replace('.', '_')}_no_round"] = synth_corpus

        for duration_scale in [1.05, 1.1, 1.15]:
        #for duration_scale in [1.1]:
          if upsampling != "gauss" or dur_pred != "pred" or align_name != "tts_align_sat/no_sil_p":
            continue
          returnn_common_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
            checkout_folder_name="returnn_common",
          ).out_repository
          returnn_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
          ).out_repository
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root_local,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            calc_speaker_embedding=True,
            use_true_durations=(dur_pred == "cheat"),
            duration_scale=duration_scale,
          )
          gl_swer(
            name=exp_name + f"/gl_swer_{dur_pred}_scale{duration_scale}",
            vocoder=default_vocoder,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            checkpoint=train_job.out_checkpoints[200],
            config=forward_config,
          )
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root if not "no_sil_p" in align_name else returnn_root_local,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
          )
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_scale_{duration_scale}",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root if not ("no_sil_p" in align_name or duration_scale in [1.05, 1.15]) else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root if not ("no_sil_p" in align_name or duration_scale in [1.05, 1.15]) else returnn_common_root_local,
            checkpoint=train_job.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            round_durations=True,
            duration_scale=duration_scale,
          )
          var_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="5cbec73032aeba2eec269624eb755fc5766f3c98",
          ).out_repository
          var_common_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
            checkout_folder_name="returnn_common",
          ).out_repository

          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=var_root if not "no_sil_p" in align_name else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=var_common_root if not "no_sil_p" in align_name else returnn_common_root_local,
            prefix=exp_name + f"/{dur_pred}_scale{duration_scale}",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=256,
            speaker_embedding_size=256,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            duration_scale=duration_scale,
            round_durations=True,
            original_durations=durations_hdf,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_scale_{str(duration_scale).replace('.', '_')}"] = synth_corpus
          if duration_scale in [1.05, 1.1, 1.15]:
            synth_dataset = get_inference_dataset(
              new_corpus,
              returnn_root=returnn_root if not "no_sil_p" in align_name else returnn_root_local,
              returnn_exe=returnn_exe,
              datastreams=training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations_hdf if dur_pred == "cheat" else None,
              process_corpus=False,
            )

            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}_scale_{duration_scale}_no_round",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=var_root if not ("no_sil_p" in align_name or duration_scale in [1.05, 1.15]) else returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=var_common_root if not ("no_sil_p" in align_name or duration_scale in [1.05, 1.15]) else returnn_common_root_local,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=(dur_pred == "cheat"),
              round_durations=False,
              duration_scale=duration_scale,
            )
            synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_scale_{str(duration_scale).replace('.', '_')}_no_round"] = synth_corpus
        if upsampling == "gauss" and "tts_align_sat" in align_name:
          synth_dataset, speaker_mapping = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
            return_mapping=True,
          )
          returnn_common_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
            checkout_folder_name="returnn_common",
          ).out_repository
          returnn_root_local = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
          ).out_repository
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_speaker_tags",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root if (align_name == "tts_align_sat" and dur_pred == "pred") else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root if (align_name == "tts_align_sat" and dur_pred == "pred") else returnn_common_root_local,
            checkpoint=train_job.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            round_durations=False,
          )
          tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
          tag_corpus_job.add_alias(exp_name + f"/{dur_pred}" + "/add_tags")
          tk.register_output(exp_name + "/add_tags/corpus.xml.gz", tag_corpus_job.out_corpus)
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_speaker_tags"] = tag_corpus_job.out_corpus
          if dur_pred == "pred" and align_name == 'tts_align_sat':
            tk.register_output("paper_nick/21_02_23_ls100/pred_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
            tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping, prefix="synth_")
            tag_corpus_job.add_alias(exp_name + f"/{dur_pred}" + "/add_tags_synth")
            tk.register_output("paper_nick/21_02_23_ls100/pred_synth_prefix_tags_corpus.xml.gz", tag_corpus_job.out_corpus)

        if basic_trainings:
          continue
        if silence_prep and False:
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_norm_fix",
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
            reconstruction_norm=False,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_norm_fix"] = synth_corpus
        if upsampling == "repeat" and align_name == "tts_align_sat" and dur_pred == "pred":
          #for add in [0.5]: REMOVED, due to space
          for add in []:
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
              duration_add=add,
            )
            synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_add_{add}"] = synth_corpus
        if upsampling == "repeat" and align_name == "tts_align_sat" and dur_pred == "cheat" and False: # REMOVED, due to space
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
            shuffle_info=False,
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_no_shuff",
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
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_no_shuff"] = synth_corpus
        if upsampling == "gauss" and align_name == "tts_align_sat" and dur_pred == "pred" and False: # REMOVED, due to space
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_no_round",
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
            round_durations=False,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_no_round"] = synth_corpus

  if return_trainings:
    return trainings
  if basic_trainings:
    print("Warning: Only basic training graph executed")
    return synthetic_data_dict
  if skip_extensions:
    print("Warning: Not including F0, Energy and other experiments")
  for align_name in ["tts_align_sat"]:
    alignment = deepcopy(alignments[align_name])
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
    (training_datasets, vocoder_data, new_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
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
    returnn_exe = tk.Path(
      "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
      hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    for variance in [
      "f0",
      "energy",
      # "f0_energy", REMOVED because of space
      # "f0_log_energy", REMOVED because of space
      # "f0_energy_vae", REMOVED because of space
      # "log_energy", REMOVED because of space
      # "energy_add_vae", REMOVED because of space
      # "energy_mul_vae", REMOVED because of space
      # "f0_test", REMOVED because of space
      # "energy_test", REMOVED because of space
      # "energy_test_mul_vae", REMOVED because of space
      # "f0_energy_test", REMOVED because of space
      # "log_energy_test", REMOVED because of space
      # "no_dropout_small", REMOVED because of space
      # "no_speaker_emb", REMOVED because of space
      # "energy_no_speaker_emb", REMOVED because of space
      # "energy_test_no_speaker_emb", REMOVED because of space
    ]:
      if "energy" in variance:
        returnn_root_job = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn",
          commit="882b63641be44d1f91566ac6d5aa517707de0c43",  # apply center to energy
        )
        returnn_root = returnn_root_job.out_repository
      if variance not in ["f0", "energy", "energy_add_vae", "energy_mul_vae"]:
        returnn_root_job = CloneGitRepositoryJob(
          "https://github.com/rwth-i6/returnn",
          commit="5cbec73032aeba2eec269624eb755fc5766f3c98",  # fix memory
        )
        returnn_root = returnn_root_job.out_repository
      for upsampling in ["repeat", "gauss"]:
        if variance in ["f0_test"] and upsampling == "gauss":
          continue
        if (
          variance in ["no_speaker_emb", "energy_test_mul_vae", "energy_no_speaker_emb", "energy_test_no_speaker_emb"]
          and upsampling == "repeat"
        ):
          continue
        exp_name = name + f"/{variance}/{upsampling}"
        var_training_datasets = deepcopy(training_datasets)
        f0_hdf = None
        energy_hdf = None
        speaker_prior_hdf = None
        if "f0" in variance:
          f0_hdf = get_ls_100_f0_hdf(
            durations=durations_hdf,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
          )
          var_training_datasets = extend_meta_datasets_with_f0(datasets=training_datasets, f0_dataset=f0_hdf)
        if "energy" in variance:
          energy_hdf = get_ls_100_energy_hdf(
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
            center=False,
            log_norm="log_energy" in variance,
          )
          var_training_datasets = extend_meta_datasets_with_energy(var_training_datasets, energy_dataset=energy_hdf)
        kwargs = {}
        if "log_energy" in variance:
          kwargs["log_energy"] = True
        if "vae" in variance:
          kwargs["use_vae"] = True
        if "energy_add_vae" in variance:
          kwargs["vae_usage"] = "energy_add"
        if "energy_mul_vae" in variance:
          kwargs["vae_usage"] = "energy_mul"
        if "test" in variance:
          kwargs["test"] = True
        if "no_dropout" in variance:
          kwargs["dropout"] = 0.0
        if "no_speaker_emb" in variance:
          kwargs["skip_speaker_embeddings"] = True
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=var_training_datasets,
          batch_size=12000 if "vae" in variance else 18000,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          use_pitch_pred=("f0" in variance),
          use_energy_pred=("energy" in variance),
          **kwargs,
        )
        if upsampling == "gauss":
          train_config.config["learning_rates"] = [0.0001, 0.001]
        train_job = tts_training(
          config=train_config,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          prefix=exp_name,
          num_epochs=200,
          mem=32 if variance in ["f0", "energy", "energy_add_vae", "energy_mul_vae"] else 8,
        )
        forward_dataset = deepcopy(training_datasets.cv)
        forward_datastreams = deepcopy(training_datasets.datastreams)
        gl_swer_kwargs = deepcopy(kwargs)
        if "vae" in variance:
          forward_datastreams["audio_features"].available_for_inference = True
          gl_swer_kwargs["use_audio_data"] = True

        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root,
          forward_dataset=TTSForwardData(dataset=forward_dataset, datastreams=forward_datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=True,
          use_pitch_pred=("f0" in variance),
          use_energy_pred=("energy" in variance),
          **gl_swer_kwargs,
        )
        if variance == "no_dropout_small":  # training broke on those points
          if upsampling == "gauss":
            checkpoint = train_job.out_checkpoints[83]
          else:
            checkpoint = train_job.out_checkpoints[138]
        else:
          checkpoint = train_job.out_checkpoints[200]
        gl_swer(
          name=exp_name + "/gl_swer",
          vocoder=default_vocoder,
          returnn_root=returnn_root,
          returnn_exe=returnn_exe,
          checkpoint=checkpoint,
          config=forward_config,
        )
        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root,
          forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=variance != "no_speaker_emb",
          use_pitch_pred=("f0" in variance),
          use_energy_pred=("energy" in variance),
          use_true_durations=True,
          **gl_swer_kwargs,
        )
        gl_swer(
          name=exp_name + "/gl_swer_true_durations",
          vocoder=default_vocoder,
          returnn_root=returnn_root,
          returnn_exe=returnn_exe,
          checkpoint=train_job.out_checkpoints[200],
          config=forward_config,
        )
        if "no_speaker_emb" not in variance:
          speaker_embedding_hdf = build_speaker_embedding_dataset(
            returnn_common_root=returnn_common_root,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            datasets=var_training_datasets,
            prefix=exp_name,
            train_job=train_job,
          )
        else:
          speaker_embedding_hdf = None

        if "vae" in variance:
          vae_dataset = deepcopy(training_datasets.cv)
          vae_dataset.datasets["audio"]["segment_file"] = None
          vae_datastreams = deepcopy(training_datasets.datastreams)
          vae_datastreams["audio_features"].available_for_inference = True
          speaker_prior_hdf = build_vae_speaker_prior_dataset(
            returnn_common_root=returnn_common_root,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            dataset=vae_dataset,
            datastreams=vae_datastreams,
            prefix=exp_name,
            train_job=train_job,
            corpus=reference_corpus.corpus_file,
          )
        if "log" in variance:
          continue
        # for synth_method in ["pred", "cheat_dur", "cheat_f0", "cheat_energy_cheat_dur", "cheat_f0_cheat_dur", "scale_en"]:
        for synth_method in ["pred", "cheat_dur", "cheat_f0", "cheat_energy_cheat_dur"]:
          if synth_method == "scale_en" and not variance == "energy_test":
            continue
          synth_kwargs = deepcopy(kwargs)
          if "cheat_f0" in synth_method:
            if "f0" not in variance:
              continue
            synth_kwargs["use_true_pitch"] = True
          if "cheat_energy" in synth_method:
            if "energy" not in variance:
              continue
            synth_kwargs["use_true_energy"] = True
          if "vae" in variance:
            synth_kwargs["use_calculated_prior"] = True
          if synth_method == "scale_en":
            synth_kwargs["energy_scale"] = 0.5
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=var_training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf if "no_speaker_emb" not in variance else None,
            durations=durations_hdf if "cheat_dur" in synth_method else None,
            process_corpus=False,
            speaker_prior_hdf=speaker_prior_hdf if "vae" in variance else None,
            pitch_hdf=f0_hdf if "cheat_f0" in synth_method else None,
            energy_hdf=energy_hdf if "cheat_energy" in synth_method else None,
          )

          if variance == "no_dropout_small":  # training broke on those points
            if upsampling == "gauss":
              checkpoint = train_job.out_checkpoints[83]
            else:
              checkpoint = train_job.out_checkpoints[138]
          else:
            checkpoint = train_job.out_checkpoints[200]
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{synth_method}",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root,
            checkpoint=checkpoint,
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=("cheat_dur" in synth_method),
            use_pitch_pred=("f0" in variance),
            use_energy_pred=("energy" in variance),
            energy_cheat=("cheat_energy" in synth_method),
            pitch_cheat=("cheat_f0" in synth_method),
            **synth_kwargs,
          )
          synthetic_data_dict[f"{align_name}_{variance}_{upsampling}_{synth_method}"] = synth_corpus
          if upsampling == "gauss" and synth_method in ["cheat_dur, pred"]:
            calculate_feature_variance(
              train_job=train_job,
              corpus=new_corpus,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root,
              prefix=exp_name + f"/{synth_method}",
              training_datasets=training_datasets,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=("cheat_dur" in synth_method),
              use_pitch_pred=("f0" in variance),
              use_energy_pred=("energy" in variance),
              energy_cheat=("cheat_energy" in synth_method),
              pitch_cheat=("cheat_f0" in synth_method),
              durations=durations_hdf if synth_method == "cheat_dur" else None,
              original_durations=durations_hdf,
              **synth_kwargs,
            )
            forward_config = get_forward_config(
              returnn_common_root=returnn_common_root_local,
              forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              calc_speaker_embedding=True,
              use_true_durations=(synth_method == "cheat_dur"),
              use_pitch_pred=("f0" in variance),
              use_energy_pred=("energy" in variance),
              energy_cheat=("cheat_energy" in synth_method),
              pitch_cheat=("cheat_f0" in synth_method),
            )
            gl_swer(
              name=exp_name + f"/gl_swer_{synth_method}",
              vocoder=default_vocoder,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              checkpoint=train_job.out_checkpoints[200],
              config=forward_config,
            )

          if variance == "energy_test" and False:
            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{synth_method}_norm_fix",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root,
              checkpoint=checkpoint,
              vocoder=default_vocoder,
              embedding_size=256,
              speaker_embedding_size=256,
              gauss_up=(upsampling == "gauss"),
              use_true_durations=("cheat_dur" in synth_method),
              use_pitch_pred=("f0" in variance),
              use_energy_pred=("energy" in variance),
              energy_cheat=("cheat_energy" in synth_method),
              pitch_cheat=("cheat_f0" in synth_method),
              reconstruction_norm=False,
              **synth_kwargs,
            )
            synthetic_data_dict[f"{align_name}_{variance}_{upsampling}_{synth_method}_norm_fix"] = synth_corpus

  return synthetic_data_dict


def gmm_side_experiments(alignments: Dict, rasr_allophones):
  """
    Done
    :param alignments:
    :param rasr_allophones:
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
  synthetic_data_dict = {}
  job_splits = 10
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(
    name="experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/vocoder/"
  )
  for align_name in ["tts_align_sat"]:
    alignment = deepcopy(alignments[align_name])
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
    (training_datasets, vocoder_data, new_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
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
    returnn_exe = tk.Path(
      "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
      hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    for variance in [
      # "f0_scale_0.5", REMOVED because of space
      # "f0_scale_2.0", REMOVED because of space
    ]:
      # for upsampling in ["repeat", "gauss"]:
      for upsampling in ["repeat"]:
        exp_name = name + f"/{variance}/{upsampling}"
        var_training_datasets = deepcopy(training_datasets)
        f0_hdf = None
        energy_hdf = None
        speaker_prior_hdf = None
        if "f0" in variance:
          f0_hdf = get_ls_100_f0_hdf(
            durations=durations_hdf,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
          )
          var_training_datasets = extend_meta_datasets_with_f0(datasets=training_datasets, f0_dataset=f0_hdf)
        if "energy" in variance:
          energy_hdf = get_ls_100_energy_hdf(
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
            center=False,
            log_norm="log_energy" in variance,
          )
          var_training_datasets = extend_meta_datasets_with_energy(var_training_datasets, energy_dataset=energy_hdf)
        kwargs = {}
        if "log_energy" in variance:
          kwargs["log_energy"] = True
        if "vae" in variance:
          kwargs["use_vae"] = True
        if "energy_add_vae" in variance:
          kwargs["vae_usage"] = "energy_add"
        if "energy_mul_vae" in variance:
          kwargs["vae_usage"] = "energy_mul"
        if "test" in variance:
          kwargs["test"] = True
        if "no_dropout" in variance:
          kwargs["dropout"] = 0.0
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=var_training_datasets,
          batch_size=12000 if "vae" in variance else 18000,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          use_pitch_pred=("f0" in variance),
          use_energy_pred=("energy" in variance),
          **kwargs,
        )
        if upsampling == "gauss":
          train_config.config["learning_rates"] = [0.0001, 0.001]
        train_job = tts_training(
          config=train_config,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          prefix=exp_name,
          num_epochs=200 if "300" not in variance else 300,
          mem=8 if "test" in variance or "small" in variance else 32,
        )
        forward_dataset = deepcopy(training_datasets.cv)
        forward_datastreams = deepcopy(training_datasets.datastreams)
        gl_swer_kwargs = deepcopy(kwargs)
        if "vae" in variance:
          forward_datastreams["audio_features"].available_for_inference = True
          gl_swer_kwargs["use_audio_data"] = True

        forward_config = get_forward_config(
          returnn_common_root=returnn_common_root,
          forward_dataset=TTSForwardData(dataset=forward_dataset, datastreams=forward_datastreams),
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=(upsampling == "gauss"),
          calc_speaker_embedding=True,
          use_pitch_pred=("f0" in variance),
          use_energy_pred=("energy" in variance),
          **gl_swer_kwargs,
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
          train_job=train_job,
        )
        if "vae" in variance:
          vae_dataset = deepcopy(training_datasets.cv)
          vae_dataset.datasets["audio"]["segment_file"] = None
          vae_datastreams = deepcopy(training_datasets.datastreams)
          vae_datastreams["audio_features"].available_for_inference = True
          speaker_prior_hdf = build_vae_speaker_prior_dataset(
            returnn_common_root=returnn_common_root,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            dataset=vae_dataset,
            datastreams=vae_datastreams,
            prefix=exp_name,
            train_job=train_job,
            corpus=reference_corpus.corpus_file,
          )
        if "log" in variance:
          continue
        for synth_method in ["pred"]:
          synth_kwargs = deepcopy(kwargs)
          if "cheat_f0" in synth_method:
            if "f0" not in variance:
              continue
            synth_kwargs["use_true_pitch"] = True
          if "cheat_energy" in synth_method:
            if "energy" not in variance:
              continue
            synth_kwargs["use_true_energy"] = True
          if "vae" in variance:
            synth_kwargs["use_calculated_prior"] = True
          if "f0_scale" in variance:
            if "2.0" in variance:
              synth_kwargs["pitch_scale"] = 2.0
            elif "0.5" in variance:
              synth_kwargs["pitch_scale"] = 0.5

          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=var_training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if "cheat_dur" in synth_method else None,
            process_corpus=False,
            speaker_prior_hdf=speaker_prior_hdf if "vae" in variance else None,
            pitch_hdf=f0_hdf if "cheat_f0" in synth_method else None,
            energy_hdf=energy_hdf if "cheat_energy" in synth_method else None,
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{synth_method}",
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
            use_true_durations=("cheat_dur" in synth_method),
            use_pitch_pred=("f0" in variance),
            use_energy_pred=("energy" in variance),
            energy_cheat=("cheat_energy" in synth_method),
            pitch_cheat=("cheat_f0" in synth_method),
            **synth_kwargs,
          )
          synthetic_data_dict[f"{align_name}_{variance}_{upsampling}_{synth_method}"] = synth_corpus
    returnn_common_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
      checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
    ).out_repository
    exp_name = name + f"/gauss_with_emb"
    train_config = get_training_config(
      returnn_common_root=returnn_common_root,
      training_datasets=training_datasets,
      embedding_size=256,
      speaker_embedding_size=256,
      gauss_up=True,
      gauss_with_speak_emb=True
    )
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
      forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
      embedding_size=256,
      speaker_embedding_size=256,
      gauss_up=True,
      calc_speaker_embedding=True,
      gauss_with_speak_emb=True
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
      datasets=training_datasets,
      prefix=exp_name,
      train_job=train_job,
    )
    for dur_pred in ["pred", "cheat"]:
      synth_dataset = get_inference_dataset(
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
        gauss_up=True,
        use_true_durations=(dur_pred == "cheat"),
        gauss_with_speak_emb=True,
      )
      synthetic_data_dict[f"{align_name}_gauss_with_emb_{dur_pred}"] = synth_corpus
  return synthetic_data_dict


def gmm_ablation_studies(alignments: Dict, rasr_allophones, return_trainings=False):
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
  trainings = {}
  job_splits = 10
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(
    name="experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/vocoder/"
  )
  for align_name, alignment in alignments.items():
    if "mono" in align_name:
      continue
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
    (training_datasets, vocoder_data, new_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
      name + "/datasets",
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      rasr_alignment=alignment,
      rasr_allophones=rasr_allophones,
    )
    for scale in [1.5, 2.0]:
      returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="45fad83c785a45fa4abfeebfed2e731dd96f960c",
      ).out_repository
      returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
        checkout_folder_name="returnn_common",
      ).out_repository
      for upsampling in ["gauss"]:
        exp_name = name + f"/{upsampling}/size_{scale}"
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=training_datasets,
          embedding_size=int(256 * scale),
          speaker_embedding_size=int(256 * scale),
          gauss_up=(upsampling == "gauss"),
          enc_lstm_size=int(256 * scale),
          dec_lstm_size=int(1024 * scale),
          hidden_dim=int(256 * scale),
          variance_dim=int(512 * scale),
          batch_size=12000 if scale < 2.1 else 6000,
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
        if return_trainings:
          trainings[f"{align_name}_{upsampling}_{str(scale)}"] = train_job
        for dur_pred in ["pred", "cheat_dur"]:
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=int(256 * scale),
            speaker_embedding_size=int(256 * scale),
            gauss_up=(upsampling == "gauss"),
            enc_lstm_size=int(256 * scale),
            dec_lstm_size=int(1024 * scale),
            hidden_dim=int(256 * scale),
            variance_dim=int(512 * scale),
            calc_speaker_embedding=True,
            use_true_durations=(dur_pred == "cheat_dur"),
          )
          gl_swer(
            name=exp_name + f"/gl_swer_{dur_pred}",
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
            datasets=training_datasets,
            prefix=exp_name,
            train_job=train_job,
            speaker_embedding_size=int(256 * scale)
          )
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat_dur" else None,
            process_corpus=False,
            speaker_embedding_size=int(256 * scale)
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
            embedding_size=int(256 * scale),
            speaker_embedding_size=int(256 * scale),
            gauss_up=(upsampling == "gauss"),
            enc_lstm_size=int(256 * scale),
            dec_lstm_size=int(1024 * scale),
            hidden_dim=int(256 * scale),
            variance_dim=int(512 * scale),
            use_true_durations=(dur_pred == "cheat_dur"),
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_{str(scale)}"] = synth_corpus

          # var_root = CloneGitRepositoryJob(
          #  "https://github.com/rwth-i6/returnn",
          #  commit="5cbec73032aeba2eec269624eb755fc5766f3c98",
          # ).out_repository
          # var_common_root = CloneGitRepositoryJob(
          #  "https://github.com/rwth-i6/returnn_common",
          #  commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
          #  checkout_folder_name="returnn_common",
          # ).out_repository
          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root,
            prefix=exp_name + f"/{dur_pred}",
            training_datasets=training_datasets,
            embedding_size=int(256 * scale),
            speaker_embedding_size=int(256 * scale),
            gauss_up=(upsampling == "gauss"),
            enc_lstm_size=int(256 * scale),
            dec_lstm_size=int(1024 * scale),
            hidden_dim=int(256 * scale),
            variance_dim=int(512 * scale),
            use_true_durations=(dur_pred == "cheat_dur"),
            durations=durations_hdf if dur_pred == "cheat_dur" else None,
            original_durations=durations_hdf,
          )

    exp_name = name + f"/gauss_big"
    train_config = get_training_config(
      returnn_common_root=returnn_common_root,
      training_datasets=training_datasets,
      embedding_size=256,
      speaker_embedding_size=256,
      gauss_up=True,
      big=True,
    )
    if True:
      train_config.config["learning_rates"] = [0.0001, 0.001]
    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      prefix=exp_name,
      num_epochs=200,
    )
    if return_trainings:
      trainings[f"{align_name}_gauss_big"] = train_job
    speaker_embedding_hdf = build_speaker_embedding_dataset(
      returnn_common_root=returnn_common_root,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      datasets=training_datasets,
      prefix=exp_name,
      train_job=train_job,
      speaker_embedding_size=256
    )
    for dur_pred in ["pred", "cheat"]:
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
        gauss_up=True,
        calc_speaker_embedding=True,
        use_true_durations=(dur_pred == "cheat"),
        big=True,
      )
      gl_swer(
        name=exp_name + f"/gl_swer_{dur_pred}",
        vocoder=default_vocoder,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
      )
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
        name=exp_name + f"/{dur_pred}_big",
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
        big=True,
      )
      synthetic_data_dict[f"{align_name}_gauss_{dur_pred}_big"] = synth_corpus
      calculate_feature_variance(
        train_job=train_job,
        corpus=new_corpus,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root,
        prefix=exp_name + f"/{dur_pred}_big",
        training_datasets=training_datasets,
        gauss_up=(upsampling == "gauss"),
        embedding_size=256,
        speaker_embedding_size=256,
        use_true_durations=(dur_pred == "cheat"),
        durations=durations_hdf if dur_pred == "cheat" else None,
        round_durations=False,
        big=True,
        original_durations = durations_hdf,
      )
  if return_trainings:
    return trainings
  return synthetic_data_dict


def train_tts_with_xvectors(alignments: Dict, rasr_allophones, speaker_embeddings: Dict):

  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  synthetic_data_dict = {}
  job_splits = 10
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(
    name="experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_duration_cheat/vocoder/"
  )
  for align_name, alignment in alignments.items():
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_xvectors/{align_name}"
    if "mono" in align_name:
      continue
    # Todo: Embedding and Speaker Embedding on same dim for comparability
    for speaker_embedding_size, speaker_embedding in speaker_embeddings.items():
      if speaker_embedding_size == "resemblyzer":
        speaker_embedding_size_int = 256
      else:
        speaker_embedding_size_int = speaker_embedding_size
      (training_datasets, vocoder_data, new_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
        name + "/datasets",
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        rasr_alignment=alignment,
        rasr_allophones=rasr_allophones,
        silence_prep=True,
        speaker_embeddings=(speaker_embedding_size_int, speaker_embedding)
      )
      for upsampling in ["gauss"]:
        exp_name = name + f"/{str(speaker_embedding_size)}/{upsampling}"
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=training_datasets,
          embedding_size=speaker_embedding_size_int,
          speaker_embedding_size=speaker_embedding_size_int,
          gauss_up=(upsampling == "gauss"),
          xvectors=True,
        )
        if speaker_embedding_size == "resemblyzer":
          train_config.config["learning_rates"] = [0.0001] * 2 + [0.001]
        elif upsampling == "gauss":
          train_config.config["learning_rates"] = [0.0001, 0.001]

        train_job = tts_training(
          config=train_config,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          prefix=exp_name,
          num_epochs=200,
        )
        speaker_embedding_hdf = speaker_embedding
        for dur_pred in ["pred", "cheat"]:
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=speaker_embedding_size_int,
            speaker_embedding_size=speaker_embedding_size_int,
            gauss_up=(upsampling == "gauss"),
            calc_speaker_embedding=True,
            xvectors=True,
            use_true_durations=(dur_pred == "cheat"),
          )
          gl_swer(
            name=exp_name + f"/gl_swer_{dur_pred}",
            vocoder=default_vocoder,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            checkpoint=train_job.out_checkpoints[200],
            config=forward_config,
          )
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
            speaker_embedding_size=speaker_embedding_size_int
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
            embedding_size=speaker_embedding_size_int,
            speaker_embedding_size=speaker_embedding_size_int,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            xvectors=True,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_xvec_{str(speaker_embedding_size)}"] = synth_corpus
          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root,
            prefix=exp_name + f"/{dur_pred}",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=speaker_embedding_size_int,
            speaker_embedding_size=speaker_embedding_size_int,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            xvectors=True,
            speaker_embedding_hdf=speaker_embedding_hdf,
            original_speakers=False,
            original_durations=durations_hdf,
          )
          if not "sat" in align_name:
            continue
          synth_dataset = get_inference_dataset(
            new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations_hdf if dur_pred == "cheat" else None,
            process_corpus=False,
            speaker_embedding_size=speaker_embedding_size_int,
            original_speakers=True
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}_orig_emb",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-100",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root,
            checkpoint=train_job.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=speaker_embedding_size_int,
            speaker_embedding_size=speaker_embedding_size_int,
            gauss_up=(upsampling == "gauss"),
            use_true_durations=(dur_pred == "cheat"),
            xvectors=True,
          )
          synthetic_data_dict[f"{align_name}_{upsampling}_{dur_pred}_xvec_{str(speaker_embedding_size)}_orig_emb"] = synth_corpus

          calculate_feature_variance(
            train_job=train_job,
            corpus=new_corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root,
            prefix=exp_name + f"/{dur_pred}_orig_emb",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=speaker_embedding_size_int,
            speaker_embedding_size=speaker_embedding_size_int,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations_hdf if dur_pred == "cheat" else None,
            xvectors=True,
            speaker_embedding_hdf=speaker_embedding_hdf,
            original_speakers=True,
            original_durations=durations_hdf,
          )
  return synthetic_data_dict
