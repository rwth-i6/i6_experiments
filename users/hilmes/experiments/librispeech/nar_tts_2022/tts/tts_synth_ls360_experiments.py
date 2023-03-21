from sisyphus import *
from typing import Dict
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.corpus.segments import SplitSegmentFileJob, SegmentCorpusJob
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
  get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
  synthesize_with_splits,
  build_speaker_embedding_dataset,
  build_vae_speaker_prior_dataset,
  calculate_feature_variance,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  get_inference_dataset,
  get_ls360_100h_data,
  get_tts_data_from_rasr_alignment,
  get_ls860_data,
  get_tts_data_from_ctc_align
)
from copy import deepcopy
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import AddSpeakerTagsFromMappingJob


def synthesize_100h_ls_360(trainings: Dict[str, ReturnnTrainingJob], alignments: Dict, rasr_allophones):

  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
  ).out_repository
  returnn_common_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  for align_name in ["tts_align_sat"]:
    alignment = alignments[align_name]
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
    (training_datasets, vocoder_data, original_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
      name + "/datasets",
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      rasr_alignment=alignment,
      rasr_allophones=rasr_allophones,
    )
    returnn_exe = tk.Path(
      "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
      hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn",
      commit="882b63641be44d1f91566ac6d5aa517707de0c43",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
      checkout_folder_name="returnn_common",
    ).out_repository
    corpus, segments = get_ls360_100h_data()
    segment_list = SplitSegmentFileJob(segments, concurrent=10).out_single_segments
    name = "experiments/librispeech/nar_tts_2022/tts/tts_synth_ls360_experiments/100h/"
    reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-360"]
    default_vocoder = get_default_vocoder(name=name)
    synthetic_data_dict = {}
    job_splits = 10
    for train_name, training in trainings.items():
      if "_2.0" in train_name:
        scale = 2.0
      elif "_1.5" in train_name:
        scale = 1.5
      else:
        scale = 1.0
      speaker_prior_hdf = None
      exp_name = name + train_name
      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=training,
        speaker_embedding_size=int(256 * scale)
      )
      if "vae" in train_name:
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
          train_job=training,
          corpus=reference_corpus.corpus_file,
        )
      kwargs = {}
      if "log_energy" in train_name:
        kwargs["log_energy"] = True
      if "vae" in train_name:
        kwargs["use_vae"] = True
      if "energy_add_vae" in train_name:
        kwargs["vae_usage"] = "energy_add"
      if "energy_mul_vae" in train_name:
        kwargs["vae_usage"] = "energy_mul"
      if "test" in train_name:
        kwargs["test"] = True
      if "no_dropout" in train_name:
        kwargs["dropout"] = 0.0
      if "big" in train_name:
        kwargs["big"] = True
      for synth_method in ["pred"]:
        synth_kwargs = deepcopy(kwargs)
        if "vae" in train_name:
          synth_kwargs["use_calculated_prior"] = True
        if scale == 1.0:
          synth_dataset = get_inference_dataset(
            corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            process_corpus=False,
            speaker_prior_hdf=speaker_prior_hdf if "vae" in train_name else None,
            original_corpus=original_corpus,
            segments=segments,
          )
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{synth_method}",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-360",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root if not "gauss" in train_name else returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root if not "gauss" in train_name else returnn_common_root_local,
            checkpoint=training.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=("gauss" in train_name),
            use_true_durations=("cheat_dur" in synth_method),
            use_pitch_pred=("f0" in train_name),
            use_energy_pred=("energy" in train_name),
            energy_cheat=("cheat_energy" in synth_method),
            pitch_cheat=("cheat_f0" in synth_method),
            segments=segment_list,
            **synth_kwargs,
          )
          if "big" in train_name:
            calculate_feature_variance(
              train_job=training,
              corpus=corpus,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_local,
              prefix=exp_name + f"/{synth_method}",
              training_datasets=training_datasets,
              gauss_up=("gauss" in train_name),
              embedding_size=256,
              speaker_embedding_size=256,
              use_true_durations=("cheat_dur" in synth_method),
              durations=None,
              original_corpus=original_corpus,
              segments=segments,
              big=True
            )
          else:
            calculate_feature_variance(
              train_job=training,
              corpus=corpus,
              returnn_root=returnn_root_local,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_local,
              prefix=exp_name + f"/{synth_method}",
              training_datasets=training_datasets,
              gauss_up=("gauss" in train_name),
              embedding_size=256,
              speaker_embedding_size=256,
              use_true_durations=("cheat_dur" in synth_method),
              durations=None,
              original_corpus=original_corpus,
              segments=segments,
            )
          synthetic_data_dict[f"ls360/100h/{train_name}"] = synth_corpus
        else:
          synth_dataset = get_inference_dataset(
            corpus,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            process_corpus=False,
            speaker_prior_hdf=speaker_prior_hdf if "vae" in train_name else None,
            original_corpus=original_corpus,
            segments=segments,
            speaker_embedding_size=int(256 * scale),
          )
          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{synth_method}",
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-360",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            checkpoint=training.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=int(256 * scale),
            speaker_embedding_size=int(256 * scale),
            gauss_up=("gauss" in train_name),
            use_true_durations=("cheat_dur" in synth_method),
            use_pitch_pred=("f0" in train_name),
            use_energy_pred=("energy" in train_name),
            energy_cheat=("cheat_energy" in synth_method),
            pitch_cheat=("cheat_f0" in synth_method),
            segments=segment_list,
            enc_lstm_size=int(256 * scale),
            dec_lstm_size=int(1024 * scale),
            hidden_dim=int(256 * scale),
            variance_dim=int(512 * scale),
            **synth_kwargs,
          )

          calculate_feature_variance(
            train_job=training,
            corpus=corpus,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            prefix=exp_name + f"/{synth_method}",
            training_datasets=training_datasets,
            embedding_size=int(256 * scale),
            speaker_embedding_size=int(256 * scale),
            gauss_up=("gauss" in train_name),
            enc_lstm_size=int(256 * scale),
            dec_lstm_size=int(1024 * scale),
            hidden_dim=int(256 * scale),
            variance_dim=int(512 * scale),
            use_true_durations=("cheat_dur" in synth_method),
            durations=None,
            segments=segments,
            original_corpus=original_corpus
          )

          synthetic_data_dict[f"ls360/100h/{train_name}"] = synth_corpus
    return synthetic_data_dict


def synthesize_ls_860(trainings: Dict[str, ReturnnTrainingJob], alignments: Dict, rasr_allophones):

  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
  ).out_repository
  returnn_common_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  for align_name in ["tts_align_sat"]:
    alignment = alignments[align_name]
    name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
    (training_datasets, vocoder_data, original_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
      name + "/datasets",
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      rasr_alignment=alignment,
      rasr_allophones=rasr_allophones,
    )
    returnn_exe = tk.Path(
      "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
      hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn",
      commit="882b63641be44d1f91566ac6d5aa517707de0c43",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
      checkout_folder_name="returnn_common",
    ).out_repository
    corpus, reference_corpus = get_ls860_data()
    name = "experiments/librispeech/nar_tts_2022/tts/tts_synth_ls860_experiments/"
    default_vocoder = get_default_vocoder(name=name)
    synthetic_data_dict = {}
    job_splits = 100
    for train_name, training in trainings.items():
      if "_2.0" in train_name:
        scale = 2.0
      elif "_1.5" in train_name:
        scale = 1.5
      else:
        scale = 1.0
      speaker_prior_hdf = None
      exp_name = name + train_name
      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=training,
        speaker_embedding_size=int(256 * scale)
      )
      kwargs = {}
      if "log_energy" in train_name:
        kwargs["log_energy"] = True
      if "vae" in train_name:
        kwargs["use_vae"] = True
      if "energy_add_vae" in train_name:
        kwargs["vae_usage"] = "energy_add"
      if "energy_mul_vae" in train_name:
        kwargs["vae_usage"] = "energy_mul"
      if "test" in train_name:
        kwargs["test"] = True
      if "no_dropout" in train_name:
        kwargs["dropout"] = 0.0
      for synth_method in ["pred"]:
        synth_kwargs = deepcopy(kwargs)
        if "vae" in train_name:
          synth_kwargs["use_calculated_prior"] = True
        synth_dataset = get_inference_dataset(
          corpus,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          datastreams=training_datasets.datastreams,
          speaker_embedding_hdf=speaker_embedding_hdf,
          process_corpus=False,
          speaker_prior_hdf=speaker_prior_hdf if "vae" in train_name else None,
          original_corpus=original_corpus,
          speaker_embedding_size=int(256 * scale),
        )
        synth_corpus = synthesize_with_splits(
          name=exp_name + f"/{synth_method}",
          reference_corpus=reference_corpus,
          corpus_name="ls860",
          job_splits=job_splits,
          datasets=synth_dataset,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          returnn_common_root=returnn_common_root_local,
          checkpoint=training.out_checkpoints[200],
          vocoder=default_vocoder,
          embedding_size=int(256 * scale),
          speaker_embedding_size=int(256 * scale),
          gauss_up=("gauss" in train_name),
          use_true_durations=("cheat_dur" in synth_method),
          use_pitch_pred=("f0" in train_name),
          use_energy_pred=("energy" in train_name),
          energy_cheat=("cheat_energy" in synth_method),
          pitch_cheat=("cheat_f0" in synth_method),
          enc_lstm_size=int(256 * scale),
          dec_lstm_size=int(1024 * scale),
          hidden_dim=int(256 * scale),
          variance_dim=int(512 * scale),
          **synth_kwargs,
        )
        """
        calculate_feature_variance(
          train_job=training,
          corpus=corpus,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          returnn_common_root=returnn_common_root_local,
          prefix=exp_name + f"/{synth_method}",
          training_datasets=training_datasets,
          embedding_size=int(256 * scale),
          speaker_embedding_size=int(256 * scale),
          gauss_up=("gauss" in train_name),
          enc_lstm_size=int(256 * scale),
          dec_lstm_size=int(1024 * scale),
          hidden_dim=int(256 * scale),
          variance_dim=int(512 * scale),
          use_true_durations=("cheat_dur" in synth_method),
          durations=None,
          original_corpus=original_corpus
        )
        """
        synthetic_data_dict[f"ls860/{train_name}"] = synth_corpus
    return synthetic_data_dict


def synthesize_ls_360(trainings: Dict[str, ReturnnTrainingJob], alignments: Dict, rasr_allophones):
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root_local = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_local = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
    ).out_repository
    for align_name in ["tts_align_sat"]:
        alignment = alignments[align_name]
        name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/gmm_align/{align_name}"
        (training_datasets, vocoder_data, original_corpus, durations_hdf,) = get_tts_data_from_rasr_alignment(
            name + "/datasets",
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            rasr_alignment=alignment,
            rasr_allophones=rasr_allophones,
        )
        returnn_exe = tk.Path(
            "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
            hash_overwrite="GENERIC_RETURNN_LAUNCHER",
        )
        returnn_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="2c0bf3666e721b86d843f2ef54cd416dfde20566",
        ).out_repository
        returnn_common_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
            checkout_folder_name="returnn_common",
        ).out_repository
        corpus, _ = get_ls360_100h_data()
        name = "experiments/librispeech/nar_tts_2022/tts/paper_sytn/360h/"
        reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-360"]
        default_vocoder = get_default_vocoder(name=name)
        synthetic_data_dict = {}
        job_splits = 30
        for train_name, training in trainings.items():
            if train_name != "tts_align_sat/gauss/baseline":
                continue
            exp_name = name + train_name
            speaker_embedding_hdf = build_speaker_embedding_dataset(
                returnn_common_root=returnn_common_root,
                returnn_exe=returnn_exe,
                returnn_root=returnn_root,
                datasets=training_datasets,
                prefix=exp_name,
                train_job=training,
                speaker_embedding_size=256
            )
            for synth_method in ["pred"]:
                # Sat Gauss Pred
                synth_dataset, speaker_mapping = get_inference_dataset(
                        corpus,
                        returnn_root=returnn_root_local,
                        returnn_exe=returnn_exe,
                        datastreams=training_datasets.datastreams,
                        speaker_embedding_hdf=speaker_embedding_hdf,
                        process_corpus=False,
                        speaker_prior_hdf=None,
                        original_corpus=original_corpus,
                        segments=None,
                        speaker_embedding_size=256,
                        return_mapping=True
                    )
                synth_corpus = synthesize_with_splits(
                        name=exp_name,
                        reference_corpus=reference_corpus.corpus_file,
                        corpus_name="train-clean-360",
                        job_splits=job_splits,
                        datasets=synth_dataset,
                        returnn_root=returnn_root_local,
                        returnn_exe=returnn_exe,
                        returnn_common_root=returnn_common_root_local,
                        checkpoint=training.out_checkpoints[200],
                        vocoder=default_vocoder,
                        embedding_size=256,
                        speaker_embedding_size=256,
                        gauss_up=("gauss" in train_name),
                        use_true_durations=False,
                        enc_lstm_size=256,
                        dec_lstm_size=1024,
                        hidden_dim=256,
                        variance_dim=512,
                )
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
                tag_corpus_job.add_alias(exp_name + "/pred" + "/add_tags")
                tk.register_output("paper_nick/21_02_23_ls360/pred_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping,
                        prefix="synth_")
                tag_corpus_job.add_alias(exp_name + "/pred" + "/add_tags_synth")
                tk.register_output("paper_nick/21_02_23_ls360/pred_synth_prefix_tags_corpus.xml.gz",
                    tag_corpus_job.out_corpus)

                # Sat Gauss Scale 1.1
                synth_dataset, speaker_mapping = get_inference_dataset(
                    corpus,
                    returnn_root=returnn_root_local,
                    returnn_exe=returnn_exe,
                    datastreams=training_datasets.datastreams,
                    speaker_embedding_hdf=speaker_embedding_hdf,
                    process_corpus=False,
                    speaker_prior_hdf=None,
                    original_corpus=original_corpus,
                    segments=None,
                    speaker_embedding_size=256,
                    return_mapping=True
                )
                synth_corpus = synthesize_with_splits(
                    name=exp_name + f"_duration_scale_1.1",
                    reference_corpus=reference_corpus.corpus_file,
                    corpus_name="train-clean-360",
                    job_splits=job_splits,
                    datasets=synth_dataset,
                    returnn_root=returnn_root_local,
                    returnn_exe=returnn_exe,
                    returnn_common_root=returnn_common_root_local,
                    checkpoint=training.out_checkpoints[200],
                    vocoder=default_vocoder,
                    embedding_size=256,
                    speaker_embedding_size=256,
                    gauss_up=("gauss" in train_name),
                    use_true_durations=False,
                    enc_lstm_size=256,
                    dec_lstm_size=1024,
                    hidden_dim=256,
                    variance_dim=512,
                    duration_scale=1.1,
                )
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
                tag_corpus_job.add_alias(exp_name + "/pred_scale_1.1" + "/add_tags")
                tk.register_output("paper_nick/21_02_23_ls360/scale_1.1_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping,
                    prefix="synth_")
                tag_corpus_job.add_alias(exp_name + "/pred_scale_1.1" + "/add_tags_synth")
                tk.register_output("paper_nick/21_02_23_ls360/scale_1.1_synth_prefix_tags_corpus.xml.gz",
                    tag_corpus_job.out_corpus)

                synth_corpus = synthesize_with_splits(
                    name=exp_name + f"_duration_scale_1.05",
                    reference_corpus=reference_corpus.corpus_file,
                    corpus_name="train-clean-360",
                    job_splits=job_splits,
                    datasets=synth_dataset,
                    returnn_root=returnn_root_local,
                    returnn_exe=returnn_exe,
                    returnn_common_root=returnn_common_root_local,
                    checkpoint=training.out_checkpoints[200],
                    vocoder=default_vocoder,
                    embedding_size=256,
                    speaker_embedding_size=256,
                    gauss_up=("gauss" in train_name),
                    use_true_durations=False,
                    enc_lstm_size=256,
                    dec_lstm_size=1024,
                    hidden_dim=256,
                    variance_dim=512,
                    duration_scale=1.05,
                )
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
                tag_corpus_job.add_alias(exp_name + "/pred_scale_1.05" + "/add_tags")
                tk.register_output("paper_nick/21_02_23_ls360/scale_1.05_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
                tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping,
                    prefix="synth_")
                tag_corpus_job.add_alias(exp_name + "/pred_scale_1.05" + "/add_tags_synth")
                tk.register_output("paper_nick/21_02_23_ls360/scale_1.05_synth_prefix_tags_corpus.xml.gz",
                    tag_corpus_job.out_corpus)

                # Sat Gauss Random Walk
                for mean in [1.0, 1.05]:
                    for std in [0.0125, 0.025, 0.0375, 0.05]:
                        synth_dataset, speaker_mapping = get_inference_dataset(
                                corpus,
                                returnn_root=returnn_root_local,
                                returnn_exe=returnn_exe,
                                datastreams=training_datasets.datastreams,
                                speaker_embedding_hdf=speaker_embedding_hdf,
                                process_corpus=False,
                                speaker_prior_hdf=None,
                                original_corpus=original_corpus,
                                segments=None,
                                speaker_embedding_size=256,
                                return_mapping=True
                            )
                        synth_corpus = synthesize_with_splits(
                                name=exp_name + f"/_{mean}_{std}",
                                reference_corpus=reference_corpus.corpus_file,
                                corpus_name="train-clean-360",
                                job_splits=job_splits,
                                datasets=synth_dataset,
                                returnn_root=returnn_root_local,
                                returnn_exe=returnn_exe,
                                returnn_common_root=returnn_common_root_local,
                                checkpoint=training.out_checkpoints[200],
                                vocoder=default_vocoder,
                                embedding_size=256,
                                speaker_embedding_size=256,
                                gauss_up=("gauss" in train_name),
                                use_true_durations=False,
                                enc_lstm_size=256,
                                dec_lstm_size=1024,
                                hidden_dim=256,
                                variance_dim=512,
                                random_duration_scaling=[mean, std, 0.9, 1.2],
                                redo=True,
                        )
                        tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
                        tag_corpus_job.add_alias(exp_name + f"/_{mean}_{std}" + "/add_tags")
                        tk.register_output(f"paper_nick/21_02_23_ls360/random_{mean}_{std}_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
                        tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping,
                                prefix="synth_")
                        tag_corpus_job.add_alias(exp_name + f"/_{mean}_{std}" + "/add_tags_synth")
                        tk.register_output(f"paper_nick/21_02_23_ls360/random_{mean}_{std}_synth_prefix_tags_corpus.xml.gz",
                            tag_corpus_job.out_corpus)

    for align_name in ["ctc_0.5", "ctc_0.0"]:
      alignment = alignments[align_name]
      name = f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/ctc_align/{align_name}"
      returnn_exe_loc = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
      )
      returnn_root_loc = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="240f119b54d52a4324ab300c301f8e003e0a398c",
      ).out_repository
      (training_datasets, original_corpus, durations_hdf) = get_tts_data_from_ctc_align(
        name + "/datasets",
        returnn_exe=returnn_exe_loc,
        returnn_root=returnn_root_loc,
        alignment=alignment
      )
      for train_name, training in trainings.items():
        if train_name != align_name:
          continue
        exp_name = name + train_name
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          datasets=training_datasets,
          prefix=exp_name,
          train_job=training,
          speaker_embedding_size=256
        )
        for synth_method in ["pred"]:
          # CTC Pred
          synth_dataset, speaker_mapping = get_inference_dataset(
            corpus,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            process_corpus=False,
            speaker_prior_hdf=None,
            original_corpus=original_corpus,
            segments=None,
            speaker_embedding_size=256,
            return_mapping=True
          )
          synth_corpus = synthesize_with_splits(
            name=exp_name,
            reference_corpus=reference_corpus.corpus_file,
            corpus_name="train-clean-360",
            job_splits=job_splits,
            datasets=synth_dataset,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            checkpoint=training.out_checkpoints[200],
            vocoder=default_vocoder,
            embedding_size=256,
            speaker_embedding_size=256,
            gauss_up=("gauss" in train_name),
            use_true_durations=False,
            enc_lstm_size=256,
            dec_lstm_size=1024,
            hidden_dim=256,
            variance_dim=512,
          )
          tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping)
          tag_corpus_job.add_alias(exp_name + "/pred" + "/add_tags")
          tk.register_output("paper_nick/27_02_23_ls360/ctc_pred_real_tags_corpus.xml.gz", tag_corpus_job.out_corpus)
          tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=synth_corpus, mapping=speaker_mapping,
            prefix="synth_")
          tag_corpus_job.add_alias(exp_name + "/pred" + "/add_tags_synth")
          tk.register_output("paper_nick/27_02_23_ls360/ctc_pred_synth_prefix_tags_corpus.xml.gz",
            tag_corpus_job.out_corpus)