"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  get_tts_data_from_ctc_align,
  get_librispeech_tts_segments,
  get_ls_train_clean_100_tts_silencepreprocessed
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.ctc_align.ctc_experiments import (
  get_baseline_ctc_alignment,
  get_loss_scale_alignments,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.tts.tts_pipeline import (
  get_training_config,
  tts_training,
  get_forward_config,
  gl_swer,
  synthesize_with_splits,
  build_speaker_embedding_dataset,
  build_vae_speaker_prior_dataset,
  tts_forward,
  calculate_feature_variance
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.default_vocoder import (
  get_default_vocoder,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
  TTSForwardData,
  get_inference_dataset_old,
  get_inference_dataset,
  get_ls_100_f0_hdf,
  extend_meta_datasets_with_f0,
  extend_meta_datasets_with_energy,
  get_ls_100_energy_hdf,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.util.asr_evaluation import (
  asr_evaluation,
)
from copy import deepcopy


def ctc_baseline(basic_experiments=False, return_trainings=False):
  """
    baseline for returnn_common ctc model with network constructor
    :return:
    """
  trainings={}
  returnn_exe = tk.Path(
    "/u/hilmes/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root_job = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="240f119b54d52a4324ab300c301f8e003e0a398c",
  )
  returnn_root_job.hash_overwrite = "ctc_baseline_returnn"
  returnn_root = returnn_root_job.out_repository
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="79876b18552f61a3af7c21c670475fee51ef3991",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  returnn_common_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/ctc_baseline"
  alignment = get_baseline_ctc_alignment()
  training_datasets, corpus, durations = get_tts_data_from_ctc_align(
    name + "/datasets",
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    alignment=alignment,
  )
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(name=name)
  synthetic_data_dict = {}
  job_splits = 10

  librispeech_trafo = tk.Path("/work/smt4/hilmes/swer_model/returnn2.config")
  train_segments, cv_segments = get_librispeech_tts_segments()
  asr_evaluation(
    config_file=librispeech_trafo,
    corpus=get_ls_train_clean_100_tts_silencepreprocessed().corpus_file,
    output_path="real_swer",
    returnn_root=returnn_root,
    returnn_python_exe=returnn_exe,
    segment_file=cv_segments,
  )
  ls_no_sil_prep = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")['train-clean-100']
  asr_evaluation(
    config_file=librispeech_trafo,
    corpus=ls_no_sil_prep.corpus_file,
    output_path="real_swer_no_sil_p",
    returnn_root=returnn_root,
    returnn_python_exe=returnn_exe,
    segment_file=cv_segments,
  )
  ls_vocoding_only = synthesize_ls_100_features()
  asr_evaluation(
    config_file=librispeech_trafo,
    corpus=ls_vocoding_only,
    output_path="vocoding_only",
    returnn_root=returnn_root,
    returnn_python_exe=returnn_exe,
    segment_file=cv_segments,
  )
  ls_vocoding_only_no_sil_p = synthesize_ls_100_features(silence_prep=False)
  asr_evaluation(
    config_file=librispeech_trafo,
    corpus=ls_vocoding_only_no_sil_p,
    output_path="vocoding_only_no_sil_p",
    returnn_root=returnn_root,
    returnn_python_exe=returnn_exe,
    segment_file=cv_segments,
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
    trainings["ctc_0.5"] = train_job
    if return_trainings:
      return trainings
    speaker_embedding_hdf = build_speaker_embedding_dataset(
      returnn_common_root=returnn_common_root,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      datasets=training_datasets,
      prefix=exp_name,
      train_job=train_job,
    )
    if upsampling == "gauss":
      synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=None,
        process_corpus=False,
      )
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root,
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
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
      )
      forward_hdf = forward_job.out_hdf_files["output.hdf"]
      tk.register_output(exp_name + "/dump_dur/durations.hdf", forward_hdf)
    synth_dataset = get_inference_dataset_old(
      corpus,
      returnn_root=returnn_root,
      returnn_exe=returnn_exe,
      datastreams=training_datasets.datastreams,
      speaker_embedding_hdf=speaker_embedding_hdf,
      durations=durations,
      process_corpus=False,
    )
    for duration in ["pred", "cheat"]:
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root_local,
        forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=(upsampling == "gauss"),
        calc_speaker_embedding=True,
        use_true_durations=(duration == "cheat"),
      )
      gl_swer(
        name=exp_name + f"gl_swer_{duration}",
        vocoder=default_vocoder,
        returnn_root=returnn_root_local,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
      )
      synth_corpus = synthesize_with_splits(
        name=exp_name + f"/{duration}",
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
        use_true_durations=(duration == "cheat"),
      )
      synthetic_data_dict[f"ctc_{upsampling}_{duration}"] = synth_corpus
      calculate_feature_variance(
          train_job=train_job,
          corpus=corpus,
          returnn_root=returnn_root_local,
          returnn_exe=returnn_exe,
          returnn_common_root=returnn_common_root_local,
          prefix=exp_name + f"/{duration}",
          training_datasets=training_datasets,
          gauss_up=(upsampling == "gauss"),
          embedding_size=256,
          speaker_embedding_size=256,
          use_true_durations=(duration == "cheat"),
          durations=durations if duration == "cheat" else None,
          original_durations=durations,
      )
  for model_scale in [2]:
    exp_name = name + "_big_model" + f"/gauss"
    returnn_root_loc = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn",
      commit="45fad83c785a45fa4abfeebfed2e731dd96f960c",
    ).out_repository
    returnn_common_root_loc = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
      checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_local = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
    ).out_repository
    train_config = get_training_config(
      returnn_common_root=returnn_common_root_loc,
      training_datasets=training_datasets,
      embedding_size=int(256 * model_scale),
      speaker_embedding_size=int(256 * model_scale),
      gauss_up=True,
      enc_lstm_size=int(256 * model_scale),
      dec_lstm_size=int(1024 * model_scale),
      hidden_dim=int(256 * model_scale),
      variance_dim=int(512 * model_scale),
      batch_size=12000 if model_scale < 1.9 else 6000,
    )
    train_config.config["learning_rates"] = [0.0001, 0.001]

    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root_loc,
      prefix=exp_name,
      num_epochs=200,
    )

    speaker_embedding_hdf = build_speaker_embedding_dataset(
      returnn_common_root=returnn_common_root_loc,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root_loc,
      datasets=training_datasets,
      prefix=exp_name,
      train_job=train_job,
      speaker_embedding_size=int(256 * model_scale)
    )
    for dur_pred in ["pred", "cheat"]:
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root_loc,
        forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
        embedding_size=int(256 * model_scale),
        speaker_embedding_size=int(256 * model_scale),
        enc_lstm_size=int(256 * model_scale),
        dec_lstm_size=int(1024 * model_scale),
        hidden_dim=int(256 * model_scale),
        variance_dim=int(512 * model_scale),
        calc_speaker_embedding=True,
        gauss_up=True,
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
      synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root_loc,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations if dur_pred == "cheat" else None,
        process_corpus=False,
        speaker_embedding_size=int(256 * model_scale)
      )

      synth_corpus = synthesize_with_splits(
        name=exp_name + f"/{dur_pred}",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root_loc,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root_loc,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=int(256 * model_scale),
        speaker_embedding_size=int(256 * model_scale),
        gauss_up=True,
        enc_lstm_size=int(256 * model_scale),
        dec_lstm_size=int(1024 * model_scale),
        hidden_dim=int(256 * model_scale),
        variance_dim=int(512 * model_scale),
        use_true_durations=(dur_pred == "cheat"),
      )
      synthetic_data_dict[f"ctc_0_5_{model_scale}_{upsampling}_{dur_pred}"] = synth_corpus
      calculate_feature_variance(
        train_job=train_job,
        corpus=corpus,
        returnn_root=returnn_root_loc,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root_loc,
        prefix=exp_name + f"/{dur_pred}",
        training_datasets=training_datasets,
        embedding_size=int(256 * model_scale),
        speaker_embedding_size=int(256 * model_scale),
        gauss_up=True,
        enc_lstm_size=int(256 * model_scale),
        dec_lstm_size=int(1024 * model_scale),
        hidden_dim=int(256 * model_scale),
        variance_dim=int(512 * model_scale),
        use_true_durations=(dur_pred == "cheat"),
        durations=durations if dur_pred == "cheat" else None,
        original_durations=durations,
      )
  if basic_experiments:
    return synthetic_data_dict
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_job = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="e75ee20b5830808062aac2821512131fdffc521d",  # fix dim tag declare same as
  )
  returnn_root = returnn_root_job.out_repository
  exp_name = name + "/vae"
  vae_dataset = deepcopy(training_datasets)
  vae_dataset.datastreams["audio_features"].available_for_inference = True
  train_config = get_training_config(
    returnn_common_root=returnn_common_root,
    training_datasets=vae_dataset,
    embedding_size=256,
    speaker_embedding_size=256,
    gauss_up=False,
    use_vae=True,
    batch_size=12000,
  )
  train_job = tts_training(
    config=train_config,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    prefix=exp_name,
    num_epochs=200,
  )
  vae_swer_dataset = deepcopy(training_datasets.cv)
  vae_swer_datastreams = deepcopy(training_datasets.datastreams)
  vae_swer_datastreams["audio_features"].available_for_inference = True
  forward_config = get_forward_config(
    returnn_common_root=returnn_common_root,
    forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
    embedding_size=256,
    speaker_embedding_size=256,
    calc_speaker_embedding=True,
    use_vae=True,
    use_audio_data=True,
  )
  gl_swer(
    name=exp_name + "/gl_swer",
    vocoder=default_vocoder,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    checkpoint=train_job.out_checkpoints[200],
    config=forward_config,
  )
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_job = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="e75ee20b5830808062aac2821512131fdffc521d",  # fix dim tag declare same as
  )
  returnn_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  returnn_common_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root = returnn_root_job.out_repository
  #for mode in ["vae", "vae_no_speaker_emb"]:, REMOVED due to space
  for mode in []:
    exp_name = name + f"/{mode}_scale"
    vae_dataset = deepcopy(training_datasets)
    vae_dataset.datastreams["audio_features"].available_for_inference = True
    train_config = get_training_config(
      returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
      training_datasets=vae_dataset,
      embedding_size=256,
      speaker_embedding_size=256,
      gauss_up=False,
      use_vae=True,
      scale_kl_loss=True,
      batch_size=12000,
      skip_speaker_embeddings=("speaker_emb" in mode),
    )
    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
      prefix=exp_name,
      num_epochs=200,
    )
    vae_swer_dataset = deepcopy(training_datasets.cv)
    vae_swer_datastreams = deepcopy(training_datasets.datastreams)
    vae_swer_datastreams["audio_features"].available_for_inference = True
    forward_config = get_forward_config(
      returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
      forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
      embedding_size=256,
      speaker_embedding_size=256,
      calc_speaker_embedding=True,
      use_vae=True,
      use_audio_data=True,
      scale_kl_loss=True,
      skip_speaker_embeddings=("speaker_emb" in mode),
    )
    gl_swer(
      name=exp_name + "/gl_swer",
      vocoder=default_vocoder,
      returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local ,
      returnn_exe=returnn_exe,
      checkpoint=train_job.out_checkpoints[200],
      config=forward_config,
    )
    if "speaker_emb" in mode:
      speaker_embedding_hdf = None
    else:
      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job,
      )
    vae_dataset_fw = deepcopy(training_datasets.cv)
    vae_dataset_fw.datasets["audio"]["segment_file"] = None
    vae_datastreams_fw = deepcopy(training_datasets.datastreams)
    vae_datastreams_fw["audio_features"].available_for_inference = True
    speaker_prior_hdf = build_vae_speaker_prior_dataset(
      returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
      dataset=vae_dataset_fw,
      datastreams=vae_datastreams_fw,
      prefix=exp_name,
      train_job=train_job,
      corpus=reference_corpus.corpus_file,
      skip_speaker_embeddings=("speaker_emb" in mode),
    )
    for synth_method in ["pred"]:
      synth_kwargs = {"use_vae": True, "use_calculated_prior": True, "skip_speaker_embeddings": ("speaker_emb" in mode)}
      synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations if "cheat_dur" in synth_method else None,
        process_corpus=False,
        speaker_prior_hdf=speaker_prior_hdf,
        speaker_prior_size=256 if "speaker_emb" in mode else 32,
      )
      synth_corpus = synthesize_with_splits(
        name=exp_name + f"/{synth_method}",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
        use_true_durations=("cheat_dur" in synth_method),
        energy_cheat=("cheat_energy" in synth_method),
        pitch_cheat=("cheat_f0" in synth_method),
        **synth_kwargs,
      )
      synthetic_data_dict[f"ctc_{mode}_{1}_{synth_method}"] = synth_corpus

    exp_name = name + f"/{mode}_2.0"
    vae_dataset = deepcopy(training_datasets)
    vae_dataset.datastreams["audio_features"].available_for_inference = True
    train_config = get_training_config(
      returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
      training_datasets=vae_dataset,
      embedding_size=512,
      speaker_embedding_size=512,
      gauss_up=True,
      use_vae=True,
      scale_kl_loss=True,
      batch_size=4500,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=512,
      dec_lstm_size=2048,
      hidden_dim=512,
      variance_dim=1024,
    )
    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
      prefix=exp_name,
      num_epochs=200,
    )
    returnn_common_root_local = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
      checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_local = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
    ).out_repository
    vae_swer_dataset = deepcopy(training_datasets.cv)
    vae_swer_datastreams = deepcopy(training_datasets.datastreams)
    vae_swer_datastreams["audio_features"].available_for_inference = True
    forward_config = get_forward_config(
      returnn_common_root=returnn_common_root_local,
      forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
      embedding_size=512,
      speaker_embedding_size=512,
      calc_speaker_embedding=True,
      use_vae=True,
      scale_kl_loss=True,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=512,
      dec_lstm_size=2048,
      hidden_dim=512,
      variance_dim=1024,
      gauss_up=True,
      use_audio_data=True,
    )
    gl_swer(
      name=exp_name + "/gl_swer",
      vocoder=default_vocoder,
      returnn_root=returnn_root_local,
      returnn_exe=returnn_exe,
      checkpoint=train_job.out_checkpoints[200],
      config=forward_config,
    )
    if "speaker_emb" in mode:
      speaker_embedding_hdf = None
    else:
      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root_local,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root_local,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job,
        speaker_embedding_size=512
      )
    vae_dataset_fw = deepcopy(training_datasets.cv)
    vae_dataset_fw.datasets["audio"]["segment_file"] = None
    vae_datastreams_fw = deepcopy(training_datasets.datastreams)
    vae_datastreams_fw["audio_features"].available_for_inference = True
    speaker_prior_hdf = build_vae_speaker_prior_dataset(
      returnn_common_root=returnn_common_root_local,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root_local,
      dataset=vae_dataset_fw,
      datastreams=vae_datastreams_fw,
      prefix=exp_name,
      train_job=train_job,
      corpus=reference_corpus.corpus_file,
      embedding_size=512,
      speaker_embedding_size=512,
      calc_speaker_embedding=True,
      use_vae=True,
      scale_kl_loss=True,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=512,
      dec_lstm_size=2048,
      hidden_dim=512,
      variance_dim=1024,
      gauss_up=True,
    )
    for synth_method in ["pred"]:
      synth_kwargs = {"use_vae": True, "use_calculated_prior": True, "skip_speaker_embeddings": ("speaker_emb" in mode)}
      synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root_local,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations if "cheat_dur" in synth_method else None,
        process_corpus=False,
        speaker_prior_hdf=speaker_prior_hdf,
        speaker_prior_size=512 if "speaker_emb" in mode else 32,
        speaker_embedding_size=512,
      )
      synth_corpus = synthesize_with_splits(
        name=exp_name + f"/{synth_method}",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root_local,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root_local,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=512,
        speaker_embedding_size=512,
        enc_lstm_size=512,
        dec_lstm_size=2048,
        hidden_dim=512,
        variance_dim=1024,
        gauss_up=True,
        use_true_durations=("cheat_dur" in synth_method),
        energy_cheat=("cheat_energy" in synth_method),
        pitch_cheat=("cheat_f0" in synth_method),
        **synth_kwargs,
      )
      synthetic_data_dict[f"ctc_{mode}_{2.0}_{synth_method}"] = synth_corpus

    exp_name = name + f"/{mode}_1.5"
    vae_dataset = deepcopy(training_datasets)
    vae_dataset.datastreams["audio_features"].available_for_inference = True
    train_config = get_training_config(
      returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local ,
      training_datasets=vae_dataset,
      embedding_size=384,
      speaker_embedding_size=384,
      gauss_up=True,
      use_vae=True,
      scale_kl_loss=True,
      batch_size=4500,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=384,
      dec_lstm_size=1536,
      hidden_dim=384,
      variance_dim=768,
    )
    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
      prefix=exp_name,
      num_epochs=200,
    )
    returnn_common_root_local = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn_common",
      commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
      checkout_folder_name="returnn_common",
    ).out_repository
    returnn_root_local = CloneGitRepositoryJob(
      "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
    ).out_repository
    vae_swer_dataset = deepcopy(training_datasets.cv)
    vae_swer_datastreams = deepcopy(training_datasets.datastreams)
    vae_swer_datastreams["audio_features"].available_for_inference = True
    forward_config = get_forward_config(
      returnn_common_root=returnn_common_root_local,
      forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
      embedding_size=384,
      speaker_embedding_size=384,
      calc_speaker_embedding=True,
      use_vae=True,
      use_audio_data=True,
      scale_kl_loss=True,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=384,
      dec_lstm_size=1536,
      hidden_dim=384,
      variance_dim=768,
      gauss_up=True,
    )
    if "speaker_emb" in mode:
      speaker_embedding_hdf = None
    else:
      speaker_embedding_hdf = build_speaker_embedding_dataset(
        returnn_common_root=returnn_common_root_local,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root_local,
        datasets=training_datasets,
        prefix=exp_name,
        train_job=train_job,
        speaker_embedding_size=384
      )
    vae_dataset_fw = deepcopy(training_datasets.cv)
    vae_dataset_fw.datasets["audio"]["segment_file"] = None
    vae_datastreams_fw = deepcopy(training_datasets.datastreams)
    vae_datastreams_fw["audio_features"].available_for_inference = True
    speaker_prior_hdf = build_vae_speaker_prior_dataset(
      returnn_common_root=returnn_common_root_local,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root_local,
      dataset=vae_dataset_fw,
      datastreams=vae_datastreams_fw,
      prefix=exp_name,
      train_job=train_job,
      corpus=reference_corpus.corpus_file,
      skip_speaker_embeddings=("speaker_emb" in mode),
      enc_lstm_size=384,
      dec_lstm_size=1536,
      hidden_dim=384,
      variance_dim=768,
      gauss_up=True,
      embedding_size=384,
      speaker_embedding_size=384,
      calc_speaker_embedding=True,
      use_vae=True,
      scale_kl_loss=True,
    )
    for synth_method in []: # removed "pred" cause it was no converging
      synth_kwargs = {"use_vae": True, "use_calculated_prior": True, "skip_speaker_embeddings": ("speaker_emb" in mode)}
      synth_dataset = get_inference_dataset(
        corpus,
        returnn_root=returnn_root_local,
        returnn_exe=returnn_exe,
        datastreams=training_datasets.datastreams,
        speaker_embedding_hdf=speaker_embedding_hdf,
        durations=durations if "cheat_dur" in synth_method else None,
        process_corpus=False,
        speaker_prior_hdf=speaker_prior_hdf,
        speaker_prior_size=384 if "speaker_emb" in mode else 32,
        speaker_embedding_size=384,
      )
      synth_corpus = synthesize_with_splits(
        name=exp_name + f"/{synth_method}",
        reference_corpus=reference_corpus.corpus_file,
        corpus_name="train-clean-100",
        job_splits=job_splits,
        datasets=synth_dataset,
        returnn_root=returnn_root_local,
        returnn_exe=returnn_exe,
        returnn_common_root=returnn_common_root_local,
        checkpoint=train_job.out_checkpoints[200],
        vocoder=default_vocoder,
        embedding_size=384,
        speaker_embedding_size=384,
        enc_lstm_size=384,
        dec_lstm_size=1536,
        hidden_dim=384,
        variance_dim=768,
        gauss_up=True,
        use_true_durations=("cheat_dur" in synth_method),
        energy_cheat=("cheat_energy" in synth_method),
        pitch_cheat=("cheat_f0" in synth_method),
        **synth_kwargs,
      )
      synthetic_data_dict[f"ctc_{mode}_{1.5}_{synth_method}"] = synth_corpus

    # for loss in [0.1, 0.01, 0]:, REMOVED because of space
    for loss in []:
      exp_name = name + f"/{mode}_scale_ls_{loss}"
      train_config = get_training_config(
        returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
        training_datasets=vae_dataset,
        embedding_size=256,
        speaker_embedding_size=256,
        gauss_up=False,
        use_vae=True,
        scale_kl_loss=True,
        batch_size=12000,
        skip_speaker_embeddings=("speaker_emb" in mode),
        kl_beta=loss,
      )
      train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        prefix=exp_name,
        num_epochs=200,
      )
      vae_swer_dataset = deepcopy(training_datasets.cv)
      vae_swer_datastreams = deepcopy(training_datasets.datastreams)
      vae_swer_datastreams["audio_features"].available_for_inference = True
      forward_config = get_forward_config(
        returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
        forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
        embedding_size=256,
        speaker_embedding_size=256,
        calc_speaker_embedding=True,
        use_vae=True,
        use_audio_data=True,
        scale_kl_loss=True,
        skip_speaker_embeddings=("speaker_emb" in mode),
        kl_beta=loss,
      )
      gl_swer(
        name=exp_name + "/gl_swer",
        vocoder=default_vocoder,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        returnn_exe=returnn_exe,
        checkpoint=train_job.out_checkpoints[200],
        config=forward_config,
      )
      if "speaker_emb" in mode:
        speaker_embedding_hdf = None
      else:
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          datasets=training_datasets,
          prefix=exp_name,
          train_job=train_job,
        )
      vae_dataset_fw = deepcopy(training_datasets.cv)
      vae_dataset_fw.datasets["audio"]["segment_file"] = None
      vae_datastreams_fw = deepcopy(training_datasets.datastreams)
      vae_datastreams_fw["audio_features"].available_for_inference = True
      speaker_prior_hdf = build_vae_speaker_prior_dataset(
        returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
        dataset=vae_dataset_fw,
        datastreams=vae_datastreams_fw,
        prefix=exp_name,
        train_job=train_job,
        corpus=reference_corpus.corpus_file,
        skip_speaker_embeddings=("speaker_emb" in mode),
      )
      for synth_method in ["pred"]:
        synth_kwargs = {
          "use_vae": True,
          "use_calculated_prior": True,
          "skip_speaker_embeddings": ("speaker_emb" in mode),
        }
        synth_dataset = get_inference_dataset(
          corpus,
          returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
          returnn_exe=returnn_exe,
          datastreams=training_datasets.datastreams,
          speaker_embedding_hdf=speaker_embedding_hdf,
          durations=durations if "cheat_dur" in synth_method else None,
          process_corpus=False,
          speaker_prior_hdf=speaker_prior_hdf,
          speaker_prior_size=256 if "speaker_emb" in mode else 32,
        )
        synth_corpus = synthesize_with_splits(
          name=exp_name + f"/{synth_method}",
          reference_corpus=reference_corpus.corpus_file,
          corpus_name="train-clean-100",
          job_splits=job_splits,
          datasets=synth_dataset,
          returnn_root=returnn_root if not mode == "vae_no_speaker_emb" else returnn_root_local,
          returnn_exe=returnn_exe,
          returnn_common_root=returnn_common_root if not mode == "vae_no_speaker_emb" else returnn_common_root_local,
          checkpoint=train_job.out_checkpoints[200],
          vocoder=default_vocoder,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=False,
          use_true_durations=("cheat_dur" in synth_method),
          energy_cheat=("cheat_energy" in synth_method),
          pitch_cheat=("cheat_f0" in synth_method),
          **synth_kwargs,
        )
        synthetic_data_dict[f"ctc_{mode}_{loss}_{synth_method}"] = synth_corpus
  """ REMOVED, due to space
  exp_name = name + "/vae_no_speaker_emb"
  vae_dataset = deepcopy(training_datasets)
  vae_dataset.datastreams["audio_features"].available_for_inference = True
  train_config = get_training_config(
    returnn_common_root=returnn_common_root,
    training_datasets=vae_dataset,
    embedding_size=256,
    speaker_embedding_size=256,
    gauss_up=False,
    use_vae=True,
    batch_size=12000,
    skip_speaker_embeddings=True,
  )
  train_job = tts_training(
    config=train_config,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    prefix=exp_name,
    num_epochs=200,
  )
  vae_swer_dataset = deepcopy(training_datasets.cv)
  vae_swer_datastreams = deepcopy(training_datasets.datastreams)
  vae_swer_datastreams["audio_features"].available_for_inference = True
  forward_config = get_forward_config(
    returnn_common_root=returnn_common_root,
    forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
    embedding_size=256,
    speaker_embedding_size=256,
    calc_speaker_embedding=True,
    use_vae=True,
    use_audio_data=True,
    skip_speaker_embeddings=True,
  )
  gl_swer(
    name=exp_name,
    vocoder=default_vocoder,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    checkpoint=train_job.out_checkpoints[200],
    config=forward_config,
  )
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="ec4688ad6c712252b8b7a320a7a8bb73aba71543",
    checkout_folder_name="returnn_common",
  ).out_repository

  exp_name = name + "/vae_test_enc_out"
  vae_dataset = deepcopy(training_datasets)
  vae_dataset.datastreams["audio_features"].available_for_inference = True
  train_config = get_training_config(
    returnn_common_root=returnn_common_root,
    training_datasets=vae_dataset,
    embedding_size=256,
    speaker_embedding_size=256,
    gauss_up=False,
    use_vae=True,
    batch_size=12000,
    test_vae=True,
  )
  train_job = tts_training(
    config=train_config,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    prefix=exp_name,
    num_epochs=200,
  )
  vae_swer_dataset = deepcopy(training_datasets.cv)
  vae_swer_datastreams = deepcopy(training_datasets.datastreams)
  vae_swer_datastreams["audio_features"].available_for_inference = True
  forward_config = get_forward_config(
    returnn_common_root=returnn_common_root,
    forward_dataset=TTSForwardData(dataset=vae_swer_dataset, datastreams=vae_swer_datastreams),
    embedding_size=256,
    speaker_embedding_size=256,
    calc_speaker_embedding=True,
    use_vae=True,
    use_audio_data=True,
    test_vae=True,
  )
  returnn_root_job = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="63a26509fd55654a7271e8469180640e69fb84b3",  # fix dim tag declare same as
  )
  returnn_root = returnn_root_job.out_repository
  gl_swer(
    name=exp_name,
    vocoder=default_vocoder,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    checkpoint=train_job.out_checkpoints[200],
    config=forward_config,
  )
  # for variance in ["f0", "energy", "f0_energy"]:, REMOVED because of space
  for variance in []:
    exp_name = name + f"/{variance}_pred"
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
    var_training_datasets = deepcopy(training_datasets)
    if "f0" in variance:
      f0_hdf = get_ls_100_f0_hdf(
        durations=durations,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
      )
      var_training_datasets = extend_meta_datasets_with_f0(datasets=training_datasets, f0_dataset=f0_hdf)
    if "energy" in variance:
      energy_hdf = get_ls_100_energy_hdf(returnn_root=returnn_root, returnn_exe=returnn_exe, prefix=exp_name)
      var_training_datasets = extend_meta_datasets_with_energy(var_training_datasets, energy_dataset=energy_hdf)
    kwargs = {}
    if "f0" in variance:
      kwargs["use_pitch_pred"] = True
    if "energy" in variance:
      kwargs["use_energy_pred"] = True
    train_config = get_training_config(
      returnn_common_root=returnn_common_root,
      training_datasets=var_training_datasets,
      embedding_size=256,
      speaker_embedding_size=256,
      **kwargs,
    )
    train_job = tts_training(
      config=train_config,
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      prefix=exp_name,
      num_epochs=200,
    )
    forward_config = get_forward_config(
      returnn_common_root=returnn_common_root,
      forward_dataset=TTSForwardData(dataset=var_training_datasets.cv, datastreams=var_training_datasets.datastreams),
      embedding_size=256,
      speaker_embedding_size=256,
      calc_speaker_embedding=True,
      **kwargs,
    )
    gl_swer(
      name=exp_name,
      vocoder=default_vocoder,
      returnn_root=returnn_root,
      returnn_exe=returnn_exe,
      checkpoint=train_job.out_checkpoints[200],
      config=forward_config,
    )
  """
  return synthetic_data_dict


def ctc_loss_scale(return_trainings=False):
  """
    baseline for returnn_common ctc model with network constructor
    :return:
    """
  trainings={}
  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root_job = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="240f119b54d52a4324ab300c301f8e003e0a398c",
  )
  returnn_root_job.hash_overwrite = "ctc_baseline_returnn"
  returnn_root = returnn_root_job.out_repository
  returnn_common_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="79876b18552f61a3af7c21c670475fee51ef3991",
    checkout_folder_name="returnn_common",
  ).out_repository
  returnn_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
  ).out_repository
  returnn_common_root_local = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn_common",
    commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
    checkout_folder_name="returnn_common",
  ).out_repository
  name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/loss_scale"
  reference_corpus = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")["train-clean-100"]
  default_vocoder = get_default_vocoder(name=name)
  synthetic_data_dict = {}
  job_splits = 10
  alignments = get_loss_scale_alignments()

  for scale, alignment in alignments.items():
    name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/loss_scale" + f"_{scale}"
    training_datasets, corpus, durations = get_tts_data_from_ctc_align(
      "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/datasets/loss_scale" + f"_{scale}",
      returnn_exe=returnn_exe,
      returnn_root=returnn_root,
      alignment=alignment,
    )
    for upsampling in ["repeat", "gauss"]:
      #if upsampling == "gauss" and float(scale) not in [0, 0.25, 0.75, 1.0]: # Removed due to Space
      if upsampling == "repeat" or float(scale) not in [0, 1.0]:
        continue
      exp_name = name + f"_{upsampling}"

      if upsampling == "gauss":
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=training_datasets,
          embedding_size=256,
          speaker_embedding_size=256,
          gauss_up=True,
        )
        train_config.config["learning_rates"] = [0.0001, 0.001]
      else:
        train_config = get_training_config(
          returnn_common_root=returnn_common_root,
          training_datasets=training_datasets,
          embedding_size=256,
          speaker_embedding_size=256,
        )
      train_job = tts_training(
        config=train_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
        num_epochs=200,
      )
      if float(scale) == 0.0:
        trainings["ctc_0.0"] = train_job
        if return_trainings:
          return trainings
      if float(scale) in [0, 0.25, 1.0]:
        speaker_embedding_hdf = build_speaker_embedding_dataset(
          returnn_common_root=returnn_common_root,
          returnn_exe=returnn_exe,
          returnn_root=returnn_root,
          datasets=training_datasets,
          prefix=exp_name,
          train_job=train_job,
        )
        if upsampling == "gauss":
          synth_dataset = get_inference_dataset(
            corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=None,
            process_corpus=False,
          )
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
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
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
          )
          forward_hdf = forward_job.out_hdf_files["output.hdf"]
          tk.register_output(exp_name + "/dump_dur/durations.hdf", forward_hdf)
        # for dur_pred in ["pred", "cheat"]:, Removed due to Space
        for dur_pred in ["pred", "cheat"]:
          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root_local,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=256,
            speaker_embedding_size=256,
            calc_speaker_embedding=True,
            gauss_up=(upsampling == "gauss"),
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
          synth_dataset = get_inference_dataset_old(
            corpus,
            returnn_root=returnn_root,
            returnn_exe=returnn_exe,
            datastreams=training_datasets.datastreams,
            speaker_embedding_hdf=speaker_embedding_hdf,
            durations=durations if dur_pred == "cheat" else None,
            process_corpus=False,
          )

          synth_corpus = synthesize_with_splits(
            name=exp_name + f"/{dur_pred}",
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
          )
          synthetic_data_dict[f"ctc_{scale}_{upsampling}_{dur_pred}"] = synth_corpus
          calculate_feature_variance(
            train_job=train_job,
            corpus=corpus,
            returnn_root=returnn_root_local,
            returnn_exe=returnn_exe,
            returnn_common_root=returnn_common_root_local,
            prefix=exp_name + f"/{dur_pred}",
            training_datasets=training_datasets,
            gauss_up=(upsampling == "gauss"),
            embedding_size=256,
            speaker_embedding_size=256,
            use_true_durations=(dur_pred == "cheat"),
            durations=durations if dur_pred == "cheat" else None,
            original_durations=durations,
          )
      if float(scale) in [0]:
        for model_scale in [2]:
          exp_name = exp_name + "_big"
          returnn_root_loc = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="45fad83c785a45fa4abfeebfed2e731dd96f960c",
          ).out_repository
          returnn_common_root_loc = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="fcfaacf0e98e9630167a29b7fe306cb8d77bcbe6",
            checkout_folder_name="returnn_common",
          ).out_repository
          train_config = get_training_config(
            returnn_common_root=returnn_common_root_loc,
            training_datasets=training_datasets,
            embedding_size=int(256 * model_scale),
            speaker_embedding_size=int(256 * model_scale),
            gauss_up=(upsampling == "gauss"),
            enc_lstm_size=int(256 * model_scale),
            dec_lstm_size=int(1024 * model_scale),
            hidden_dim=int(256 * model_scale),
            variance_dim=int(512 * model_scale),
            batch_size=12000 if model_scale < 1.9 else 6000,
            )
          train_config.config["learning_rates"] = [0.0001, 0.001]

          train_job = tts_training(
            config=train_config,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root_loc,
            prefix=exp_name,
            num_epochs=200,
          )

          forward_config = get_forward_config(
            returnn_common_root=returnn_common_root_loc,
            forward_dataset=TTSForwardData(dataset=training_datasets.cv, datastreams=training_datasets.datastreams),
            embedding_size=int(256 * model_scale),
            speaker_embedding_size=int(256 * model_scale),
            enc_lstm_size=int(256 * model_scale),
            dec_lstm_size=int(1024 * model_scale),
            hidden_dim=int(256 * model_scale),
            variance_dim=int(512 * model_scale),
            calc_speaker_embedding=True,
            gauss_up=(upsampling == "gauss"),
          )

          gl_swer(
            name=exp_name + "/gl_swer",
            vocoder=default_vocoder,
            returnn_root=returnn_root_loc,
            returnn_exe=returnn_exe,
            checkpoint=train_job.out_checkpoints[200],
            config=forward_config,
          )

          speaker_embedding_hdf = build_speaker_embedding_dataset(
            returnn_common_root=returnn_common_root_loc,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root_loc,
            datasets=training_datasets,
            prefix=exp_name,
            train_job=train_job,
            speaker_embedding_size=int(256 * model_scale)
          )
          for dur_pred in ["pred", "cheat"]:
            synth_dataset = get_inference_dataset(
              corpus,
              returnn_root=returnn_root_loc,
              returnn_exe=returnn_exe,
              datastreams=training_datasets.datastreams,
              speaker_embedding_hdf=speaker_embedding_hdf,
              durations=durations if dur_pred == "cheat" else None,
              process_corpus=False,
              speaker_embedding_size=int(256 * model_scale)

            )

            synth_corpus = synthesize_with_splits(
              name=exp_name + f"/{dur_pred}",
              reference_corpus=reference_corpus.corpus_file,
              corpus_name="train-clean-100",
              job_splits=job_splits,
              datasets=synth_dataset,
              returnn_root=returnn_root_loc,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_loc,
              checkpoint=train_job.out_checkpoints[200],
              vocoder=default_vocoder,
              embedding_size=int(256 * model_scale),
              speaker_embedding_size=int(256 * model_scale),
              gauss_up=(upsampling == "gauss"),
              enc_lstm_size=int(256 * model_scale),
              dec_lstm_size=int(1024 * model_scale),
              hidden_dim=int(256 * model_scale),
              variance_dim=int(512 * model_scale),
              use_true_durations=(dur_pred == "cheat"),
            )
            synthetic_data_dict[f"ctc_{scale}_{model_scale}_{upsampling}_{dur_pred}"] = synth_corpus
            calculate_feature_variance(
              train_job=train_job,
              corpus=corpus,
              returnn_root=returnn_root_loc,
              returnn_exe=returnn_exe,
              returnn_common_root=returnn_common_root_loc,
              prefix=exp_name + f"/{dur_pred}",
              training_datasets=training_datasets,
              embedding_size=int(256 * model_scale),
              speaker_embedding_size=int(256 * model_scale),
              gauss_up=(upsampling == "gauss"),
              enc_lstm_size=int(256 * model_scale),
              dec_lstm_size=int(1024 * model_scale),
              hidden_dim=int(256 * model_scale),
              variance_dim=int(512 * model_scale),
              use_true_durations=(dur_pred == "cheat"),
              durations=durations if dur_pred == "cheat" else None,
              original_durations=durations,
            )
  return synthetic_data_dict


def synthesize_ls_100_features(silence_prep=True, add_speaker_tags=False, rasr_alignment=None, rasr_allophones=None):
  from i6_core.tools.git import CloneGitRepositoryJob
  from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import get_ls_100_features, \
    get_ls_100_real_feature_variance, get_ls_100_real_synth_feature_variance

  returnn_exe = tk.Path(
    "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
  )
  returnn_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
  ).out_repository
  name = "experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/real_features"
  if not silence_prep:
    name = name + "_no_silence_prep"
  if add_speaker_tags:
    name = name + "_added_tags"

  default_vocoder = get_default_vocoder(name=name)
  corpus = get_ls_100_features(
    vocoder=default_vocoder,
    returnn_root=returnn_root,
    returnn_exe=returnn_exe,
    prefix=name,
    silence_prep=silence_prep,
    add_speaker_tags=add_speaker_tags,
  )
  if rasr_alignment is not None and rasr_allophones is not None:
    get_ls_100_real_feature_variance(
      name=f"experiments/librispeech/nar_tts_2022/tts/tts_baseline_experiments/real_data_silence_prep_{silence_prep}",
      returnn_root=returnn_root,
      returnn_exe=returnn_exe,
      silence_prep=silence_prep,
      rasr_alignment=rasr_alignment,
      rasr_allophones=rasr_allophones)
    feature_corpus = get_ls_100_features(
      vocoder=default_vocoder,
      returnn_root=returnn_root,
      returnn_exe=returnn_exe,
      prefix=name + "_no_center",
      silence_prep=silence_prep,
      add_speaker_tags=add_speaker_tags,
      center=False,
    )
    get_ls_100_real_synth_feature_variance(
      name=name + "_no_center",
      returnn_root=returnn_root,
      returnn_exe=returnn_exe,
      rasr_alignment=rasr_alignment,
      rasr_allophones=rasr_allophones,
      corpus_bliss=corpus)

  return corpus
