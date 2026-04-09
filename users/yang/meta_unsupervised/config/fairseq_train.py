import copy

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import SetupFairseqJob
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob

from sisyphus import tk

# fairseq_root = tk.Path(
#     "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq")
# python_env = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
# setup_job = SetupFairseqJob(fairseq_root=fairseq_root, python_env=python_env)
# tk.register_output("fairseq/setup", setup_job.out_fairseq_root)

environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
#task_data = tk.Path(
    #"/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing/Wav2VecUFeaturizeAudioJob.ip1QrE3GuqCQ/output/audio_features/precompute_pca512_cls128_mean_pooled")
task_data = tk.Path("/work/asr4/zyang/unsupervised_meta/data/precompute_pca512_cls128_mean_pooled")
task_text_data = tk.Path(
    "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/PrepareWav2VecTextDataJob.RZfllsI3R2Pd/output/text/phones")
fairseq_root = tk.Path(
    "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/SetupFairseqJob.1QrCsdXOGqsu/output/fairseq")
prefix = "wav2vec_u_lbs_gan"
#config_dir = "/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/wav2vec_data_utils/SetupFairseqJob.1QrCsdXOGqsu/output/fairseq/examples/wav2vec/unsupervised/config/gan"
config_dir = "/work/asr4/zyang/unsupervised_meta/meta_config/gan"
config_name = "w2vu"
extra_configs = {'model.code_penalty': 2, 'model.gradient_penalty': 1.5, 'model.smoothness_weight': 0.75,
                 'common.seed': 4}
user_dir = tk.Path("/work/asr4/zyang/unsupervised_meta/fairseq_unsupervised/fairseq/examples/wav2vec/unsupervised")
task_kenlm_path = None


# try different seeds to see the convergence:
for seed in range(1,10):
    extra_config = copy.deepcopy(extra_configs)
    extra_config['common.seed'] = seed
    fairseq_train_job = FairseqHydraTrainWav2VecUJob(
        environment=environment,
        task_data=task_data,
        task_text_data=task_text_data,
        fairseq_root=fairseq_root,
        prefix=prefix,
        config_dir=config_dir,
        config_name=config_name,
        extra_configs=extra_config,
        user_dir=user_dir,
        task_kenlm_path=task_kenlm_path
    )

    tk.register_output(f"fairseq/train/model_seed-{seed}", fairseq_train_job.out_best_model)


# task_data = tk.Path("/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/output/fairseq/merged_data")
# extra_configs["model.input_dim"] = 768
# fairseq_train_job = FairseqHydraTrainWav2VecUJob(
#     environment=environment,
#     task_data=task_data,
#     task_text_data=task_text_data,
#     fairseq_root=fairseq_root,
#     prefix=prefix,
#     config_dir=config_dir,
#     config_name=config_name,
#     extra_configs=extra_configs,
#     task_kenlm_path=task_kenlm_path
# )
# tk.register_output("fairseq/train/model", fairseq_train_job.out_best_model)
#
# extra_configs["model.input_dim"] = 768
# extra_configs["+model.generator_batch_norm"] = 30
# fairseq_train_job = FairseqHydraTrainWav2VecUJob(
#     environment=environment,
#     task_data=task_data,
#     task_text_data=task_text_data,
#     fairseq_root=fairseq_root,
#     prefix=prefix,
#     config_dir=config_dir,
#     config_name=config_name,
#     extra_configs=extra_configs,
#     task_kenlm_path=task_kenlm_path
# )
# tk.register_output("fairseq/train/model", fairseq_train_job.out_best_model)
#
# extra_configs["model.input_dim"] = 768
# extra_configs["model.generator_stride"] = 2
# del extra_configs["+model.generator_batch_norm"]
# fairseq_train_job = FairseqHydraTrainWav2VecUJob(
#     environment=environment,
#     task_data=task_data,
#     task_text_data=task_text_data,
#     fairseq_root=fairseq_root,
#     prefix=prefix,
#     config_dir=config_dir,
#     config_name=config_name,
#     extra_configs=extra_configs,
#     task_kenlm_path=task_kenlm_path
# )
# tk.register_output("fairseq/train/model", fairseq_train_job.out_best_model)
#
#
# extra_configs["model.input_dim"] = 768
# extra_configs["+model.generator_batch_norm"] = 30
# extra_configs["model.generator_stride"] = 2
# fairseq_train_job = FairseqHydraTrainWav2VecUJob(
#     environment=environment,
#     task_data=task_data,
#     task_text_data=task_text_data,
#     fairseq_root=fairseq_root,
#     prefix=prefix,
#     config_dir=config_dir,
#     config_name=config_name,
#     extra_configs=extra_configs,
#     task_kenlm_path=task_kenlm_path
# )
# tk.register_output("fairseq/train/model", fairseq_train_job.out_best_model)