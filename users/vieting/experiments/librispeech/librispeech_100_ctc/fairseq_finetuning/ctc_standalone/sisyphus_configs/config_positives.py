from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining

# pretraining
# positive sampling
pos_sampling_5_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_positive_sampling_5_v2",
    commit="24d7d72c1e00f69689dc8a8ba2e0d75fe5f1cccd",
    num_positives=5,
)

pos_sampling_10_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_positive_sampling_10_v2",
    commit="24d7d72c1e00f69689dc8a8ba2e0d75fe5f1cccd",
    num_positives=10,
)

pos_sampling_15_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_positive_sampling_15_v2",
    commit="24d7d72c1e00f69689dc8a8ba2e0d75fe5f1cccd",
    num_positives=15,
)


# fairseq root
fairseq_root = get_fairseq_root(fairseq_exe=tk.Path("/usr/bin/python3"))

# Finetuning
base_model_conf = {
    "_name": "wav2vec_ctc",
    "apply_mask": True,
    "mask_prob": 0.65,
    "mask_channel_prob": 0.5,
    "mask_channel_length": 64,
    "layerdrop": 0.1,
    "activation_dropout": 0.1,
    "feature_grad_mult": 0.0,
    "freeze_finetune_updates": 10000,  # was 0 in fairseq config
}

#checkpoint = 400
for checkpoint in [100, 200, 300, 400, 500, 600]:
    # positive sampling 5
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = pos_sampling_5_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_positive_sampling", "pos_samples_5", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )
    # positive sampling 10
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = pos_sampling_10_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_positive_sampling", "pos_samples_10", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )
    # positive sampling 15
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = pos_sampling_15_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_positive_sampling", "pos_samples_15", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )