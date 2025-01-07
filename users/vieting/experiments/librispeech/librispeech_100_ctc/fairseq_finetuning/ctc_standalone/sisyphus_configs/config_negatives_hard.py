from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining


# pretraining
other_target_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_negatives_other_target_v1",
    commit="1397363c5c0e3c4e3ab620be562730399c852493",
    python_exe_hash_overwrite="itc_python_launcher_py310_torch",
    negative_sampling_strategy="other_target",
)


neg_hard_pretrain_job = run_fairseq_pretraining(
        exp_name="monophone_negatives_hard_v1",
        commit="be51394d876428ad531e0786d80de43d6a8818af",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        negative_sampling_strategy="hard_negatives",
    )

neg_hard_pretrain_jobs = dict()
neg_hard_pretrain_jobs[0] = neg_hard_pretrain_job
for start_cp in [50, 100, 150, 200]:
    neg_hard_pretrain_jobs[start_cp] = run_fairseq_pretraining(
        exp_name=f"monophone_negatives_hard_after_{start_cp}ep_other_v1",
        commit="be51394d876428ad531e0786d80de43d6a8818af",
        python_exe_hash_overwrite="itc_python_launcher_py310_torch",
        checkpoint=other_target_pretrain_job.out_models[start_cp].model,
        negative_sampling_strategy="hard_negatives",
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

for start_cp in [50, 100, 150, 200]:
    for additional_cp in range(50, 600+1-start_cp, 50):
        model_conf_w2v = base_model_conf.copy()
        model_conf_w2v["w2v_path"] = neg_hard_pretrain_jobs[start_cp].out_models[start_cp + additional_cp].model
        eow_phon_ls100_ctc_base(
            model_conf_w2v=model_conf_w2v,
            train_name_suffix=os.path.join("w2v_negatives_hard", f"other_{start_cp}_hard_{additional_cp}"),
            fairseq_root=fairseq_root,
        )
