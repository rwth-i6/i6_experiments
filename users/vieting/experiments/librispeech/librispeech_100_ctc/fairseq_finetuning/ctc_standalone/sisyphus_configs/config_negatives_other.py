from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining


# pretraining
neg_other_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_negatives_other_target_v1",
    commit="1397363c5c0e3c4e3ab620be562730399c852493",
    python_exe_hash_overwrite="itc_python_launcher_py310_torch",
    negative_sampling_strategy="other_target",
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

checkpoints = [100, 200, 300, 400, 500, 600]
for checkpoint in checkpoints:
    # negative sampling
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_neg_sampling_other_target", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )


# finetuning experiments only for the last checkpoint
final_cp = 600
# random vs phoneme mask in finetuning
model_conf_w2v = base_model_conf.copy()  # base model, no need to set `mask_strategy` and `mask_length`
model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[final_cp].model
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_neg_sampling_other_target",
        "random_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)
model_conf_w2v = base_model_conf.copy()
model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[final_cp].model
model_conf_w2v["mask_strategy"] = "phonemes"
model_conf_w2v["mask_length"] = 1
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_neg_sampling_other_target",
        "phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)

# phoneme mask lengths in finetuning
for mask_len in [1, 2]:
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[final_cp].model
    model_conf_w2v["mask_strategy"] = "phonemes"
    model_conf_w2v["mask_length"] = mask_len
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join(
            "w2v_neg_sampling_other_target",
            f"{mask_len}_phoneme_spec",
            f"checkpoint_{final_cp}"
            ),
        fairseq_root=fairseq_root,
    )

model_conf_w2v = base_model_conf.copy()
model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[final_cp].model
model_conf_w2v["mask_strategy"] = "phonemes"
model_conf_w2v["mask_length"] = 1
model_conf_w2v["mask_selection"] = "uniform"
model_conf_w2v["mask_other"] = 1
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_neg_sampling_other_target",
        "1_2_phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)

# mask probability in finetuning
for mask_prob in [0.35, 0.5, 0.65, 0.8]:
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_pretrain_job.out_models[final_cp].model
    model_conf_w2v["mask_strategy"] = "phonemes"
    model_conf_w2v["mask_prob"] = mask_prob
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join(
            "w2v_neg_sampling_other_target",
            f"{str(mask_prob).replace('.', '_')}_phoneme_mask_prob",  # replace "." with "_" for the folder name
            f"checkpoint_{final_cp}"
            ),
        fairseq_root=fairseq_root,
    )
