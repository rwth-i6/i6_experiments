from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining \

# Pretraining
phon_boundary_pretrain_job = run_fairseq_pretraining(
    exp_name="monophone_boundary_masking_v1",
    #commit="3a67ce092a569e85449a8e911936e87c8948b6d7",
    commit="87dec4ffcba2fd71e8838ca099a09816cddeff5b",
    python_exe_hash_overwrite="itc_python_launcher_py310_torch",
    mask_strategy="phonemes",
    mask_length=1,
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
    # phoneme boundary masking
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_phoneme_boundary_masking", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )

# finetuning experiments only for the last checkpoint
final_cp = 600
# random vs phoneme mask in finetuning
model_conf_w2v = base_model_conf.copy()  # base model, no need to set `mask_strategy` and `mask_length`
model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_phoneme_boundary_masking",
        "phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)
model_conf_w2v = base_model_conf.copy()
model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
model_conf_w2v["mask_strategy"] = "random"
model_conf_w2v["mask_length"] = 10
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_phoneme_boundary_masking",
        "random_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)

# phoneme mask lengths in finetuning
model_conf_w2v = base_model_conf.copy()  # base model, no need to set `mask_strategy` and `mask_length`
model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_phoneme_boundary_masking",
        "1_phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)
model_conf_w2v = base_model_conf.copy()
model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
model_conf_w2v["mask_strategy"] = "phonemes"
model_conf_w2v["mask_length"] = 2
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_phoneme_boundary_masking",
        "2_phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)

model_conf_w2v = base_model_conf.copy()
model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
model_conf_w2v["mask_strategy"] = "phonemes"
model_conf_w2v["mask_other"] = 1
model_conf_w2v["mask_selection"] = "uniform"
eow_phon_ls100_ctc_base(
    model_conf_w2v=model_conf_w2v,
    train_name_suffix=os.path.join(
        "w2v_phoneme_boundary_masking",
        "1_2_phoneme_spec",
        f"checkpoint_{final_cp}"
        ),
    fairseq_root=fairseq_root,
)

# mask probability in finetuning
for mask_prob in [0.35, 0.5, 0.65, 0.8]:
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[final_cp].model
    model_conf_w2v["mask_prob"] = mask_prob
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join(
            "w2v_phoneme_boundary_masking",
            f"{str(mask_prob).replace('.', '_')}_phoneme_mask_prob",  # replace '.' with '_'
            f"checkpoint_{final_cp}"
            ),
        fairseq_root=fairseq_root,
    )
