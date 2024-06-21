from sisyphus import tk
import os

from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_100_ctc.fairseq_finetuning.ctc_standalone.experiments.ctc_phon.baseline import eow_phon_ls100_ctc_base
from i6_experiments.users.vieting.experiments.librispeech.\
    librispeech_960_pretraining.wav2vec2.config_02_fairseq_phoneme import \
        get_fairseq_root, \
        run_fairseq_pretraining_negatives_other_target, \
        run_fairseq_pretraining_phoneme_boundary_masking, \
        run_fairseq_pretraining_phoneme_negatives_other_target_boundary_masking

# Pretraining
neg_other_trg_pretrain_job = run_fairseq_pretraining_negatives_other_target()
phon_boundary_pretrain_job = run_fairseq_pretraining_phoneme_boundary_masking()
neg_other_trg_phon_boundary_pretrain_job = run_fairseq_pretraining_phoneme_negatives_other_target_boundary_masking()

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
    model_conf_w2v["w2v_path"] = neg_other_trg_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_neg_sampling_other_target", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )

    # phoneme boundary masking
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = phon_boundary_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join("w2v_phoneme_boundary_masking", f"checkpoint_{checkpoint}"),
        fairseq_root=fairseq_root,
    )

    # negative sampling + phoneme boundary masking
    model_conf_w2v = base_model_conf.copy()
    model_conf_w2v["w2v_path"] = neg_other_trg_phon_boundary_pretrain_job.out_models[checkpoint].model
    eow_phon_ls100_ctc_base(
        model_conf_w2v=model_conf_w2v,
        train_name_suffix=os.path.join(
            "w2v_neg_sampling_other_target_phoneme_boundary_masking",
            f"checkpoint_{checkpoint}"
            ),
        fairseq_root=fairseq_root,
    )
