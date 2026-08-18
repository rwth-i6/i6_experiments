import os

files = [
    "config_librispeech_960_wo_sil_denoise_pretrain_unfreeze_v4.py",
    "config_librispeech_960_wo_sil_MLM_pretrain_unfreeze_v4.py",
    "config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v4.py",
    "config_librispeech_960_w_sil_MLM_pretrain_unfreeze_v4.py",
]

target_block = """        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            rasr_recog_opts={"line_based_lexicon_file": train_data.add_opts["line_based_lexicon_file"]},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )"""

replacement_block = """        pretrain_steps = train_args.get("denoise_pretrain_steps", train_args.get("mlm_pretrain_steps", 0))
        pretrain_epochs = int(pretrain_steps / 5580)
        checkpoints = [pretrain_epochs, pretrain_epochs + 10, pretrain_epochs + 100]
        
        keep_epochs = get_keep_epochs(base_num_epochs)
        if keep_epochs is None:
            keep_epochs = []
        keep_epochs.extend(checkpoints)
        
        vis_epochs = [250, 500, 750, 1000] + checkpoints

        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_epochs,
            skip_eval=False,
            rasr_recog_opts={"line_based_lexicon_file": train_data.add_opts["line_based_lexicon_file"]},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            vis_epochs=vis_epochs,
        )"""

for f in files:
    if os.path.exists(f):
        with open(f, 'r') as file:
            content = file.read()
        if target_block in content:
            content = content.replace(target_block, replacement_block)
            with open(f, 'w') as file:
                file.write(content)
            print(f"Replaced in {f}")
        else:
            print(f"Target block not found in {f}")
    else:
        print(f"File not found: {f}")
