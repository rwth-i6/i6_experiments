import copy
import numpy as np
import torch
import os
import json

import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Dim, batch_dim
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.frontend.decoder.transformer import TransformerDecoder

from sisyphus import tk

from i6_core.util import uopen

from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model, Wav2VecModel
from i6_experiments.users.mueller.experiments.ctc_baseline.decoding import recog_ffnn
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm
from i6_experiments.users.mueller.experiments.ctc_baseline.utils import hyps_ids_to_label, convert_to_output_hyps
from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad


def ctc_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, nbest_lengths: rf.Tensor = None, scores: rf.Tensor = None, seq_tags: rf.Tensor = None):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    nbest = config.int("ps_nbest", 1)
    decode_every_step = config.bool("decode_every_step", False)
    start_with_prior_gamma_steps = config.int("start_with_prior_gamma_steps", 0)
    version = config.int("version", 1)
    
    if isinstance(model, Wav2VecModel):
        w2v_opts = config.typed_value("w2v_opts", {})
        freeze_encoder_first_n_steps = w2v_opts.get("freeze_encoder_first_n_steps", 0)
        if config.typed_value("gradient_penalty_opts", {}) != {} or (freeze_encoder_first_n_steps > 0 and rf.get_run_ctx().step == w2v_opts.get("freeze_encoder_first_n_steps", 0)):
            model.set_wav2vec_encoder_trainable(True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    if start_with_prior_gamma_steps > 0 and rf.get_run_ctx().step < start_with_prior_gamma_steps:
        collected_outputs = {}
        logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
        log_probs = model.log_probs_wb_from_logits(logits)
        
        if config.typed_value("frame_prior", None) is not None:
            prior_file = config.typed_value("frame_prior")
        else:
            prior_file = config.typed_value("empirical_prior")
        assert prior_file
        batch_dims = data.remaining_dims(data_spatial_dim)
        
        ce_targets_spatial_dim = model.wb_target_dim.dimension * enc_spatial_dim
        
        ce_prior = np.loadtxt(prior_file, dtype="float32")
        ce_prior = torch.tensor(ce_prior, dtype=torch.float32, device=data.raw_tensor.device).exp()
        blank_prob = 0.2
        ce_prior[model.blank_idx] = 0.0
        ce_prior = (ce_prior / ce_prior.sum()) * (1 - blank_prob)
        ce_prior[model.blank_idx] = blank_prob
        assert ce_prior.shape[0] == model.wb_target_dim.dimension
        assert ce_prior.sum().allclose(torch.tensor(1.0, device=data.raw_tensor.device), atol=0.01)
        ce_prior = torch.cat([ce_prior] * enc_spatial_dim.get_dim_value(), dim=0)
        ce_prior = ce_prior.unsqueeze(0).expand(batch_dims[0].get_dim_value(), -1)
        ce_targets = rtf.TorchBackend.convert_to_tensor(ce_prior, dims=[batch_dim, ce_targets_spatial_dim], dtype="float32")
        
        ce_targets_indices = rf.range_over_dim(model.wb_target_dim)
        ce_targets_indices = rf.expand_dims(ce_targets_indices, dims=batch_dims + [enc_spatial_dim])
        ce_targets_indices = ce_targets_indices.copy_transpose([*batch_dims, enc_spatial_dim, model.wb_target_dim])
        
        ce_hyperparameters = {
            "aux_loss_layers": aux_loss_layers,
            "aux_loss_scales": aux_loss_scales,
            "use_normalized_loss": use_normalized_loss,
            "ps_nbest": 1,
            "grad_nbest": model.wb_target_dim.dimension,
            "decode_every_step": False,
            "version": version,
        }
        ce_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=ce_targets, targets_spatial_dim=ce_targets_spatial_dim, targets_indices=ce_targets_indices, hyperparameters=ce_hyperparameters)
    elif nbest == 0:
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        curr_step = rf.get_run_ctx().step
        lm_scale = hyperparameters.get("lm_weight")
        am_scale = 1.0
        prior_scale = hyperparameters.get("prior_weight")
        
        label_prior = False
        prior = np.loadtxt(config.typed_value("empirical_prior"), dtype="float32")
        if prior.shape[0] != model.wb_target_dim.dimension:
            assert prior.shape[0] == model.target_dim.dimension, f"prior shape {prior.shape[0]} != target shape {model.target_dim.dimension}"
            label_prior = True
        
        if "decay" in hyperparameters and hyperparameters["decay"] < 1.0:
            assert isinstance(curr_step, int)
            decay = hyperparameters.pop("decay")
            decay_limit = hyperparameters.pop("decay_limit", 0.0)
            start_weight = hyperparameters["lm_weight"]
            # lm_scale = 0.2 + (0.2 * decay ** curr_step)
            lm_scale = 0.3
            # am_scale = 1.0 - (0.9 * 0.99997 ** curr_step)
            am_scale = 0.1
            # prior_scale = 0.3 - (0.2 * 0.99999 ** curr_step)
            prior_scale = 0.0
            if curr_step % 100 == 0:
                print("LM weight:", lm_scale, "Prior weight:", prior_scale, "AM weight:", am_scale)
        
        fs_hyperparameters = {
            "train_lm_model": hyperparameters.get("lm_order"),
            "empirical_prior": config.typed_value("empirical_prior"),
            "am_scale": am_scale,
            "lm_scale": lm_scale,
            "prior_scale": prior_scale,
            "horizontal_prior": not label_prior,
            "blank_prior": not label_prior,
        }
        full_sum_train(model=model, data=data, data_spatial_dim=data_spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim, hyperparameters=fs_hyperparameters)
    else:
        collected_outputs = {}
        logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
        log_probs = model.log_probs_wb_from_logits(logits)
        
        # pg = SimplePrintGradients.apply
        # if rf.get_run_ctx().step % 10 == 0 or rf.get_run_ctx().step == start_with_prior_gamma_steps:
        #     log_probs.raw_tensor = pg(log_probs.raw_tensor, f"log_probs_nbest_24GB_{rf.get_run_ctx().step}", "plots", 0, f"Gradients FFNN LM 2 nbest {nbest} step {rf.get_run_ctx().step}", enc_spatial_dim.dyn_size_ext.raw_tensor[0].item())
        
        if config.float("prior_penalty_scale", 0.0) > 0.0:
            empirical_prior = config.typed_value("empirical_prior")
            assert empirical_prior is not None
            _prior_penalty(
                log_probs,
                enc_spatial_dim,
                empirical_prior,
            )
        
        norm_dim = targets_spatial_dim
        print_original_ctc = False
        original_targets = None
        original_log_probs = log_probs
        
        if seq_tags is not None:
            seq = "train-other-500/7492-105653-0055/7492-105653-0055"
            if seq in seq_tags.raw_tensor.tolist():
                print("FOUND SEQ")
        
        hyperparameters = config.typed_value("hyperparameters_decoder", None)
        hyperparameters = hyperparameters.copy() if hyperparameters is not None else None
        am_weight = None
        prior_am_normed = False
        
        if decode_every_step and rf.get_run_ctx().train_flag:
            assert seq_tags is not None
            save_seqs = scores is not None
            if seq in seq_tags.raw_tensor.tolist():
                idx = np.where(seq_tags == seq)[0]
                print("Found seq", seq, enc_spatial_dim.dyn_size_ext.raw_tensor[idx])
            
            prior_file = config.typed_value("empirical_prior")
            assert hyperparameters and prior_file
            hyperparameters["beam_size"] = 128
            if nbest > 1:
                hyperparameters["ps_nbest"] = nbest
            curr_step = rf.get_run_ctx().step
            curr_epoch = rf.get_run_ctx().epoch - 1
            if "decay" in hyperparameters and hyperparameters["decay"] < 1.0:
                assert isinstance(curr_step, int)
                decay = hyperparameters.pop("decay")
                decay_limit = hyperparameters.pop("decay_limit", 0.0)
                start_weight = hyperparameters["lm_weight"]
                # hyperparameters["lm_weight"] = 0.2 + (0.4 * decay ** curr_step)
                hyperparameters["lm_weight"] = 0.2 + (0.4 * 0.95 ** curr_epoch)
                # hyperparameters["lm_weight"] = 0.8
                # am_weight = 1.0 - (0.9 * 0.99997 ** curr_step)
                am_weight = 1.0 - (0.9 * 0.95 ** curr_epoch)
                # am_weight = 0.1
                # hyperparameters["prior_weight"] = 0.3 - (0.3 * 0.99999 ** curr_step)
                hyperparameters["prior_weight"] = 0.3 - (0.3 * 0.95 ** curr_epoch)
                # hyperparameters["prior_weight"] = 0.0
                if curr_step % 100 == 0:
                    print("LM weight:", hyperparameters["lm_weight"], "Prior weight:", hyperparameters["prior_weight"], "AM weight:", am_weight)
            
            if True:
                prior_am_normed = True
                if am_weight is not None:
                    log_probs = log_probs * am_weight
                if prior_file and hyperparameters["prior_weight"] > 0.0:
                    prior = np.loadtxt(prior_file, dtype="float32")
                    prior *= hyperparameters["prior_weight"]
                    prior = torch.tensor(prior, dtype=torch.float32, device=log_probs.device)
                    assert prior.shape[0] == log_probs.raw_tensor.shape[-1]
                    prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
                    log_probs = log_probs - prior
                log_probs = rf.log_softmax(log_probs, axis=model.wb_target_dim)
            else:
                if am_weight is not None:
                    log_probs = log_probs * am_weight
                    log_probs = rf.log_softmax(log_probs, axis=model.wb_target_dim)
            
            if save_seqs:
                device_id = torch.cuda.current_device()
                if device_id == 0:
                    hdf_writer = config.typed_value("train_hdf_writer")
                    assert isinstance(hdf_writer, SimpleHDFWriter)
            with torch.no_grad():
                batch_size = log_probs.raw_tensor.shape[0]
                batch_dims = data.remaining_dims(data_spatial_dim)
                hyps, new_scores = recog_ffnn(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file if not prior_am_normed else None, train_lm=True)
                assert len(hyps) == batch_size
                assert len(hyps[0]) == nbest
                hyps = [[convert_to_output_hyps(model, h, True) for h in hyps_batch] for hyps_batch in hyps]
                
                print("HYP:", hyps[0][0])
                print(hyps_ids_to_label(model, hyps[0][0]))
                if curr_step % 20 == 0:
                    print("REFERENCE:", targets.raw_tensor[0].tolist())
                    print(hyps_ids_to_label(model, targets.raw_tensor[0].tolist()))
                
                lengths = [[len(h) for h in hyps_batch] for hyps_batch in hyps]
                new_targets_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
                new_nbest_lengths = None
                if nbest > 1:
                    nbest_dim = Dim(nbest, name="nbest")
                    new_nbest_lengths = rf.convert_to_tensor(new_targets_spatial_dim, dims=(batch_dim, nbest_dim))
                    nbest_max = new_targets_spatial_dim.max(dim=1).values
                    hyps = [[h + [model.eos_idx] * (nbest_max[b] - len(h)) for h in hyps[b]] for b in range(batch_size)]
                    new_targets_spatial_dim = nbest_max * nbest
                else:
                    new_targets_spatial_dim = new_targets_spatial_dim.squeeze(dim=1)
                max_length = new_targets_spatial_dim.max().item()
                new_targets_spatial_dim = rf.convert_to_tensor(new_targets_spatial_dim, dims=(batch_dim,))
                new_targets_spatial_dim = Dim(new_targets_spatial_dim, name="out_spatial", dyn_size_ext=new_targets_spatial_dim)
                hyps = [sum(hyps_batch, []) for hyps_batch in hyps]
                hyps = [h + [model.eos_idx] * (max_length - len(h)) for h in hyps]
                hyps = torch.tensor(hyps, dtype=torch.int32, device=data.raw_tensor.device)
                new_targets = rf.convert_to_tensor(hyps, dims=(batch_dim, new_targets_spatial_dim), sparse_dim=model.target_dim)
                
                if save_seqs:
                    assert nbest == 1, "nbest > 1 not supported yet"
                    # Only keep new hyp if it has a better unsupervised metric score than the old one
                    keep_old = scores.raw_tensor[:, 0] == float("inf")
                    # TODO use blank prior to add length penalty
                    targets, targets_spatial_dim = _compare_targets(targets, targets_spatial_dim, new_targets, new_targets_spatial_dim, keep_old, model, hyperparameters, version=version)
                    
                    assert targets.raw_tensor.shape[0] == batch_size
                    
                    # Dump to HDF
                    device_id = torch.cuda.current_device()
                    if device_id == 0:
                        hdf_writer.insert_batch(
                            inputs=targets.raw_tensor.cpu().numpy(),
                            seq_len=targets_spatial_dim.dyn_size_ext.raw_tensor.tolist(),
                            seq_tag=seq_tags.raw_tensor.tolist(),
                        )
                    # hdf_scores_filename = f"scores-epoch-{(rf.get_run_ctx().epoch - 1) // partition_epoch + 1}.hdf"
                    # hdf_scores_dataset = SimpleHDFWriter(
                    #     filename=hdf_scores_filename,
                    #     dim=None,
                    #     ndim=1,
                    #     extend_existing_file=os.path.exists(hdf_scores_filename),
                    # )
                    # hdf_scores_dataset.insert_batch(
                    #     inputs=new_scores.raw_tensor.cpu().numpy(), # TODO update scores dependent on the selection
                    #     seq_len={0: [1]},
                    #     seq_tag=seq_tags.raw_tensor.tolist(),
                    # )
                    norm_dim = targets_spatial_dim
                else:
                    print_original_ctc = True
                    original_targets = (targets, targets_spatial_dim)
                    targets = new_targets
                    targets_spatial_dim = new_targets_spatial_dim
                    norm_dim = targets_spatial_dim + 1
                    nbest_lengths = new_nbest_lengths
                
            # targets_ls = []
            # lengths_ls = []
            # for i in range(scores.raw_tensor.shape[0]):
            #     old_score = scores.raw_tensor[i, :]
            #     new_score = new_scores.raw_tensor[i, :]
            #     if old_score[0] == float("inf") or old_score[0] >= new_score[0] or len(hyps[i]) == 0:
            #         targets_ls.append(targets.raw_tensor[i])
            #         lengths_ls.append(targets_spatial_dim.dyn_size_ext.raw_tensor[i].item())
            #     else:
            #         print("Using new hyp")
            #         print("Old score:", old_score[0], "New score:", new_score[0])
            #         print("Old hyp:", targets.raw_tensor[i].tolist())
            #         print("New hyp:", hyps[i])
            #         targets_ls.append(torch.tensor(hyps[i], dtype=torch.int32, device=data.raw_tensor.device))
            #         lengths_ls.append(len(hyps[i]))
            # max_len = max(lengths_ls)
            # targets_ls = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=model.eos_idx) for t in targets_ls]
            # targets = torch.stack(targets_ls, dim=0)
            # targets_spatial_dim = torch.tensor(lengths_ls, dtype=torch.int32, device=data.raw_tensor.device)
            # targets_spatial_dim = rf.convert_to_tensor(targets_spatial_dim, dims=(batch_dim,))
            # targets_spatial_dim = Dim(targets_spatial_dim, name="out_spatial", dyn_size_ext=targets_spatial_dim)
            # targets = rf.convert_to_tensor(targets, dims=(batch_dim, targets_spatial_dim), sparse_dim=model.target_dim)
        
        if nbest > 1 and rf.get_run_ctx().train_flag:
            assert nbest_lengths is not None
            from .sum_criterion import safe_logaddexp
            
            prior_file = config.typed_value("empirical_prior")
            norm_rescore = config.bool("norm_rescore", False)
            rescore_alignment_prior = config.bool("rescore_alignment_prior", False)
            arpa_file = config.typed_value("arpa_file", None)
            assert hyperparameters and prior_file
            assert not (norm_rescore and rescore_alignment_prior)
            assert not prior_am_normed or rescore_alignment_prior
            
            new_spatial_dim = targets_spatial_dim.div_left(nbest)
            new_spatial_dim_raw = new_spatial_dim.dyn_size_ext.raw_tensor
            targets_raw = targets.raw_tensor
            lengths_raw = nbest_lengths.raw_tensor
            
            # Split targets into nbest connsidering the nbest lengths
            tensor_ls = []
            sizes_ls = []
            for i in range(nbest):
                max_len = lengths_raw[:, i].max()
                # rf.pad_packed
                targets_i = []
                for b in range(targets_raw.shape[0]):
                    if lengths_raw[b][i] > 0:
                        s = new_spatial_dim_raw[b] * i
                        t_i = targets_raw[b][s:s+lengths_raw[b][i]]
                        t_i = torch.nn.functional.pad(t_i, (0, max_len - lengths_raw[b][i]), value=model.eos_idx)
                        targets_i.append(t_i)
                    else:
                        t_i = torch.full((max_len,), model.eos_idx, dtype=torch.int32, device=data.raw_tensor.device)
                        targets_i.append(t_i)
                targets_i = torch.stack(targets_i, dim=0)
                new_s = rf.convert_to_tensor(lengths_raw[:, i], dims=(batch_dim,))
                new_s = Dim(new_s, name=f"out_spatial_{i}", dyn_size_ext=new_s)
                targets_i = rf.convert_to_tensor(targets_i, dims=(batch_dim, new_s), sparse_dim=targets.sparse_dim)
                tensor_ls.append(targets_i)
                sizes_ls.append(new_s)
            
            if norm_rescore:
                with torch.no_grad():
                    lm_prior_scores_norm = _norm_rescore(tensor_ls, sizes_ls, model, hyperparameters, prior_file, arpa_file)
            
            loss_sum = None
            if aux_loss_layers:
                aux_probs = {}
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                        continue
                    aux_loss_sum = {}
                    linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                    aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                    aux_probs[i] = model.log_probs_wb_from_logits(aux_logits)
                    
            if rescore_alignment_prior and not prior_am_normed:
                prior_weight = hyperparameters.get("prior_weight", 0.0)
                if prior_file and prior_weight > 0.0:
                    prior = np.loadtxt(prior_file, dtype="float32")
                    prior *= prior_weight
                    prior = torch.tensor(prior, dtype=torch.float32, device=log_probs.raw_tensor.device)
                    assert prior.size(0) == log_probs.raw_tensor.size(2), "Prior size does not match!"
                    prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
                    log_probs = log_probs - prior
                    for i, layer_idx in enumerate(aux_loss_layers):
                        if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                            continue
                        aux_probs[i] = aux_probs[i] - prior
            
            for j in range(nbest):
                targets_s = tensor_ls[j]
                targets_spatial_dim_s = sizes_ls[j]
                
                if norm_rescore:
                    lm_prior_score = lm_prior_scores_norm[j]
                else:
                    with torch.no_grad():
                        lm_prior_score = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file if not rescore_alignment_prior else None, arpa_file=arpa_file).raw_tensor
                
                if config.bool("use_eos_postfix", False):
                    targets_s, (targets_spatial_dim_s,) = rf.pad(
                        targets_s, axes=[targets_spatial_dim_s], padding=[(0, 1)], value=model.eos_idx
                    )

                if aux_loss_layers:
                    for i, layer_idx in enumerate(aux_loss_layers):
                        if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                            continue
                        aux_loss = ctc_loss_fixed_grad(
                            logits=aux_probs[i],
                            logits_normalized=True,
                            targets=targets_s,
                            input_spatial_dim=enc_spatial_dim,
                            targets_spatial_dim=targets_spatial_dim_s,
                            blank_index=model.blank_idx,
                        )
                        if version != 3:
                            aux_loss_rescored = (-aux_loss).raw_tensor + lm_prior_score
                        else:
                            aux_loss_rescored = (-aux_loss).raw_tensor
                        if j > 0:
                            # Set loss to -inf if target length is 0
                            aux_loss_rescored = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), aux_loss_rescored)
                        if i in aux_loss_sum:
                            aux_loss_sum[i] = safe_logaddexp(aux_loss_sum[i], aux_loss_rescored)
                        else:
                            aux_loss_sum[i] = aux_loss_rescored
                        

                loss = ctc_loss_fixed_grad(
                    logits=log_probs,
                    logits_normalized=True,
                    targets=targets_s,
                    input_spatial_dim=enc_spatial_dim,
                    targets_spatial_dim=targets_spatial_dim_s,
                    blank_index=model.blank_idx,
                )
                if version != 3:
                    loss_rescored = (-loss).raw_tensor + lm_prior_score
                else:
                    loss_rescored = (-loss).raw_tensor
                if j > 0:
                    # Set loss to -inf if target length is 0
                    loss_rescored = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), loss_rescored)
                if loss_sum is not None:
                    loss_sum = safe_logaddexp(loss_sum, loss_rescored)
                else:
                    loss_sum = loss_rescored
                    
                if version == 3 or version == 4:
                    break
            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                        continue
                    aux_loss_sum_i = rtf.TorchBackend.convert_to_tensor(-aux_loss_sum[i], dims = [batch_dim], dtype = "float32", name=f"ctc_aux_loss_{layer_idx}")
                    aux_loss_sum_i.mark_as_loss(
                        f"ctc_{layer_idx}",
                        scale=aux_loss_scales[i],
                        custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                        use_normalized_loss=use_normalized_loss,
                    )
            loss_sum = rtf.TorchBackend.convert_to_tensor(-loss_sum, dims = [batch_dim], dtype = "float32", name=f"ctc_loss")
            loss_sum.mark_as_loss(
                "ctc",
                custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            
            if config.typed_value("gradient_penalty_opts", {}) != {}:
                _gradient_penalty(
                    loss_sum,
                    model
                )
            
            if print_original_ctc:
                _ctc_error(original_log_probs, original_targets[0], model, original_targets[1], enc_spatial_dim)
                wer_targets = original_targets[0].raw_tensor.tolist()
                wer_targets = [wer_targets[i][:original_targets[1].get_size_tensor().raw_tensor[i].item()] for i in range(len(wer_targets))]
                wer_inputs = tensor_ls[0].raw_tensor.tolist()
                wer_inputs = [wer_inputs[i][:sizes_ls[0].get_size_tensor().raw_tensor[i].item()] for i in range(len(wer_inputs))]
                _edit_distance(wer_inputs, wer_targets)
                _seq_len_error(original_log_probs, model, original_targets[1], enc_spatial_dim)
            else:
                _ctc_error(original_log_probs, tensor_ls[0], model, sizes_ls[0], enc_spatial_dim)
                _seq_len_error(original_log_probs, model, sizes_ls[0], enc_spatial_dim)
        else:
            if config.bool("use_eos_postfix", False):
                targets, (targets_spatial_dim,) = rf.pad(
                    targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
                )

            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                        continue
                    linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                    aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                    aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
                    aux_loss = ctc_loss_fixed_grad(
                        logits=aux_log_probs,
                        logits_normalized=True,
                        targets=targets,
                        input_spatial_dim=enc_spatial_dim,
                        targets_spatial_dim=targets_spatial_dim,
                        blank_index=model.blank_idx,
                    )
                    aux_loss.mark_as_loss(
                        f"ctc_{layer_idx}",
                        scale=aux_loss_scales[i],
                        custom_inv_norm_factor=norm_dim.get_size_tensor(),
                        use_normalized_loss=use_normalized_loss,
                    )

            loss = ctc_loss_fixed_grad(
                logits=log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            loss.mark_as_loss(
                "ctc",
                custom_inv_norm_factor=norm_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            
            if rf.get_run_ctx().train_flag and config.typed_value("gradient_penalty_opts", {}) != {}:
                _gradient_penalty(
                    loss,
                    model
                )
            
            _seq_len_error(log_probs, model, targets_spatial_dim, enc_spatial_dim)
            if print_original_ctc:
                _ctc_error(log_probs, original_targets[0], model, original_targets[1], enc_spatial_dim)

            assert not model.decoder
            if model.decoder:
                # potentially also other types but just assume
                # noinspection PyTypeChecker
                decoder: TransformerDecoder = model.decoder

                input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
                    targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
                )
                targets_w_eos, _ = rf.pad(
                    targets,
                    axes=[targets_spatial_dim],
                    padding=[(0, 1)],
                    value=model.eos_idx,
                    out_dims=[targets_w_eos_spatial_dim],
                )

                batch_dims = data.remaining_dims(data_spatial_dim)
                logits, _ = model.decoder(
                    input_labels,
                    spatial_dim=targets_w_eos_spatial_dim,
                    encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
                    state=model.decoder.default_initial_state(batch_dims=batch_dims),
                )

                logits_packed, pack_dim = rf.pack_padded(
                    logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
                )
                targets_packed, _ = rf.pack_padded(
                    targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
                )

                log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
                log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
                loss = rf.cross_entropy(
                    target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
                )
                loss.mark_as_loss("aed_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

                best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
                frame_error = best != targets_packed
                frame_error.mark_as_loss(name="aed_fer", as_error=True)

 
def full_sum_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, seq_tags: rf.Tensor = None, targets: rf.Tensor, targets_spatial_dim: Dim, hyperparameters: dict = None):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    from i6_experiments.users.mueller.experiments.ctc_baseline.sum_criterion import sum_loss_bigram, sum_loss_ngram, sum_loss_ffnn, safe_logsumexp, get_lm_logits, PrintGradients, NormGradients
    
    # torch.autograd.set_detect_anomaly(True)
    pg = PrintGradients.apply
    ng = NormGradients.apply

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    
    if hyperparameters is None:
        lm_name = config.typed_value("train_lm_model")
        
        am_scale = config.float("am_scale", 1.0)
        lm_scale = config.float("lm_scale", 1.0)
        prior_scale = config.float("prior_scale", 1.0)
        
        horizontal_prior = config.bool("horizontal_prior", True)
        blank_prior = config.bool("blank_prior", True)
        prior_gradient = config.bool("prior_gradient", True)
        empirical_prior = config.typed_value("empirical_prior", None)
        max_prior = config.bool("max_prior", False)
        top_k = config.int("top_k", 0)
        alignment_topk = config.bool("alignment_topk", True)
        blank_correction_version = config.int("blank_correction_version", 0)
        correction_in_final_score = config.bool("correction_in_final_score", False)
    else:
        lm_name = hyperparameters.get("train_lm_model")
        
        am_scale = hyperparameters.get("am_scale", 1.0)
        lm_scale = hyperparameters.get("lm_scale", 1.0)
        prior_scale = hyperparameters.get("prior_scale", 1.0)
        
        horizontal_prior = hyperparameters.get("horizontal_prior", True)
        blank_prior = hyperparameters.get("blank_prior", True)
        prior_gradient = hyperparameters.get("prior_gradient", True)
        empirical_prior = hyperparameters.get("empirical_prior", None)
        max_prior = hyperparameters.get("max_prior", False)
        top_k = hyperparameters.get("top_k", 0)
        alignment_topk = hyperparameters.get("alignment_topk", True)
        blank_correction_version = hyperparameters.get("blank_correction_version", 0)
        correction_in_final_score = hyperparameters.get("correction_in_final_score", False)
    
    use_prior = prior_scale > 0.0
    
    print_gradients = config.bool("print_gradients", False)
    version = config.int("version", 1)
    # if version == 4:
    #     am_scale = 1.0
    #     lm_scale = 1.0
    #     prior_scale = 0.0
    #     use_prior = prior_scale > 0.0
    #     blank_correction_version = 16
    #     correction_in_final_score = True
    #     top_k = 1
    #     print_gradients = True
    
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    use_ffnn_lm = lm_name.startswith("ffnn")
    if not use_ffnn_lm:
        with uopen(lm_name, "rb") as f:
            lm = torch.load(f, map_location=data.device)
            assert isinstance(lm, torch.Tensor), "Loaded LM is not a tensor"
        lm_order = lm.ndim
        # lm = torch.log_softmax(lm, dim=-1)
    else:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
        lm_order = int(lm_name[len("ffnn"):])
        assert lm.conv_filter_size_dim.dimension == lm_order
        
        if top_k == 0:
            with torch.no_grad():
                context_size = lm_order
                context_dim = rf.Dim(context_size, name="context")
                lm_out_dim = rf.Dim(context_size + 1, name="context+1")
                target = torch.arange(model.target_dim.dimension, device=data.raw_tensor.device)
                if context_size == 2:
                    target1 = target.unsqueeze(1).expand(model.target_dim.dimension, model.target_dim.dimension)
                    target2 = target.unsqueeze(0).expand(model.target_dim.dimension, model.target_dim.dimension)
                    target = torch.stack([target1, target2], dim=-1)
                    batch_dims = [Dim(model.target_dim.dimension, name="v1"), Dim(model.target_dim.dimension, name="v2")]
                elif context_size == 1:
                    target = target.unsqueeze(1)
                    batch_dims = [Dim(model.target_dim.dimension, name="v1")]
                else:
                    raise NotImplementedError(f"Full-sum on context size {context_size} not implemented")
                target = rf.convert_to_tensor(target, dims=batch_dims + [context_dim], sparse_dim=model.target_dim)
                lm_state = lm.default_initial_state(batch_dims=[])
                lm_logits, lm_state = get_lm_logits(batch_dims, target, lm, context_dim, lm_out_dim, lm_state)
                lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
                assert lm_logits.dims == (*batch_dims, model.target_dim)
                lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
                lm = lm_log_probs.raw_tensor
            use_ffnn_lm = False
            lm_order = lm.ndim

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_log_probs_raw = aux_log_probs.raw_tensor
            
            if use_prior:
                if empirical_prior is not None:
                    aux_log_prior = np.loadtxt(empirical_prior, dtype="float32")
                    aux_log_prior = torch.tensor(aux_log_prior, device=aux_log_probs.device)
                    if blank_prior:
                        assert aux_log_prior.size(0) == aux_log_probs_raw.size(2), f"Empirical prior size does not match (full_sum) ({aux_log_prior.size(0)} != {aux_log_probs_raw.size(2)})!"
                    else:
                        assert aux_log_prior.size(0) == aux_log_probs_raw.size(2) - 1, f"Empirical prior size does not match (full_sum) ({aux_log_prior.size(0)} != {aux_log_probs_raw.size(2)} - 1)!"
                        assert model.blank_idx == aux_log_prior.size(0)
                        aux_log_prior = torch.cat([aux_log_prior, torch.tensor([0.0], device=aux_log_probs_raw.device)], dim=0)
                else:
                    aux_log_prior = _model_log_prior(aux_log_probs_raw, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
                    if not prior_gradient:
                        aux_log_prior = aux_log_prior.detach()
                        
                    if not blank_prior:
                        aux_log_prior[model.blank_idx] = float("-inf")
                        aux_log_prior = torch.log_softmax(aux_log_prior, dim=-1)
                        aux_log_prior[model.blank_idx] = 0.0
            else:
                aux_log_prior = None
            if use_ffnn_lm:
                aux_log_probs.raw_tensor = aux_log_probs_raw
                aux_log_prior = rf.convert_to_tensor(aux_log_prior, dims=[model.wb_target_dim], dtype="float32")
                
                aux_loss = sum_loss_ffnn(
                    model=model,
                    log_probs=aux_log_probs,
                    lm=lm,
                    context_size=lm_order,
                    log_prior=aux_log_prior,
                    input_lengths=enc_spatial_dim,
                    top_k=top_k,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                    prior_scale=prior_scale,
                    horizontal_prior=horizontal_prior,
                    blank_prior=blank_prior,
                    device=aux_log_probs.device,
                    use_recombination = not alignment_topk,
                    recomb_blank = True,
                    recomb_after_topk = True,
                    recomb_with_sum = True,
                    blank_correction_version=blank_correction_version,
                )
            else:
                # (B, T, F) -> (T, B, F)
                aux_log_probs_raw = aux_log_probs_raw.permute(1, 0, 2)
                aux_loss = sum_loss_ngram(
                    log_probs=aux_log_probs_raw,
                    log_lm_probs=lm,
                    log_prior=aux_log_prior,
                    input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
                    top_k=top_k,
                    LM_order=lm_order,
                    am_scale=am_scale,
                    lm_scale=lm_scale,
                    prior_scale=prior_scale,
                    horizontal_prior=horizontal_prior,
                    blank_prior=blank_prior,
                    blank_idx=model.blank_idx,
                    eos_idx=model.eos_idx,
                    unk_idx=1,
                    device=aux_log_probs_raw.device,
                    alignment_topk=alignment_topk,
                    blank_correction_version=blank_correction_version,
                    correction_in_final_score = correction_in_final_score
                )
            aux_loss = rtf.TorchBackend.convert_to_tensor(aux_loss, dims = [batch_dim], dtype = "float32", name=f"aux_full_sum_{layer_idx}")
            aux_loss.mark_as_loss(
                f"aux_full_sum_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
    
    fixed_seqs = []
    # fixed_seqs = ["train-other-500/5756-305214-0041/5756-305214-0041", "train-clean-360/2498-134786-0003/2498-134786-0003", "train-other-500/1643-138089-0068/1643-138089-0068"] # MONICA DREW FRESH HOPE FROM HER SON'S WRITINGS THEY WERE FULL OF NOBLE THOUGHTS AND HIGH ASPIRATIONS, HERE IT IS
    print_for_idx = []
    
    # seq = seq_tags[0]
    # idx = np.where(seq_tags == seq)[0]
    # print_for_idx.append(idx[0])
    
    if seq_tags is not None:
        seq_tags = seq_tags.raw_tensor
        for seq in fixed_seqs:
            if seq in seq_tags:
                idx = np.where(seq_tags == seq)[0]
                print("Found seq", seq, enc_spatial_dim.dyn_size_ext.raw_tensor[idx])
                print_for_idx.append(idx[0])
            
    if print_gradients and print_for_idx:
        # logits_raw = logits.raw_tensor
        alias_name = config.typed_value("alias")
        # for idx_t in print_for_idx:
            # logits_raw = pg(logits_raw, "logits", alias_name, False, 1, False, idx_t)
            # logits_raw = pg(logits_raw, "logits", alias_name, False, None, False, idx_t, [8, 9, 10, 11], ["<blank>", "H", "<blank>", "ERE"])
            # logits_raw = pg(logits_raw, "logits", alias_name, False, None, True, idx_t, [], ["Logits"], enc_spatial_dim.dyn_size_ext.raw_tensor[idx_t])
        # logits.raw_tensor = logits_raw
        log_probs = model.log_probs_wb_from_logits(logits)
        log_probs_raw = log_probs.raw_tensor
        for idx_t in print_for_idx:
            print("Target:", targets.raw_tensor[idx_t].detach().cpu().numpy())
            # log_probs_raw = pg(log_probs_raw, "log_probs", alias_name, False, 1, False, idx_t)
            # log_probs_raw = pg(log_probs_raw, "log_probs", alias_name, False, None, False, idx_t, [8, 9, 10, 11], ["<blank>", "H", "<blank>", "ERE"])
            log_probs_raw = pg(log_probs_raw, "log_probs", alias_name, False, None, True, idx_t, [], ["Log Probs"], enc_spatial_dim.dyn_size_ext.raw_tensor[idx_t])
    else:
        log_probs = model.log_probs_wb_from_logits(logits)
        log_probs_raw = log_probs.raw_tensor
        
        # spg = SimplePrintGradients.apply
        # if rf.get_run_ctx().step % 10 == 0 or rf.get_run_ctx().step == 3:
        #     log_probs_raw = spg(log_probs_raw, f"log_probs_full_sum_{rf.get_run_ctx().step}", "plots", 0, f"Gradients FFNN LM 2 Full-sum step {rf.get_run_ctx().step}", enc_spatial_dim.dyn_size_ext.raw_tensor[0].item())
        
    if config.float("prior_penalty_scale", 0.0) > 0.0:
        assert empirical_prior is not None
        _prior_penalty(
            log_probs,
            enc_spatial_dim,
            empirical_prior,
        )

    if use_prior:
        if empirical_prior is not None:
            log_prior = np.loadtxt(empirical_prior, dtype="float32")
            log_prior = torch.tensor(log_prior, device=log_probs_raw.device)
            if blank_prior:
                assert log_prior.size(0) == log_probs_raw.size(2), f"Empirical prior size does not match (full_sum) ({log_prior.size(0)} != {log_probs_raw.size(2)})!"
            else:
                assert log_prior.size(0) == log_probs_raw.size(2) - 1, f"Empirical prior size does not match (full_sum) ({log_prior.size(0)} != {log_probs_raw.size(2)} - 1)!"
                assert model.blank_idx == log_prior.size(0)
                log_prior = torch.cat([log_prior, torch.tensor([0.0], device=log_probs_raw.device)], dim=0)
        else:
            log_prior = _model_log_prior(log_probs_raw, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
            if not prior_gradient:
                log_prior = log_prior.detach()
                
            if not blank_prior:
                log_prior[model.blank_idx] = float("-inf")
                log_prior = torch.log_softmax(log_prior, dim=-1)
                log_prior[model.blank_idx] = 0.0
    else:
        log_prior = None
        
    # log_probs_raw = ng(log_probs_raw) # TODO
    
    if use_ffnn_lm:
        log_probs.raw_tensor = log_probs_raw
        log_prior = rf.convert_to_tensor(log_prior, dims=[model.wb_target_dim], dtype="float32")
        
        loss = sum_loss_ffnn(
            model=model,
            log_probs=log_probs,
            lm=lm,
            context_size=lm_order,
            log_prior=log_prior,
            input_lengths=enc_spatial_dim,
            top_k=top_k,
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            horizontal_prior=horizontal_prior,
            blank_prior=blank_prior,
            device=log_probs.device,
            use_recombination = not alignment_topk,
            recomb_blank = True,
            recomb_after_topk = True,
            recomb_with_sum = True,
            blank_correction_version=blank_correction_version,
            print_best_path_for_idx=print_for_idx,
        )
    else:
        # (B, T, V) -> (T, B, V)
        log_probs_raw = log_probs_raw.permute(1, 0, 2)
        
        loss = sum_loss_ngram(
            log_probs=log_probs_raw,
            log_lm_probs=lm,
            log_prior=log_prior,
            input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
            top_k=top_k,
            LM_order=lm_order,
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            horizontal_prior=horizontal_prior,
            blank_prior=blank_prior,
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
            unk_idx=1,
            device=log_probs_raw.device,
            print_best_path_for_idx=print_for_idx,
            alignment_topk=alignment_topk,
            blank_correction_version=blank_correction_version,
            correction_in_final_score = correction_in_final_score
        )
    # if print_gradients and fixed_seqs[1] in seq_tags:
    #     print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9392214] 0.0009690238, 1: [0.01604532], 0.984082720
    loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"full_sum")
    loss.mark_as_loss(
        f"full_sum",
        custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )
    
    if rf.get_run_ctx().train_flag and config.typed_value("gradient_penalty_opts", {}) != {}:
        _gradient_penalty(
            loss,
            model
        )
    
    _ctc_error(log_probs, targets, model, targets_spatial_dim, enc_spatial_dim)
    _seq_len_error(log_probs, model, targets_spatial_dim, enc_spatial_dim)
    if rf.get_run_ctx().step % 10 == 0:
        _print_argmax(log_probs, model, targets, enc_spatial_dim)
    
    # if version == 5:
    #     loss = torch.ctc_loss( # ctc_loss_fixed_grad
    #         log_probs_raw,
    #         targets.raw_tensor,
    #         enc_spatial_dim.dyn_size_ext.raw_tensor,
    #         targets_spatial_dim.dyn_size_ext.raw_tensor,
    #         blank=model.blank_idx,
    #         reduction=0,
    #         zero_infinity=False
    #     )
    #     if print_gradients and fixed_seqs[1] in seq_tags:
    #         print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9210505], 0.0009867928 , 1: [0.00390251], 0.99610509
    #     loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"ctc")
    #     loss.mark_as_loss(
    #         "ctc",
    #         custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
    #         use_normalized_loss=use_normalized_loss,
    #     )
    # elif version in [10, 11]:
    #     if version == 10:
    #         loss = torch.logsumexp(log_probs_raw[0], dim=-1)
    #     else:
    #         loss = safe_logsumexp(log_probs_raw[0], -1)
    #     for t in range(1, log_probs_raw.size(0)):
    #         if version == -1:
    #             A = loss.unsqueeze(1)
    #             B = log_probs_raw[t].unsqueeze(1).expand(-1, log_probs_raw.size(-1), log_probs_raw.size(-1))
    #             new_loss = A.matmul(B).squeeze(1)
    #             time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs_raw.device)).unsqueeze(-1)
    #             loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
    #         else:
    #             # A = loss.unsqueeze(1).expand(-1, log_probs_raw.size(-1), log_probs_raw.size(-1))
    #             # B = log_probs_raw[t].unsqueeze(-1).expand(-1, log_probs_raw.size(-1), log_probs_raw.size(-1))
    #             # new_loss = safe_logsumexp(A + B, dim=-1)
    #             # new_loss = safe_logsumexp(torch.stack([loss, safe_logsumexp(log_probs_raw[t], -1)], dim=-1), dim=-1)
    #             if version == 10:
    #                 new_loss = loss + torch.logsumexp(log_probs_raw[t], -1)
    #             else:
    #                 new_loss = loss + safe_logsumexp(log_probs_raw[t], -1)
    #             time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs_raw.device))#.unsqueeze(-1)
    #             loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
    #     if version == -1:
    #         loss = loss.sum(-1)
    #     # else:
    #     #     loss = safe_logsumexp(loss, dim=-1)
    #     loss = -loss
    #     # print(loss[0].detach().cpu().numpy())
    #     loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"sum")
    #     loss.mark_as_loss(
    #         f"sum",
    #         custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
    #         use_normalized_loss=use_normalized_loss,
    #     )


def ce_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, targets_indices: rf.Tensor = None, hyperparameters: dict = None):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    if hyperparameters is None:
        config = get_global_config()  # noqa
        aux_loss_layers = config.typed_value("aux_loss_layers")
        aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
        use_normalized_loss = config.bool("use_normalized_loss", True)
        nbest = config.int("ps_nbest", 1)
        grad_nbest = config.int("grad_nbest", -1)
        decode_every_step = config.bool("decode_every_step", False)
        version = config.int("version", 1)
        use_eos_postfix = config.bool("use_eos_postfix", False)
    else:
        config = hyperparameters
        aux_loss_layers = config.get("aux_loss_layers")
        aux_loss_scales = config.get("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
        use_normalized_loss = config.get("use_normalized_loss", True)
        nbest = config.get("ps_nbest", 1)
        grad_nbest = config.get("grad_nbest", -1)
        decode_every_step = config.get("decode_every_step", False)
        version = config.get("version", 1)
        use_eos_postfix = config.get("use_eos_postfix", False)
    assert not decode_every_step
    assert nbest == 1

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    
    is_gradients = not torch.equal(enc_spatial_dim.dyn_size_ext.raw_tensor, targets_spatial_dim.dyn_size_ext.raw_tensor)
    assert (grad_nbest == -1 and not is_gradients) or (grad_nbest > 0 and is_gradients)
    if is_gradients:
        assert targets_indices is not None
        targets_indices = rf.set_sparse_dim(targets_indices, model.wb_target_dim)
        targets.sparse_dim = None
        # new_spatial_dim = targets_spatial_dim.div_left(grad_nbest)
        new_spatial_dim = targets_indices.dims[1]
        # nbest_dim = Dim(grad_nbest, name="nbest")
        nbest_dim = targets_indices.dims[2]
        targets = rf.split_dims(targets, axis=targets_spatial_dim, dims=[new_spatial_dim, nbest_dim])
        targets_spatial_dim = new_spatial_dim
        assert torch.equal(enc_spatial_dim.dyn_size_ext.raw_tensor, targets_spatial_dim.dyn_size_ext.raw_tensor)
    else:
        targets = rf.set_sparse_dim(targets, model.wb_target_dim)
    
    if targets_spatial_dim not in logits.dims:
        logits = rf.replace_dim_v2(logits, in_dim=enc_spatial_dim, out_dim=targets_spatial_dim, allow_expand=False, allow_shrink=False)
    enc_spatial_dim = targets_spatial_dim
    
    # batch_dims = data.remaining_dims(data_spatial_dim)
    # logits, pack_dim = rf.pack_padded(
    #     logits, dims=batch_dims + [enc_spatial_dim], enforce_sorted=False
    # )
    log_probs = model.log_probs_wb_from_logits(logits)
    
    if not is_gradients and use_eos_postfix:
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            if is_gradients:
                aux_log_probs_nbest = rf.gather(aux_log_probs, indices=targets_indices, axis=model.wb_target_dim)
                aux_loss = rf.cross_entropy(
                    target=targets, estimated=aux_log_probs_nbest, estimated_type="log-probs", axis=nbest_dim
                )
            else:
                aux_loss = rf.cross_entropy(
                    target=targets, estimated=aux_log_probs, estimated_type="log-probs", axis=model.wb_target_dim
                )
            aux_loss.mark_as_loss(f"ce_{layer_idx}", scale=aux_loss_scales[i], use_normalized_loss=use_normalized_loss)
            
    # targets, _ = rf.pack_padded(
    #     targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    # )

    if is_gradients:
        log_probs_nbest = rf.gather(log_probs, indices=targets_indices, axis=model.wb_target_dim)
        loss = rf.cross_entropy(
            target=targets, estimated=log_probs_nbest, estimated_type="log-probs", axis=nbest_dim
        )
    else:
        loss = rf.cross_entropy(
            target=targets, estimated=log_probs, estimated_type="log-probs", axis=model.wb_target_dim
        )
    loss.mark_as_loss("ce", use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(log_probs, axis=model.wb_target_dim)
    if is_gradients:
        targets_max = rf.reduce_argmax(targets, axis=nbest_dim)
        targets_max = rf.gather(targets_indices, indices=targets_max, axis=nbest_dim)
    else:
        targets_max = targets
    frame_error = best != targets_max
    frame_error.mark_as_loss(name="fer", as_error=True)


def seq_gamma_ctc_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim, nbest_lengths: rf.Tensor = None, scores: rf.Tensor = None, seq_tags: rf.Tensor = None):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    nbest = config.int("ps_nbest", 1)
    version = config.int("version", 1)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    log_probs = model.log_probs_wb_from_logits(logits)
        
    if nbest > 1 and rf.get_run_ctx().train_flag:
        assert nbest_lengths is not None
        from .sum_criterion import safe_logsumexp
        
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        prior_file = config.typed_value("empirical_prior")
        rescore_alignment_prior = config.bool("rescore_alignment_prior", False)
        gamma_scaling = config.float("gamma_scaling", 1.0)
        assert hyperparameters and prior_file
        
        new_spatial_dim = targets_spatial_dim.div_left(nbest)
        new_spatial_dim_raw = new_spatial_dim.dyn_size_ext.raw_tensor
        targets_raw = targets.raw_tensor
        lengths_raw = nbest_lengths.raw_tensor
        
        # Split targets into nbest connsidering the nbest lengths
        tensor_ls = []
        sizes_ls = []
        for i in range(nbest):
            max_len = lengths_raw[:, i].max()
            # rf.pad_packed
            targets_i = []
            for b in range(targets_raw.shape[0]):
                if lengths_raw[b][i] > 0:
                    s = new_spatial_dim_raw[b] * i
                    t_i = targets_raw[b][s:s+lengths_raw[b][i]]
                    t_i = torch.nn.functional.pad(t_i, (0, max_len - lengths_raw[b][i]), value=model.eos_idx)
                    targets_i.append(t_i)
                else:
                    t_i = torch.full((max_len,), model.eos_idx, dtype=torch.int32, device=data.raw_tensor.device)
                    targets_i.append(t_i)
            targets_i = torch.stack(targets_i, dim=0)
            new_s = rf.convert_to_tensor(lengths_raw[:, i], dims=(batch_dim,))
            new_s = Dim(new_s, name=f"out_spatial_{i}", dyn_size_ext=new_s)
            targets_i = rf.convert_to_tensor(targets_i, dims=(batch_dim, new_s), sparse_dim=targets.sparse_dim)
            tensor_ls.append(targets_i)
            sizes_ls.append(new_s)
        
        nbest_scores = []
        if aux_loss_layers:
            aux_probs = {}
            aux_nbest_scores = {}
            for i, layer_idx in enumerate(aux_loss_layers):
                if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                    continue
                aux_nbest_scores[i] = []
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                aux_probs[i] = model.log_probs_wb_from_logits(aux_logits)
                
        if rescore_alignment_prior:
            prior_weight = hyperparameters.get("prior_weight", 0.0)
            if prior_file and prior_weight > 0.0:
                prior = np.loadtxt(prior_file, dtype="float32")
                prior *= prior_weight
                prior = torch.tensor(prior, dtype=torch.float32, device=log_probs.raw_tensor.device)
                assert prior.size(0) == log_probs.raw_tensor.size(2), "Prior size does not match!"
                prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
                log_probs = log_probs - prior
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                        continue
                    aux_probs[i] = aux_probs[i] - prior
        
        for j in range(nbest):
            targets_s = tensor_ls[j]
            targets_spatial_dim_s = sizes_ls[j]
            
            with torch.no_grad():
                lm_prior_score = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file if not rescore_alignment_prior else None).raw_tensor

            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                        continue
                    aux_loss = ctc_loss_fixed_grad(
                        logits=aux_probs[i],
                        logits_normalized=True,
                        targets=targets_s,
                        input_spatial_dim=enc_spatial_dim,
                        targets_spatial_dim=targets_spatial_dim_s,
                        blank_index=model.blank_idx,
                    )
                    # if version != 3:
                    aux_loss_rescored = (-aux_loss).raw_tensor + lm_prior_score
                    # else:
                    #     aux_loss_rescored = (-aux_loss).raw_tensor
                    if j > 0:
                        # Set loss to -inf if target length is 0
                        aux_loss_rescored = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), aux_loss_rescored)
                    aux_nbest_scores[i].append(aux_loss_rescored)
                    
            loss = ctc_loss_fixed_grad(
                logits=log_probs,
                logits_normalized=True,
                targets=targets_s,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim_s,
                blank_index=model.blank_idx,
            )
            # if version != 3:
            loss_rescored = (-loss).raw_tensor + lm_prior_score
            # else:
            #     loss_rescored = (-loss).raw_tensor
            if j > 0:
                # Set loss to -inf if target length is 0
                loss_rescored = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), loss_rescored)
            
            nbest_scores.append(loss_rescored)
                
            # if version == 3 or version == 4:
            #     break
        
        nbest_dim = Dim(nbest, name="nbest")
        norm_dim = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, data.device)
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
                if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                    continue
                aux_nbest_scores_i = torch.stack(aux_nbest_scores[i], dim=-1)
                with torch.no_grad():
                    aux_scores_scaled = aux_nbest_scores_i.detach() * gamma_scaling
                    aux_nbest_scores_i_sum = safe_logsumexp(aux_scores_scaled, dim=-1, keepdim=True)
                    assert not (aux_nbest_scores_i_sum == float("-inf")).any(), f"-inf in {aux_nbest_scores_i_sum.tolist()}\noriginal: {aux_nbest_scores_i.tolist()}"
                    aux_nbest_scores_i_norm_log = aux_scores_scaled - aux_nbest_scores_i_sum
                    aux_nbest_scores_i_norm = aux_nbest_scores_i_norm_log.exp()
                    assert aux_nbest_scores_i_norm.sum(dim=-1).allclose(torch.tensor(1.0, device=aux_nbest_scores_i.device), rtol = 0.001), f"Prior probs do not sum to 1.0, but to {aux_nbest_scores_i_norm.sum(dim=-1)}"
                    aux_targets = rtf.TorchBackend.convert_to_tensor(aux_nbest_scores_i_norm, dims = [batch_dim, nbest_dim], dtype = "float32", name=f"seq_gammas_{layer_idx}")
                
                aux_nbest_scores_i = torch.where(aux_nbest_scores_i == float("-inf"), 0.0, aux_nbest_scores_i)
                aux_nbest_scores_i = rtf.TorchBackend.convert_to_tensor(aux_nbest_scores_i, dims = [batch_dim, nbest_dim], dtype = "float32", name=f"am_lm_pr_scores_{layer_idx}")
                aux_loss = rf.cross_entropy(
                    target=aux_targets, estimated=aux_nbest_scores_i, estimated_type="log-probs", axis=nbest_dim
                )
                if use_normalized_loss:
                    aux_loss = aux_loss / norm_dim
                aux_loss.mark_as_loss(f"seq_ce_{layer_idx}", use_normalized_loss=False)
                
        nbest_scores = torch.stack(nbest_scores, dim=-1)
        with torch.no_grad():
            scores_scaled = nbest_scores.detach() * gamma_scaling
            nbest_scores_exp_sum = safe_logsumexp(scores_scaled, dim=-1, keepdim=True)
            assert not (nbest_scores_exp_sum == float("-inf")).any(), f"-inf in {nbest_scores_exp_sum.tolist()}\noriginal: {nbest_scores.tolist()}"
            nbest_scores_norm_log = scores_scaled - nbest_scores_exp_sum
            nbest_scores_norm = nbest_scores_norm_log.exp()
            assert nbest_scores_norm.sum(dim=-1).allclose(torch.tensor(1.0, device=nbest_scores.device), rtol = 0.001), f"Prior probs do not sum to 1.0, but to {nbest_scores_norm.sum(dim=-1)}"
            targets = rtf.TorchBackend.convert_to_tensor(nbest_scores_norm, dims = [batch_dim, nbest_dim], dtype = "float32", name=f"seq_gammas")
        
        nbest_scores = torch.where(nbest_scores == float("-inf"), 0.0, nbest_scores)
        nbest_scores = rtf.TorchBackend.convert_to_tensor(nbest_scores, dims = [batch_dim, nbest_dim], dtype = "float32", name=f"am_lm_pr_scores")
        loss = rf.cross_entropy(
            target=targets, estimated=nbest_scores, estimated_type="log-probs", axis=nbest_dim
        )
        if use_normalized_loss:
            loss = loss / norm_dim
        loss.mark_as_loss("seq_ce", use_normalized_loss=False)
        
        _ctc_error(log_probs, tensor_ls[0], model, sizes_ls[0], enc_spatial_dim)
        _seq_len_error(log_probs, model, sizes_ls[0], enc_spatial_dim)
    
    else:
        norm_dim = targets_spatial_dim
        if config.bool("use_eos_postfix", False):
            targets, (targets_spatial_dim,) = rf.pad(
                targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
            )

        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
                if layer_idx > len(model.encoder.layers) or str(layer_idx - 1) not in collected_outputs:
                    continue
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
                aux_loss = ctc_loss_fixed_grad(
                    logits=aux_log_probs,
                    logits_normalized=True,
                    targets=targets,
                    input_spatial_dim=enc_spatial_dim,
                    targets_spatial_dim=targets_spatial_dim,
                    blank_index=model.blank_idx,
                )
                aux_loss.mark_as_loss(
                    f"ctc_{layer_idx}",
                    scale=aux_loss_scales[i],
                    custom_inv_norm_factor=norm_dim.get_size_tensor(),
                    use_normalized_loss=use_normalized_loss,
                )

        loss = ctc_loss_fixed_grad(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        loss.mark_as_loss(
            "ctc",
            custom_inv_norm_factor=norm_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
        
        _seq_len_error(log_probs, model, targets_spatial_dim, enc_spatial_dim)

# Helper functions ------------------------------------------------------

def _vocab_usage_score(
    targets: rf.Tensor,
    model: Model,
) -> torch.Tensor:
    vocab = model.target_dim.get_size_tensor().raw_tensor
    vocab = torch.arange(vocab, device=targets.raw_tensor.device)
    vocab = vocab[(vocab != model.bos_idx) & (vocab != model.eos_idx) & (vocab != 1)] # NOTE assume that 1 is the unk_idx
    vocab_len = vocab.size(0)
    targets_pt = targets.raw_tensor
    targets_ls = targets_pt.unbind(0)
    targets_ls = [t[(t != model.bos_idx) & (t != model.eos_idx) & (t != 1)] for t in targets_ls]
    targets_vocab_len = [torch.tensor(t.size(0), dtype=torch.int32, device=targets_pt.device) for t in targets_ls]
    targets_vocab_len = torch.stack(targets_vocab_len, dim=0)
    return targets_vocab_len / vocab_len

def _LM_score(
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    model: Model,
    lm_name: str,
    train_lm: bool = True,
    arpa_file: tk.Path = None,
) -> rf.Tensor:
    assert lm_name
    if lm_name.startswith("ffnn"):
        if train_lm:
            assert model.train_language_model
            assert model.train_language_model.vocab_dim == model.target_dim
            lm: FeedForwardLm = model.train_language_model
        else:
            assert model.recog_language_model
            assert model.recog_language_model.vocab_dim == model.target_dim
            lm: FeedForwardLm = model.recog_language_model
        lm_order = int(lm_name[len("ffnn"):])
        assert lm.conv_filter_size_dim.dimension == lm_order
        
        targets_w_eos, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
        
        batch_dims = targets.remaining_dims(targets_spatial_dim)
        lm_state = lm.default_initial_state(batch_dims=batch_dims)
        lm_logits, lm_state = lm(
            targets,
            spatial_dim=targets_spatial_dim,
            out_spatial_dim=targets_w_eos_spatial_dim,
            state=lm_state,
        )  # Flat_Batch_Beam, Vocab / ...
        # logits_packed, pack_dim = rf.pack_padded(
        #     lm_logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        # )
        assert lm_logits.dims == (*batch_dims, targets_w_eos_spatial_dim, model.target_dim)
        lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
        lm_log_probs = rf.gather(lm_log_probs, axis=model.target_dim, indices=targets_w_eos)
        lm_log_probs = rf.reduce_sum(lm_log_probs, axis=targets_w_eos_spatial_dim)
    else:
        assert lm_name == "word4gram"
        assert arpa_file is not None, "ARPA file must be provided for word4gram LM"
        import kenlm
        assert targets.raw_tensor.ndim in [2, 3]
        
        batch_dims = targets.remaining_dims(targets_spatial_dim)
        dev = targets.raw_tensor.device
        lm = kenlm.Model(arpa_file)
        targets = targets.raw_tensor
        lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
        lm_log_probs = []
        if targets.ndim == 2:
            for i in range(targets.size(0)):
                t = targets[i, :lengths[i]].tolist()
                word_target = hyps_ids_to_label(model, t)
                lm_log_probs.append(lm.score(word_target, bos=True, eos=True))
        else:
            for i in range(targets.size(0)):
                lm_log_probs_i = []
                for j in range(targets.size(1)):
                    t = targets[i, j, :lengths[i, j]].tolist()
                    word_target = hyps_ids_to_label(model, t)
                    lm_log_probs_i.append(lm.score(word_target, bos=True, eos=True))
                lm_log_probs.append(lm_log_probs_i)
        lm_log_probs = torch.tensor(lm_log_probs, dtype=torch.float32, device=dev)
        lm_log_probs = rf.convert_to_tensor(lm_log_probs, dims=batch_dims, dtype="float32", name="lm_log_probs")
    
    return lm_log_probs

def _prior_score(
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    model: Model,
    prior_file: tk.Path,
    force_label_prior: bool,
) -> rf.Tensor:
    assert prior_file
    prior = np.loadtxt(prior_file, dtype="float32")
    prior = torch.tensor(prior, dtype=torch.float32, device=targets.raw_tensor.device)
    label_prior = True
    if prior.size(0) == int(model.wb_target_dim.get_dim_value()):
        assert model.blank_idx == prior.size(0) - 1
        if force_label_prior:
            prior = prior[:-1]
            prior = torch.log_softmax(prior, dim=-1)
            prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.target_dim], dtype="float32")
        else:
            label_prior = False
            prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
    else:
        assert prior.size(0) == int(model.target_dim.get_dim_value()), f"Prior size does not match! {prior.size(0)} vs {int(model.target_dim.get_dim_value())}"
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.target_dim], dtype="float32")
    batch_dims = targets.remaining_dims(targets_spatial_dim)
    prior = rf.expand_dims(prior, [*batch_dims, targets_spatial_dim])
    if label_prior:
        prior_log_probs = rf.gather(prior, axis=model.target_dim, indices=targets)
    else:
        prior_log_probs = rf.gather(prior, axis=model.wb_target_dim, indices=targets)
    prior_log_probs = rf.reduce_sum(prior_log_probs, axis=targets_spatial_dim)
    
    return prior_log_probs

def _compare_targets(
    target_A: rf.Tensor,
    target_A_dim: Dim,
    target_B: rf.Tensor,
    target_B_dim: Dim,
    keep_A: torch.Tensor,
    model: Model,
    hyperparameters: dict,
    version: int = 1
) -> tuple[rf.Tensor, Dim]:
    """Compare quality of two targets based on unsupervised metric from https://arxiv.org/pdf/2105.11084
    """
    assert target_A.dims[0] == target_B.dims[0]
    lm_A = _LM_score(target_A, target_A_dim, model, hyperparameters.get("lm_order", None)).raw_tensor
    lm_B = _LM_score(target_B, target_B_dim, model, hyperparameters.get("lm_order", None)).raw_tensor
    lm_A_norm = -(lm_A / (target_A_dim.get_size_tensor().raw_tensor + 1).to(lm_A.device))
    lm_B_norm = -(lm_B / (target_B_dim.get_size_tensor().raw_tensor + 1).to(lm_A.device))
    u_A = _vocab_usage_score(target_A, model)
    u_B = _vocab_usage_score(target_B, model)
    assert u_A.ndim == lm_A.ndim == lm_B.ndim == u_B.ndim == 1
    assert u_A.size(0) == lm_A.size(0) == lm_B.size(0) == u_B.size(0)
    score_A = lm_A_norm - torch.log(u_A)
    score_B = lm_B_norm - torch.log(u_B)
    
    margin = torch.log(torch.tensor(1.2, dtype=torch.float32, device=lm_A.device))
    
    targets_ls = []
    lengths_ls = []
    for i in range(u_A.size(0)):
        if keep_A[i] == True:
            targets_ls.append(target_A.raw_tensor[i])
            assert target_A_dim.dyn_size_ext.raw_tensor[i].item() > 0
            lengths_ls.append(target_A_dim.dyn_size_ext.raw_tensor[i].item())
            continue
        
        score_A_i = score_A[i]
        score_B_i = score_B[i]
        lm_A_i = lm_A[i]
        lm_B_i = lm_B[i]
        A_better = None
        # Vocab-usage adjusted ppl of A is better
        if score_A_i <= score_B_i:
            if version >= 2:
                A_better = True
            else:
                # Also better with a margin
                if score_A_i + margin <= score_B_i:
                    A_better = True
                # B within margin and has better LM score
                elif lm_B_i > lm_A_i:
                    A_better = False
                # Otherwise
                else:
                    A_better = True
        else:
            if version >= 2:
                A_better = False
            else:
                if score_B_i + margin <= score_A_i:
                    A_better = False
                elif lm_A_i > lm_B_i:
                    A_better = True
                else:
                    A_better = False
        assert A_better is not None
        if A_better:
            targets_ls.append(target_A.raw_tensor[i])
            # assert target_A_dim.dyn_size_ext.raw_tensor[i].item() > 0
            lengths_ls.append(target_A_dim.dyn_size_ext.raw_tensor[i].item())
        else:
            # print("Using new hyp")
            # print("Old hyp:", target_A.raw_tensor[i].tolist())
            # print("New hyp:", target_B.raw_tensor[i].tolist())
            # print("Scores:", score_A, score_B)
            targets_ls.append(target_B.raw_tensor[i])
            assert target_B_dim.dyn_size_ext.raw_tensor[i].item() > 0
            lengths_ls.append(target_B_dim.dyn_size_ext.raw_tensor[i].item())
    max_len = max(lengths_ls)
    targets_ls = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=model.eos_idx) for t in targets_ls]
    targets = torch.stack(targets_ls, dim=0)
    targets_spatial_dim = torch.tensor(lengths_ls, dtype=torch.int32, device=targets.device)
    targets_spatial_dim = rf.convert_to_tensor(targets_spatial_dim, dims=(batch_dim,))
    targets_spatial_dim = Dim(targets_spatial_dim, name="out_spatial", dyn_size_ext=targets_spatial_dim)
    targets = rf.convert_to_tensor(targets, dims=(batch_dim, targets_spatial_dim), sparse_dim=model.target_dim)
    
    return targets, targets_spatial_dim

def _rescore(
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    model: Model,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    train_lm: bool = True,
    arpa_file: tk.Path = None,
) -> rf.Tensor | tuple[rf.Tensor, rf.Tensor]:
    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    
    # Calculate labelwise prior score if available
    prior_log_probs = None
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    if prior_file and prior_weight > 0.0:
        prior_log_probs = _prior_score(
            targets,
            targets_spatial_dim,
            model,
            prior_file,
            force_label_prior=True,
        )
    
    # Calculate LM score
    lm_log_probs = _LM_score(
        targets,
        targets_spatial_dim,
        model,
        lm_name,
        train_lm=train_lm,
        arpa_file=arpa_file,
    )
        
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]
    lm_log_probs *= lm_scale
    if prior_file and prior_weight > 0.0:
        prior_log_probs *= prior_weight
        lm_log_probs -= prior_log_probs
    return lm_log_probs

def _norm_rescore(
    targets_ls: list[rf.Tensor],
    targets_spatial_dim_ls: list[Dim],
    model: Model,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    arpa_file: tk.Path = None,
) -> list[torch.Tensor]:
    n = len(targets_ls)
    lm_prior_scores = []
    for j in range(n):
        targets_s = targets_ls[j]
        targets_spatial_dim_s = targets_spatial_dim_ls[j]
        
        ret = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file, arpa_file=arpa_file).raw_tensor
        if j > 0:
            ret = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), ret)
        lm_prior_scores.append(ret)
        
    # renormalize
    lm_prior_scores = torch.stack(lm_prior_scores, dim=0)
    lm_prior_scores = torch.log_softmax(lm_prior_scores, dim = 0)
    
    return torch.unbind(lm_prior_scores, dim=0)

def _model_log_prior(log_probs: torch.Tensor, lengths: torch.Tensor, use_max: bool = False, separate_eos: bool = False) -> torch.Tensor:
    from i6_experiments.users.mueller.experiments.ctc_baseline.sum_criterion import safe_logsumexp
    # assumes log_probs as (B, T, V)
    lengths = lengths.to(log_probs.device)
    assert lengths.size(0) == log_probs.size(0), "Prior calculation batch lengths are not the same (full_sum)!"
    
    # Length mask
    mask_bool = torch.arange(log_probs.size(1), device=log_probs.device).expand(log_probs.size(0), -1) < lengths.unsqueeze(1)
    mask = torch.where(mask_bool, 0.0, float("-inf"))
    mask = mask.unsqueeze(-1).expand(-1, -1, log_probs.size(2))
    log_probs = log_probs + mask
    
    sum_frames = lengths.sum()
    if use_max:
        if separate_eos:
            raise NotImplementedError("Separate EOS not implemented for max prior")
        else:
            argmaxs = log_probs.argmax(dim=2)
            argmaxs = argmaxs.flatten()
            argmaxs = argmaxs[mask_bool.flatten()]
            assert argmaxs.size(0) == sum_frames, f"Prior calculation frame count does not match (max) ({argmaxs.size(0)} != {sum_frames})"
            sum_probs = argmaxs.bincount(minlength=log_probs.size(2))
            sum_frames += (sum_probs == 0).sum()
            sum_probs = torch.where(sum_probs == 0, 1, sum_probs)
            log_sum_probs = sum_probs.log()
    else:
        if separate_eos:
            log_sum_probs = torch.full((log_probs.size(2) + 1,), float("-inf"), device=log_probs.device)
            log_sum_probs[1:-1] = safe_logsumexp(safe_logsumexp(log_probs[:,:,1:], dim=0), dim=0) # Sum over batch and time
            log_sum_probs[0] = safe_logsumexp(log_probs[:,0,0], dim=0) # BOS prob
            log_sum_probs[-1] = safe_logsumexp(safe_logsumexp(log_probs[:,1:,0], dim=0), dim=0) # EOS prob
        else:
            log_sum_probs = safe_logsumexp(safe_logsumexp(log_probs, dim=0), dim=0)
        
    log_mean_probs = log_sum_probs - sum_frames.log()
    
    with torch.no_grad():
        assert log_mean_probs.exp().sum().allclose(torch.tensor(1.0, device=log_mean_probs.device)), f"Prior probs do not sum to 1.0, but to {log_mean_probs.exp().sum()}"
        if log_mean_probs.isclose(torch.tensor([0.0], device=log_probs.device)).any() or log_mean_probs.isinf().any() or log_mean_probs.isnan().any():
            print("Prior probs contain inf or nan or 0 values!", log_mean_probs, log_mean_probs.exp())
    
    return log_mean_probs

def _prior_penalty(log_probs: rf.Tensor, enc_spatial_dim: Dim, empirical_prior_path: tk.Path) -> rf.Tensor:
    from i6_experiments.users.mueller.experiments.ctc_baseline.sum_criterion import safe_logaddexp, safe_logsumexp
    from returnn.config import get_global_config
    config = get_global_config()
    prior_penalty_scale = config.float("prior_penalty_scale", 1.0)
    
    model_prior = _model_log_prior(
        log_probs.raw_tensor,
        enc_spatial_dim.get_size_tensor().raw_tensor,
    )
    empirical_prior = np.loadtxt(empirical_prior_path, dtype="float32")
    empirical_prior = torch.tensor(empirical_prior, dtype=torch.float32, device=log_probs.device)
    assert empirical_prior.shape[0] == log_probs.raw_tensor.shape[-1]
    
    # prior_penalty = safe_logaddexp(empirical_prior, -model_prior) * 2
    # prior_penalty = safe_logsumexp(prior_penalty, dim=0)
    
    prior_penalty = torch.nn.functional.mse_loss(model_prior, empirical_prior)
    prior_penalty = prior_penalty.unsqueeze(0)
    
    prior_penalty = rf.convert_to_tensor(
        prior_penalty,
        dims=[Dim(1, name="prior_penalty")],
        dtype="float32",
        name="prior_penalty",
    )
    prior_penalty.mark_as_loss(
        "prior_penalty",
        scale=prior_penalty_scale
    )
    
def _gradient_penalty(loss: rf.Tensor, model: Model):
    from returnn.config import get_global_config
    config = get_global_config()
    opts = config.typed_value("gradient_penalty_opts")
    target_gradient_log_l2_norm = opts.get("target_gradient_log_l2_norm", -1.0)
    norm = opts.get("norm", "l2")
    assert norm in ["l2", "l1"]
    penalty_pow = opts.get("penalty_pow", 2)
    assert penalty_pow in [1, 2]
    gradient_penalty_scale = opts.get("gradient_penalty_scale", 1.0)
    
    loss_raw = loss.raw_tensor
    # loss_sum_raw = torch.sum(loss_raw)

    print(model._current_extracted_features)
    model._current_extracted_features.retain_grad()
    loss_raw.retain_grad()
    

    # loss_sum_raw.backward(retain_graph=True)
    loss_raw.backward(torch.ones_like(loss_raw, device=loss_raw.device), retain_graph=True)
    feature_gradients_raw = model._current_extracted_features.grad
    normed_feature_gradients_raw = torch.linalg.vector_norm(feature_gradients_raw, dim=-1, ord = 2 if norm == "l2" else 1)
    log_mean_normed_feature_gradients_raw = torch.log(torch.mean(normed_feature_gradients_raw))
    log_mean_normed_diff = log_mean_normed_feature_gradients_raw - target_gradient_log_l2_norm
        
    if penalty_pow == 2:
        feature_gradients_penalty = torch.pow(log_mean_normed_diff, 2)
    else:
        feature_gradients_penalty = torch.abs(log_mean_normed_diff)

    rf.get_run_ctx().mark_as_loss(
        feature_gradients_penalty,
        name="feature_gradients_penalty",
        scale=gradient_penalty_scale
    )
    
def _seq_len_error(log_probs, model, targets_spatial_dim, enc_spatial_dim):
    with torch.no_grad():
        argmax = rf.reduce_argmax(log_probs, axis=model.wb_target_dim)
        argmax = argmax.copy_transpose([batch_dim, enc_spatial_dim])
        argmax = argmax.raw_tensor.detach().cpu().numpy()
        enc_spatial_sizes = enc_spatial_dim.dyn_size_ext.raw_tensor.numpy()
        num_argmax_non_blank = 0
        for batch_idx, seq in enumerate(argmax):
            seq_ = list(seq)
            seq_collapsed = [t1 for i, (t1, t2) in enumerate(zip(seq_, [None] + seq_)) if t1 != t2 and i < enc_spatial_sizes[batch_idx] and t1 != model.blank_idx]
            num_argmax_non_blank += len(seq_collapsed)

        num_target_non_blank = targets_spatial_dim.dyn_size_ext.raw_tensor.sum()
        if num_target_non_blank > 0:
            fraction = num_argmax_non_blank / num_target_non_blank
        else:
            fraction = float(num_argmax_non_blank)

        greedy_non_blank_to_ground_truth_fraction = rf.convert_to_tensor(
            fraction,
        )
        greedy_non_blank_to_ground_truth_fraction.mark_as_loss("greedy_non_blank_to_ground_truth_fraction_error", as_error=True)
    
def _ctc_error(log_probs, targets, model, targets_spatial_dim, enc_spatial_dim):
    with torch.no_grad():
        loss = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        loss.mark_as_loss(
            "ctc_error",
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            as_error=True
        )
        
def _edit_distance(inputs, targets):
    from torchaudio.functional import edit_distance
    with torch.no_grad():
        loss = [edit_distance(inputs[i], targets[i]) / len(targets[i]) for i in range(len(inputs))]
        loss = torch.Tensor(loss)
        loss = rf.convert_to_tensor(loss, dims=[batch_dim], dtype = "float32", name="ed")
        loss.mark_as_loss(
            "ed",
            as_error=True
        )
        
def _print_argmax(log_probs, model, target, enc_spatial_dim):
    with torch.no_grad():
        argmax = rf.reduce_argmax(log_probs, axis=model.wb_target_dim)
        argmax = argmax.copy_transpose([batch_dim, enc_spatial_dim])
        argmax = argmax.raw_tensor.detach().tolist()
        output_argmax = [convert_to_output_hyps(model, h) for h in argmax]
        print("HYP:", output_argmax[0])
        print("REF:", target.raw_tensor[0].tolist())
    
class SimplePrintGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name, prefix, batch_idx: int, title: str, length: int):
        ctx.name = name
        ctx.prefix = prefix
        assert isinstance(batch_idx, int)
        ctx.batch_idx = [batch_idx]
        ctx.title = title
        ctx.length = length
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        prefix = ctx.prefix
        x, = ctx.saved_tensors
        prefix += "/"
        prefix = "/u/marten.mueller/dev/ctc_baseline/output/" + prefix
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        gradients = grad_output[ctx.batch_idx, :ctx.length].detach().squeeze(0).cpu().numpy()
        
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from datetime import datetime
        
        fig, ax = plt.subplots(figsize=(15, 15))
        fig.supylabel("Vocab")
        fig.supxlabel("Timestep")
        
        ax.imshow(gradients.T, origin="lower", cmap=cm.gray)
        ax.set_yticks(np.arange(0, 185, 10))
        ax.set_title("Gradients " + ctx.title)
        # ax.text(2, -20, f'black: 1.0, white: 0.0', bbox={'facecolor': 'white', 'pad': 10})
        
        now = datetime.now()
        fig.savefig(prefix + name + now.strftime("_%H:%M:%S_%d-%m") + ".png")
        
        return grad_output, None, None, None, None, None