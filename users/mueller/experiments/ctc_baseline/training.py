import copy
import numpy as np
import torch

import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Dim, batch_dim
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.frontend.decoder.transformer import TransformerDecoder

from sisyphus import tk

from i6_core.util import uopen

from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm

def _is_separator(tensor: torch.Tensor, vocab: Vocabulary, nbest: int) -> list[list, list]:
    with torch.no_grad():
        batch_size = tensor.size(0)
        start_sep = vocab.label_to_id("Z@@")
        end_sep = vocab.label_to_id("Z")
        idxs = torch.where(tensor == start_sep)
        n = len(idxs[1])
        m = tensor.size(1)
        final_idxs = [[], []]
        idxs_cnt = dict.fromkeys(list(range(batch_size)), 0)
        i = 0
        for b in range(batch_size):
            for _ in range(nbest - 1):
                found_all = 0
                first_idx = None
                while found_all == 0:
                    for j in range(4):
                        if i >= n or idxs[0][i].item() != b:
                            idxs_cnt[b] += 1
                            final_idxs[0].append(torch.tensor(b, device=tensor.device))
                            final_idxs[1].append(torch.tensor(-1, device=tensor.device))
                            found_all = 2
                            break
                        else:
                            if j > 0 and idxs[1][i - 1] + 1 != idxs[1][i]:
                                break
                            elif j == 3:
                                found_all = 1
                                break
                            elif j == 0:
                                first_idx = i
                            i += 1
                    if found_all == 1:
                        if tensor[idxs[0][i], idxs[1][i] + 1] == end_sep:
                            idxs_cnt[b] += 1
                            final_idxs[0].append(idxs[0][first_idx])
                            final_idxs[1].append(idxs[1][first_idx])
                            i += 1
                        else:
                            found_all = 0
                            i = first_idx + 1
        for b in range(batch_size):
            assert idxs_cnt[b] == nbest - 1, f"Batch {b} has {idxs_cnt[b]} separators, should have {nbest - 1}"
        return final_idxs
    
def _split_on_sep(tensor: torch.Tensor, sizes: torch.Tensor, vocab_dim: Dim, nbest: int) -> tuple[list[rf.Tensor], list[Dim]]:
    idxs = _is_separator(tensor, vocab_dim.vocab, nbest)
    batch_size = tensor.size(0)
    assert len(idxs[0]) == batch_size * (nbest - 1), f"Not enough separators found: {len(idxs[0])}, should be {batch_size * (nbest - 1)}"
    ret = []
    new_sizes = []
    old_lengths = [-5] * batch_size
    for n in range(nbest):
        if n < nbest - 1:
            lengths = [(idxs[1][i] if idxs[1][i].item() != -1 else sizes[int((i - n) / (nbest - 1))]) for i in range(len(idxs[0])) if (i - n) % (nbest - 1) == 0]
        else:
            lengths = [sizes[i] for i in range(batch_size)]
        assert len(lengths) == batch_size, f"Lengths: {len(lengths)}, should be {batch_size}"
        new_list = []
        for i in range(batch_size):
            if lengths[i].item() == -1:
                new_list.append(torch.tensor([], dtype=tensor.dtype, device=tensor.device))
            else:
                t_slice = tensor[i, (old_lengths[i] + 5):lengths[i]]
                new_list.append(t_slice)
        new_s = [l.size(0) for l in new_list]
        max_length = max(new_s)
        new_list = [torch.cat([t_slice, torch.tensor([0] * (max_length - t_slice.size(0)), device=tensor.device)]) for t_slice in new_list]
        new_tensor = torch.stack(new_list, dim=0)
        new_tensor = new_tensor.to(torch.int32)
        new_s = torch.tensor(new_s, dtype=torch.int32, device=tensor.device)
        new_s = rf.convert_to_tensor(new_s, dims=(batch_dim,))
        new_s = Dim(new_s, name="out_spatial", dyn_size_ext=new_s)
        new_tensor = rf.convert_to_tensor(new_tensor, dims=(batch_dim, new_s), sparse_dim=vocab_dim)
        ret.append(new_tensor)
        new_sizes.append(new_s)
        old_lengths = lengths
    return ret, new_sizes

def _rescore(
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    model: Model,
    hyperparameters: dict,
    prior_file: tk.Path = None,
) -> rf.Tensor:
    import json
    
    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    
    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune
        
    # Subtract prior of labels if available
    if prior_file and prior_weight > 0.0:
        prior = np.loadtxt(prior_file, dtype="float32")
        prior *= prior_weight
        prior = torch.tensor(prior, dtype=torch.float32, device=dev)
        prior[model.blank_idx] = float("-inf")
        prior = torch.log_softmax(prior, dim=-1)
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
        
    assert lm_name.startswith("ffnn")
    assert model.train_language_model
    assert model.train_language_model.vocab_dim == model.target_dim
    lm: FeedForwardLm = model.train_language_model
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]
    
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
    logits_packed, pack_dim = rf.pack_padded(
        lm_logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    ) # TODO necessary?
    assert logits_packed.dims == (*batch_dims, pack_dim, model.target_dim)
    lm_log_probs = rf.log_softmax(logits_packed, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
    lm_log_probs *= lm_scale
    if prior_file and prior_weight > 0.0:
        lm_log_probs = lm_log_probs - prior
    lm_log_probs = rf.gather(lm_log_probs, axis=model.target_dim, indices=targets_w_eos)
    lm_log_probs = rf.reduce_sum(lm_log_probs, axis=pack_dim)
    return lm_log_probs


def ctc_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    nbest = config.int("ps_nbest", 1)
    decode_every_step = config.bool("decode_every_step", False)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    log_probs = model.log_probs_wb_from_logits(logits)
    
    if decode_every_step:
        def _output_hyps(hyp: list) -> list:
            prev = None
            ls = []
            for h in hyp:
                if h != prev:
                    ls.append(h)
                    prev = h
            ls = [h for h in ls if h != model.blank_idx]
            return ls
        
        if nbest > 1:
            raise NotImplementedError("nbest > 1 with decode_every_step not implemented")
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        prior_file = config.typed_value("empirical_prior")
        assert hyperparameters and prior_file
        if "decay" in hyperparameters and hyperparameters["decay"] < 1.0:
            curr_step = rf.get_run_ctx().step
            assert isinstance(curr_step, int)
            decay = hyperparameters.pop("decay")
            decay_limit = hyperparameters.pop("decay_limit", 0.0)
            start_weight = hyperparameters["lm_weight"]
            hyperparameters["lm_weight"] = decay_limit + ((start_weight - decay_limit) * decay ** curr_step)
            print("LM weight:", hyperparameters["lm_weight"])
        with torch.no_grad():
            batch_dims = data.remaining_dims(data_spatial_dim)
            hyps = recog_ffnn(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, train_lm=True)
        assert len(hyps[0]) == 1
        hyps = [_output_hyps(hyps_batch[0]) for hyps_batch in hyps]
        if len(hyps[0]) < 2:
            print("SHORT HYP:", hyps[0])
        lengths = [len(h) for h in hyps]
        lengths2 = [l + 1 for l in lengths]
        max_length = max(lengths)
        targets_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim = rf.convert_to_tensor(targets_spatial_dim, dims=(batch_dim,))
        targets_spatial_dim = Dim(targets_spatial_dim, name="out_spatial", dyn_size_ext=targets_spatial_dim)
        targets_spatial_dim2 = torch.tensor(lengths2, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim2 = rf.convert_to_tensor(targets_spatial_dim2, dims=(batch_dim,))
        targets_spatial_dim2 = Dim(targets_spatial_dim2, name="out_spatial2", dyn_size_ext=targets_spatial_dim2)
        hyps = [h + [0] * (max_length - len(h)) for h in hyps]
        hyps = torch.tensor(hyps, dtype=torch.int32, device=data.raw_tensor.device)
        targets = rf.convert_to_tensor(hyps, dims=(batch_dim, targets_spatial_dim), sparse_dim=model.target_dim)
    
    if nbest > 1:
        from .sum_criterion import safe_logaddexp
        
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        prior_file = config.typed_value("empirical_prior")
        assert hyperparameters and prior_file
        
        tensor_ls, sizes_ls = _split_on_sep(targets.raw_tensor, targets_spatial_dim.dyn_size_ext.raw_tensor, model.target_dim, nbest)
        n = len(tensor_ls)
        
        loss_sum = None
        if aux_loss_layers:
            aux_probs = {}
            for i, layer_idx in enumerate(aux_loss_layers):
                aux_loss_sum = {}
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                aux_probs[i] = model.log_probs_wb_from_logits(aux_logits)
        
        for j in range(n):
            targets_s = tensor_ls[j]
            targets_spatial_dim_s = sizes_ls[j]
            
            # TODO add alignment prior
            lm_prior_score = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file).raw_tensor
            
            if config.bool("use_eos_postfix", False):
                targets_s, (targets_spatial_dim_s,) = rf.pad(
                    targets_s, axes=[targets_spatial_dim_s], padding=[(0, 1)], value=model.eos_idx
                )

            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers):
                        continue
                    aux_loss = rf.ctc_loss(
                        logits=aux_probs[i],
                        logits_normalized=True,
                        targets=targets_s,
                        input_spatial_dim=enc_spatial_dim,
                        targets_spatial_dim=targets_spatial_dim_s,
                        blank_index=model.blank_idx,
                    )
                    if i in aux_loss_sum:
                        aux_loss_sum[i] = safe_logaddexp(aux_loss_sum[i], (-aux_loss).raw_tensor + lm_prior_score)
                    else:
                        aux_loss_sum[i] = (-aux_loss).raw_tensor + lm_prior_score
                    

            loss = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets_s,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim_s,
                blank_index=model.blank_idx,
            )
            if loss_sum is not None:
                loss_sum = safe_logaddexp(loss_sum, (-loss).raw_tensor + lm_prior_score)
            else:
                loss_sum = (-loss).raw_tensor + lm_prior_score
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
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
        return
        
    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = rf.ctc_loss(
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
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor() if not decode_every_step else targets_spatial_dim2.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    loss = rf.ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor() if not decode_every_step else targets_spatial_dim2.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

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

 
def full_sum_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, lm_path: tk.Path, seq_tags: rf.Tensor = None, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    from i6_experiments.users.mueller.experiments.ctc_baseline.sum_criterion import sum_loss, sum_loss2, safe_logsumexp, PrintGradients, NormGradients
    
    # torch.autograd.set_detect_anomaly(True)
    pg = PrintGradients.apply
    ng = NormGradients.apply
    
    def _calc_log_prior(log_probs: torch.Tensor, lengths: torch.Tensor, use_max: bool = False, separate_eos: bool = False) -> torch.Tensor:
        lengths = lengths.to(log_probs.device)
        assert lengths.size(0) == log_probs.size(0), "Prior calculation batch lengths are not the same (full_sum)!"
        
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

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    
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
    use_prior = prior_scale > 0.0
    
    print_gradients = config.bool("print_gradients", False)
    version = config.int("version", 2)
    if version == 4:
        am_scale = 1.0
        lm_scale = 1.0
        prior_scale = 0.0
        use_prior = prior_scale > 0.0
        blank_correction_version = 16
        correction_in_final_score = True
        top_k = 1
        print_gradients = True
    
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    if not lm_path.startswith("ffnn"):
        with uopen(lm_path, "rb") as f:
            lm = torch.load(f, map_location=data.device)
            assert isinstance(lm, torch.Tensor), "Loaded LM is not a tensor"
        lm_order = len(lm.size())
        lm = torch.log_softmax(lm, dim=-1)
    else:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
        lm_order = int(lm_path[len("ffnn"):])
        raise NotImplementedError("FFNN LM not implemented")

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_log_probs = aux_log_probs.raw_tensor
            if use_prior:
                if empirical_prior is not None:
                    aux_log_prior = np.loadtxt(empirical_prior, dtype="float32")
                    aux_log_prior = torch.tensor(aux_log_prior, device=log_probs.device)
                    assert aux_log_prior.size(0) == log_probs.size(2), "Empirical prior size does not match (full_sum)!"
                else:
                    aux_log_prior = _calc_log_prior(aux_log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
                    if not prior_gradient:
                        aux_log_prior = aux_log_prior.detach()
                        
                if not blank_prior:
                    aux_log_prior[model.blank_idx] = float("-inf")
                    aux_log_prior = torch.log_softmax(aux_log_prior, dim=-1)
            else:
                aux_log_prior = None
            # (B, T, F) -> (T, B, F)
            aux_log_probs = aux_log_probs.permute(1, 0, 2)
            aux_loss = sum_loss(
                log_probs=aux_log_probs,
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
                device=aux_log_probs.device,
                alignment_topk=alignment_topk
            )
            aux_loss = rtf.TorchBackend.convert_to_tensor(aux_loss, dims = [batch_dim], dtype = "float32", name=f"aux_full_sum_{layer_idx}")
            aux_loss.mark_as_loss(
                f"aux_full_sum_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
    
    fixed_seqs = ["train-other-500/5756-305214-0041/5756-305214-0041", "train-clean-360/2498-134786-0003/2498-134786-0003", "train-other-500/1643-138089-0068/1643-138089-0068"] # MONICA DREW FRESH HOPE FROM HER SON'S WRITINGS THEY WERE FULL OF NOBLE THOUGHTS AND HIGH ASPIRATIONS, HERE IT IS
    print_for_idx = []
    
    # seq = seq_tags[0]
    # idx = np.where(seq_tags == seq)[0]
    # print_for_idx.append(idx[0])
    
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
        log_probs = log_probs.raw_tensor
        for idx_t in print_for_idx:
            print("Target:", targets.raw_tensor[idx_t].detach().cpu().numpy())
            # log_probs = pg(log_probs, "log_probs", alias_name, False, 1, False, idx_t)
            # log_probs = pg(log_probs, "log_probs", alias_name, False, None, False, idx_t, [8, 9, 10, 11], ["<blank>", "H", "<blank>", "ERE"])
            log_probs = pg(log_probs, "log_probs", alias_name, False, None, True, idx_t, [], ["Log Probs"], enc_spatial_dim.dyn_size_ext.raw_tensor[idx_t])
    else:
        log_probs = model.log_probs_wb_from_logits(logits)
        log_probs = log_probs.raw_tensor

    if use_prior:
        if empirical_prior is not None:
            log_prior = np.loadtxt(empirical_prior, dtype="float32")
            log_prior = torch.tensor(log_prior, device=log_probs.device)
            assert log_prior.size(0) == log_probs.size(2), "Empirical prior size does not match (full_sum)!"
        else:
            log_prior = _calc_log_prior(log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor, use_max=max_prior)
            if not prior_gradient:
                log_prior = log_prior.detach()
                
        if not blank_prior:
            log_prior[model.blank_idx] = float("-inf")
            log_prior = torch.log_softmax(log_prior, dim=-1)
    else:
        log_prior = None
        
    # log_probs = ng(log_probs) # TODO
        
    # (B, T, V) -> (T, B, V)
    log_probs = log_probs.permute(1, 0, 2)
    
    if version == 5:
        loss = torch.ctc_loss(
            log_probs,
            targets.raw_tensor,
            enc_spatial_dim.dyn_size_ext.raw_tensor,
            targets_spatial_dim.dyn_size_ext.raw_tensor,
            blank=model.blank_idx,
            reduction=0,
            zero_infinity=False
        )
        if print_gradients and fixed_seqs[1] in seq_tags:
            print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9210505], 0.0009867928 , 1: [0.00390251], 0.99610509
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"ctc")
        loss.mark_as_loss(
            "ctc",
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
    elif version in [10, 11]:
        if version == 10:
            loss = torch.logsumexp(log_probs[0], dim=-1)
        else:
            loss = safe_logsumexp(log_probs[0], -1)
        for t in range(1, log_probs.size(0)):
            if version == -1:
                A = loss.unsqueeze(1)
                B = log_probs[t].unsqueeze(1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                new_loss = A.matmul(B).squeeze(1)
                time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs.device)).unsqueeze(-1)
                loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
            else:
                # A = loss.unsqueeze(1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                # B = log_probs[t].unsqueeze(-1).expand(-1, log_probs.size(-1), log_probs.size(-1))
                # new_loss = safe_logsumexp(A + B, dim=-1)
                # new_loss = safe_logsumexp(torch.stack([loss, safe_logsumexp(log_probs[t], -1)], dim=-1), dim=-1)
                if version == 10:
                    new_loss = loss + torch.logsumexp(log_probs[t], -1)
                else:
                    new_loss = loss + safe_logsumexp(log_probs[t], -1)
                time_mask = (t < enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs.device))#.unsqueeze(-1)
                loss = torch.where(time_mask.expand_as(new_loss), new_loss, loss)
        if version == -1:
            loss = loss.sum(-1)
        # else:
        #     loss = safe_logsumexp(loss, dim=-1)
        loss = -loss
        # print(loss[0].detach().cpu().numpy())
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"sum")
        loss.mark_as_loss(
            f"sum",
            custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )
    else:
        loss = sum_loss2(
            log_probs=log_probs,
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
            device=log_probs.device,
            print_best_path_for_idx=print_for_idx,
            alignment_topk=alignment_topk,
            blank_correction_version=blank_correction_version,
            correction_in_final_score = correction_in_final_score
        )
        if print_gradients and fixed_seqs[1] in seq_tags:
            print("Loss:", loss[np.where(seq_tags == fixed_seqs[1])[0]].detach().cpu().numpy()) # 0: [6.9392214] 0.0009690238, 1: [0.01604532], 0.984082720
        loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"full_sum")
        loss.mark_as_loss(
            f"full_sum",
            custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
            use_normalized_loss=use_normalized_loss,
        )


def ce_train(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    nbest = config.int("ps_nbest", 1)
    decode_every_step = config.bool("decode_every_step", False)
    assert not decode_every_step
    assert nbest == 1

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    
    # batch_dims = data.remaining_dims(data_spatial_dim)
    # logits, pack_dim = rf.pack_padded(
    #     logits, dims=batch_dims + [enc_spatial_dim], enforce_sorted=False
    # )
    log_probs = model.log_probs_wb_from_logits(logits)
    
    if decode_every_step:
        def _output_hyps(hyp: list) -> list:
            prev = None
            ls = []
            for h in hyp:
                if h != prev:
                    ls.append(h)
                    prev = h
            ls = [h for h in ls if h != model.blank_idx]
            return ls
        
        if nbest > 1:
            raise NotImplementedError("nbest > 1 with decode_every_step not implemented")
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        prior_file = config.typed_value("empirical_prior")
        assert hyperparameters and prior_file
        if "decay" in hyperparameters and hyperparameters["decay"] < 1.0:
            curr_step = rf.get_run_ctx().step
            assert isinstance(curr_step, int)
            decay = hyperparameters.pop("decay")
            decay_limit = hyperparameters.pop("decay_limit", 0.0)
            start_weight = hyperparameters["lm_weight"]
            hyperparameters["lm_weight"] = decay_limit + ((start_weight - decay_limit) * decay ** curr_step)
            print("LM weight:", hyperparameters["lm_weight"])
        with torch.no_grad():
            batch_dims = data.remaining_dims(data_spatial_dim)
            hyps = decode_albert(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, train_lm=True)
        assert len(hyps[0]) == 1
        hyps = [_output_hyps(hyps_batch[0]) for hyps_batch in hyps]
        if len(hyps[0]) < 2:
            print("SHORT HYP:", hyps[0])
        lengths = [len(h) for h in hyps]
        lengths2 = [l + 1 for l in lengths]
        max_length = max(lengths)
        targets_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim = rf.convert_to_tensor(targets_spatial_dim, dims=(batch_dim,))
        targets_spatial_dim = Dim(targets_spatial_dim, name="out_spatial", dyn_size_ext=targets_spatial_dim)
        targets_spatial_dim2 = torch.tensor(lengths2, dtype=torch.int32, device=data.raw_tensor.device)
        targets_spatial_dim2 = rf.convert_to_tensor(targets_spatial_dim2, dims=(batch_dim,))
        targets_spatial_dim2 = Dim(targets_spatial_dim2, name="out_spatial2", dyn_size_ext=targets_spatial_dim2)
        hyps = [h + [0] * (max_length - len(h)) for h in hyps]
        hyps = torch.tensor(hyps, dtype=torch.int32, device=data.raw_tensor.device)
        targets = rf.convert_to_tensor(hyps, dims=(batch_dim, targets_spatial_dim), sparse_dim=model.target_dim)
    
    if nbest > 1:
        from .sum_criterion import safe_logaddexp
        
        hyperparameters = config.typed_value("hyperparameters_decoder").copy()
        prior_file = config.typed_value("empirical_prior")
        assert hyperparameters and prior_file
        
        tensor_ls, sizes_ls = _split_on_sep(targets.raw_tensor, targets_spatial_dim.dyn_size_ext.raw_tensor, model.target_dim, nbest)
        n = len(tensor_ls)
        
        loss_sum = None
        if aux_loss_layers:
            aux_probs = {}
            for i, layer_idx in enumerate(aux_loss_layers):
                aux_loss_sum = {}
                linear = getattr(model, f"enc_aux_logits_{layer_idx}")
                aux_logits = linear(collected_outputs[str(layer_idx - 1)])
                aux_probs[i] = model.log_probs_wb_from_logits(aux_logits)
        
        for j in range(n):
            targets_s = tensor_ls[j]
            targets_spatial_dim_s = sizes_ls[j]
            
            # TODO add alignment prior
            lm_prior_score = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file).raw_tensor
            
            if config.bool("use_eos_postfix", False):
                targets_s, (targets_spatial_dim_s,) = rf.pad(
                    targets_s, axes=[targets_spatial_dim_s], padding=[(0, 1)], value=model.eos_idx
                )

            if aux_loss_layers:
                for i, layer_idx in enumerate(aux_loss_layers):
                    if layer_idx > len(model.encoder.layers):
                        continue
                    aux_loss = rf.ctc_loss(
                        logits=aux_probs[i],
                        logits_normalized=True,
                        targets=targets_s,
                        input_spatial_dim=enc_spatial_dim,
                        targets_spatial_dim=targets_spatial_dim_s,
                        blank_index=model.blank_idx,
                    )
                    if i in aux_loss_sum:
                        aux_loss_sum[i] = safe_logaddexp(aux_loss_sum[i], (-aux_loss).raw_tensor + lm_prior_score)
                    else:
                        aux_loss_sum[i] = (-aux_loss).raw_tensor + lm_prior_score
                    

            loss = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets_s,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim_s,
                blank_index=model.blank_idx,
            )
            if loss_sum is not None:
                loss_sum = safe_logaddexp(loss_sum, (-loss).raw_tensor + lm_prior_score)
            else:
                loss_sum = (-loss).raw_tensor + lm_prior_score
        if aux_loss_layers:
            for i, layer_idx in enumerate(aux_loss_layers):
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
        return
        
    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = rf.cross_entropy(
                target=targets, estimated=aux_log_probs, estimated_type="log-probs", axis=model.wb_target_dim
            )
            aux_loss.mark_as_loss(f"ce_{layer_idx}", scale=aux_loss_scales[i], use_normalized_loss=use_normalized_loss)
            
    # targets, _ = rf.pack_padded(
    #     targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    # )

    loss = rf.cross_entropy(
        target=targets, estimated=log_probs, estimated_type="log-probs", axis=model.wb_target_dim
    )
    loss.mark_as_loss("ce", use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits, axis=model.wb_target_dim)
    frame_error = best != targets
    frame_error.mark_as_loss(name="fer", as_error=True)

