from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Tuple

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

if TYPE_CHECKING:
    import torch


class CtcPrefixScorer:
    """
    ESPnet label scorer

    Copied from denoising LM...
    original derived from i6_experiments.users.zeyer.decoding.beam_search_torch.interface.LabelScorerIntf
    """

    # original: get_initial_state
    def __init__(
        self,
        *,
        log_probs: Tensor,
        batch_dims: Sequence[Dim],
        enc_spatial_dim: Dim,
        vocab_dim: Dim,
        vocab_wb_dim: Dim,
        blank_idx: int,
        eos_idx: int,
    ):
        """
        :param log_probs: shape [Batch, Spatial, Vocab]
        :param batch_dims: batch dims
        :param enc_spatial_dim: spatial dim
        :param vocab_dim: vocab dim, without blank
        :param vocab_wb_dim: vocab dim. we expect that this includes both blank and EOS.
        :param blank_idx: blank
        :param eos_idx: EOS
        """
        from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH

        assert log_probs.dims_set == set(batch_dims) | {enc_spatial_dim, vocab_wb_dim}
        assert enc_spatial_dim.dyn_size_ext.dims_set == set(batch_dims)
        self.batch_dims = list(batch_dims)
        self.enc_spatial_dim = enc_spatial_dim
        self.vocab_dim = vocab_dim
        self.vocab_wb_dim = vocab_wb_dim
        self.state01_dim = Dim(2, name="state01")
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx

        if len(self.batch_dims) == 1:
            self.batch_dim = self.batch_dims[0]
            enc_seq_lens = enc_spatial_dim.dyn_size_ext
        else:
            log_probs, self.batch_dim = rf.merge_dims(log_probs, dims=self.batch_dims)
            enc_seq_lens, _ = rf.merge_dims(enc_spatial_dim.dyn_size_ext, dims=self.batch_dims, out_dim=self.batch_dim)

        # espnet.nets.scorers.ctc.CTCPrefixScorer.batch_init_state incorrectly assumes batch_size=1,
        # and is wrong otherwise, thus we don't use that here.
        # Instead, directly use CTCPrefixScoreTH.
        self._espnet_ctc_prefix_score_th = CTCPrefixScoreTH(
            log_probs.copy_compatible_to_dims_raw([self.batch_dim, enc_spatial_dim, vocab_wb_dim]),
            enc_seq_lens.copy_compatible_to_dims_raw([self.batch_dim]),
            blank_idx,
            eos_idx,
        )

    @staticmethod
    def initial_state():
        return None

    def score_and_update_state(self, *, prev_state: Any, prev_label: Tensor, beam_dim: Dim) -> Tuple[Tensor, Any]:
        """
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape {Batch..., Beam, ...}.
        :param prev_label: shape {Batch..., Beam} -> index in [0...Label-1].
            Use some dummy value for the first step (e.g. SOS).
        :param beam_dim: beam dim
        :return: (scores, state).
            scores: shape {Batch..., Beam, Label}, log-prob-like scores.
            state: all tensors are expected to have shape {Batch..., Beam, ...}.
        """
        import torch
        import tree

        assert prev_label.dims_set == set(self.batch_dims) | {beam_dim} and prev_label.sparse_dim == self.vocab_dim
        if len(self.batch_dims) != 1:
            prev_label, _ = rf.merge_dims(prev_label, dims=self.batch_dims, out_dim=self.batch_dim)
        prev_label = _target_extend_blank(
            prev_label, target_dim=self.vocab_dim, wb_target_dim=self.vocab_wb_dim, blank_idx=self.blank_idx
        )
        prev_label_raw = prev_label.copy_compatible_to_dims_raw([self.batch_dim, beam_dim])
        batch_size, beam_size = prev_label_raw.shape

        if prev_state is not None:

            def _map(x):
                if x is None:
                    return None
                if isinstance(x, Dim):
                    assert not x.dyn_size_ext  # not implemented...
                    return x
                assert isinstance(x, Tensor) and x.dims_set.issuperset(self.batch_dims)
                x, _ = rf.merge_dims(x, dims=self.batch_dims, out_dim=self.batch_dim)
                return x

            if len(self.batch_dims) != 1:
                prev_state = tree.map_structure(_map, prev_state)
            ys, out_spatial_dim, prev_state = prev_state
            ys, out_spatial_dim = rf.cum_concat_step(
                prev_label, prev_accum=ys, axis=out_spatial_dim
            )  # [batch,beam,out_len]
        else:
            out_spatial_dim = Dim(1, name="out_spatial")
            ys = rf.expand_dim(prev_label, out_spatial_dim)  # [batch,beam,out_len]
        assert ys.dims_set == {self.batch_dim, beam_dim, out_spatial_dim}
        ys_raw = ys.copy_compatible_to_dims_raw([self.batch_dim, beam_dim, out_spatial_dim])
        ys_raw_flat = ys_raw.flatten(0, 1)  # [batch*beam,out_len]

        # Convert all [batch,beam,...] tensors to [batch*beam,...].
        def _map(x):
            if x is None:
                return None
            assert isinstance(x, Tensor) and x.dims_set.issuperset((self.batch_dim, beam_dim))
            x_raw = x.copy_compatible_to_dims_raw(
                [self.batch_dim, beam_dim] + x.remaining_dims([self.batch_dim, beam_dim])
            )
            return x_raw.flatten(0, 1)

        prev_state_raw = tree.map_structure(_map, prev_state)

        # if isinstance(espnet_scorer, CTCPrefixScorer):
        # Unfortunately the CTCPrefixScorer breaks our assumption that the batch dim is the first dim.
        # Thus, we must permute the corresponding entries in the state.
        # Also, the initial state is None, so we need to cover this case as well.
        if prev_state_raw is not None:
            # 4-tuple. first has batch in dim=2, second has batch in dim=0, third and forth don't have batch?
            # n_bh = self.batch * n_hyps. snum = odim.
            # first: r: (self.input_length, 2, n_bh, snum) in func,
            #   then with select_state resulting in: (in_len, 2, batch * new_n_hyps)
            #   or: r_prev: (self.input_length, 2, self.batch * n_hyps)
            # second: log_psi: (n_bh, self.odim) in func,
            #   then with select_state resulting in: (batch * new_n_hyps, self.odim) ?
            # third/forth: f_min, f_max: scalars, no batch, only used anyway with att_w, can just set 0 and 1.
            # we even get a fifth as output: scoring_idmap: but not used.
            # So, only care about first, second.
            # Apply the select_state logic here, i.e. espnet.nets.scorers.ctc.CTCPrefixScorer.select_state.
            r, log_psi = prev_state_raw
            r: torch.Tensor  # [batch*beam,in_len,2,snum]
            r = _batch_gather_torch(r, indices=prev_label_raw.flatten(), index_dim=3)  # [batch*beam,in_len,2]
            r = r.permute(1, 2, 0)  # [in_len,2,batch*beam]
            log_psi: torch.Tensor  # [batch*beam,odim]
            log_psi = _batch_gather_torch(log_psi, indices=prev_label_raw.flatten())  # [batch*beam]
            log_psi = log_psi[:, None]  # [batch*beam,1]. must broadcast to [batch*beam,odim]
            prev_state_raw = (r, log_psi, 0, 1)

        # Inline espnet.nets.scorers.ctc.CTCPrefixScorer.batch_score_partial,
        # as we already have it batched.
        scores, states = self._espnet_ctc_prefix_score_th(ys_raw_flat, prev_state_raw)
        # scores: (n_bh, vocab)
        scores = scores.unflatten(0, (batch_size, beam_size))  # [batch,beam,vocab]
        scores_rf = rf.convert_to_tensor(scores, dims=[self.batch_dim, beam_dim, self.vocab_wb_dim])
        r, log_psi = states[:2]
        r: torch.Tensor  # [in_len,2,batch*beam,snum]
        r = r.permute(2, 0, 1, 3)  # [batch*beam,in_len,2,snum]
        r = r.unflatten(0, (batch_size, beam_size))  # [batch,beam,in_len,2,snum]
        r_rf = rf.convert_to_tensor(
            r, dims=[self.batch_dim, beam_dim, self.enc_spatial_dim, self.state01_dim, self.vocab_wb_dim]
        )
        # log_psi: (n_bh, odim)
        log_psi = log_psi.unflatten(0, (batch_size, beam_size))  # [batch,beam,odim]
        log_psi_rf = rf.convert_to_tensor(log_psi, dims=[self.batch_dim, beam_dim, self.vocab_wb_dim])

        scores_rf = _target_dense_remove_blank(
            scores_rf, target_dim=self.vocab_dim, wb_target_dim=self.vocab_wb_dim, blank_idx=self.blank_idx
        )

        if len(self.batch_dims) != 1:

            def _map(x):
                if x is None:
                    return None
                if isinstance(x, Dim):
                    assert not x.dyn_size_ext or self.batch_dim not in x.dyn_size_ext.dims  # not implemented...
                    return x
                assert isinstance(x, Tensor) and self.batch_dim in x.dims
                return rf.split_dims(x, axis=self.batch_dim, dims=self.batch_dims)

            scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf)) = tree.map_structure(
                _map, (scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf)))
            )
        return scores_rf, (ys, out_spatial_dim, (r_rf, log_psi_rf))


def _target_dense_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert wb_target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.slice(target, axis=wb_target_dim, size=target_dim)
    return res


def _target_extend_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, wb_target_dim)


# noinspection PyShadowingNames
def _batch_gather_torch(
    values: torch.Tensor, *, indices: torch.Tensor, batch_dim: int = 0, index_dim: int = 1
) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :param batch_dim: in values. in indices, batch is assumed first.
    :param index_dim: in values. must be >batch_dim (not implemented otherwise).
        in indices, index dims are expected after batch.
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...],
        if batch_dim=0 and index_dim=1.
        Batch and index dim stays at the same place, index dim is replaced by indices dims from indices.
    """
    import torch

    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[batch_dim] and batch_dim < index_dim
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good, [Batch,IndexDim]
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]  # [Batch,IndexDim=1]
    else:
        indices_flat = indices.flatten(1)  # [Batch,FlatIndexDim]
    indices_flat_bc = indices_flat.reshape(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else 1)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,1s...].
    indices_flat_exp = indices_flat_bc.expand(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else d)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,ValuesDims...]
    out = torch.gather(values, dim=index_dim, index=indices_flat_exp.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(index_dim)
    else:
        out = out.unflatten(index_dim, indices.shape[1:])
    if batch_dim == 0 and index_dim == 1:
        assert out.shape == indices.shape + values.shape[2:]
    return out
