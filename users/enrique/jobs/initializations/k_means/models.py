from typing import Optional, Tuple, Sequence, Dict, Any
import logging
import numpy as np
from sympy import python
import torch
import random
from sisyphus import tk


import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.datasets.util.vocabulary import Vocabulary

from i6_experiments.users.enrique.experiments.marten_exp.language_models.ffnn import FeedForwardLm

OUT_BLANK_LABEL = "<blank>"


class Wav2VecModel(rf.Module):
    """Model definition"""

    def __init__(
        self,
        *,
        w2v_opts: Dict[str, Any],
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        train_language_model: Optional[FeedForwardLm] = None,
        recog_language_model: Optional[FeedForwardLm] = None,
        rescore_language_model: Optional[FeedForwardLm] = None,
    ):
        super(Wav2VecModel, self).__init__()

        import transformers
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        w2v_config_file = w2v_opts["config_file"]
        wav2vec_config = transformers.Wav2Vec2Config.from_pretrained(w2v_config_file)

        self.pca_enabled = False
        if not w2v_opts.get("pca_dim", None) == None:
            print("PCA is enabled, with dimension:", w2v_opts["pca_dim"])
            self.pca_enabled = True
            self.pca_dim = w2v_opts["pca_dim"]
            self.n_points_to_calculate_pca = w2v_opts["n_points_to_calculate_pca"]

            # pca_transform_matrix = w2v_opts.get(
            #     "pca_transform_matrix_tk_path",
            #     tk.Path(
            #         "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing/Wav2VecUFeaturizeAudioJob.Zoop3SkhqfaA/output/audio_features/pca/512_pca_A.npy"
            #     ),
            # )
            # pca_bias_matrix = w2v_opts.get(
            #     "pca_bias_matrix_tk_path",
            #     tk.Path(
            #         "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing/Wav2VecUFeaturizeAudioJob.Zoop3SkhqfaA/output/audio_features/pca/512_pca_b.npy"
            #     ),
            # )

            pca_transform_matrix = w2v_opts.get("pca_transform_matrix_tk_path", None)
            pca_bias_matrix = w2v_opts.get("pca_bias_matrix_tk_path", None)

            if pca_transform_matrix and pca_bias_matrix:
                logging.warning(
                    f"[Wav2VecModel] Using PCA transform and bias matrices, n_points_to_calculate_pca={self.n_points_to_calculate_pca} will be ingored"
                )

                def load_or_convert(path: str) -> torch.Tensor:
                    if path.endswith(".pt"):
                        return torch.load(path)
                    elif path.endswith(".npy"):
                        # Convert and save as .pt next to the .npy file
                        data = np.load(path)
                        pt_path = path[:-4] + ".pt"
                        torch.save(torch.tensor(data, dtype=torch.float32), pt_path)
                        print(f"[Wav2VecModel] Converted {path} to {pt_path}")
                        return torch.tensor(data, dtype=torch.float32)
                    else:
                        raise ValueError(f"Unsupported file format for PCA: {path}")

                self.pca_transform = load_or_convert(pca_transform_matrix.get_path())
                self.pca_bias = load_or_convert(pca_bias_matrix.get_path())
                self.pca_ready = True
                assert self.pca_transform.dim() == 2, "PCA transform must be 2D"
                assert self.pca_bias.dim() == 1, "PCA bias must be 1D"
                assert (
                    self.pca_transform.shape[1] == self.pca_bias.shape[0]
                ), "Bias must match output dimension of transform"
                
            else:
                assert (
                    not pca_transform_matrix and not pca_bias_matrix
                ), "pca_transform_matrix and pca_bias_matrix must be either both provided, or both None."

                self.pca_transform = None
                self.pca_bias = None
                self.pca_ready = False

                

        self._current_extracted_features = None

        self.wav2vec2 = transformers.Wav2Vec2Model(wav2vec_config)
        self.wav2vec2.freeze_feature_encoder()

        if not w2v_opts.get("use_spec_augment", True):
            self.wav2vec2.config.apply_spec_augment = False

        if w2v_opts["freeze_encoder_first_n_steps"] > 0:
            self.set_wav2vec_encoder_trainable(False, except_layers=w2v_opts.get("unfrozen_encoder_layers", None))

        self.output_raw_features = w2v_opts.get("output_raw_features", False)

        num_enc_layers = w2v_opts.get("num_enc_layers", len(self.wav2vec2.encoder.layers))
        if num_enc_layers != len(self.wav2vec2.encoder.layers):
            assert num_enc_layers < len(self.wav2vec2.encoder.layers)
            n_layers_to_remove = len(self.wav2vec2.encoder.layers) - num_enc_layers
            for i in range(n_layers_to_remove):
                del self.wav2vec2.encoder.layers[-1]

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1

        if target_dim.vocab and not wb_target_dim.vocab:
            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
            )
        self.wb_target_dim = wb_target_dim

        w2v_hidden_size = self.wav2vec2.encoder.layers[0].feed_forward.output_dense.out_features
        self.enc_out_dim = Dim(name="enc", dimension=w2v_hidden_size, kind=Dim.Types.Feature)

        if config.bool("use_subsampled_enc_logits", False):
            # Subsampled encoder logits.
            self.enc_logits = EncLogitsSubsample(
                in_dim=self.enc_out_dim,
                out_dim=wb_target_dim,
            )
        else:
            enc_logits = []
            enc_logits_n_layers = w2v_opts.get("enc_logits_n_layers", 1)
            for i in range(enc_logits_n_layers):
                if i == enc_logits_n_layers - 1:
                    out_dim = wb_target_dim
                else:
                    out_dim = self.enc_out_dim
                enc_logits.append(rf.Linear(self.enc_out_dim, out_dim))

                if i != enc_logits_n_layers - 1:
                    enc_logits.append(rf.relu)

            if len(enc_logits) > 1:
                self.enc_logits = rf.Sequential(enc_logits)
            else:
                self.enc_logits = rf.Linear(self.enc_out_dim, wb_target_dim)

        model_prior = config.typed_value("model_prior", {"type": "batch-wise"})
        if model_prior["type"] == "exp-moving-average":
            self.model_prior = rf.RunningMean(wb_target_dim, alpha=model_prior["alpha"], is_prob_distribution=True)

        self.train_language_model = train_language_model
        self.recog_language_model = recog_language_model
        self.rescore_language_model = rescore_language_model
        self.decoder = None

    def set_wav2vec_encoder_trainable(self, trainable: bool, except_layers: Optional[Sequence[int]] = None):
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = trainable
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = trainable

        if except_layers is not None:
            for layer_idx in except_layers:
                print(f"Setting layer {layer_idx} trainable: {not trainable}")
                for param in self.wav2vec2.encoder.layers[layer_idx].parameters():
                    param.requires_grad = not trainable

    def set_param_grads_to_zero(self):
        for param in self.parameters(recurse=True):
            param.raw_tensor.grad = None

    def __call__(
        self,
        source: Tensor,  # [B, T] or [B, T, 1]
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dim]:
        from returnn.config import get_global_config

        config = get_global_config()

        # remove feature_dim if it is 1
        if source.feature_dim and source.feature_dim.dimension == 1:
            source = rf.squeeze(source, axis=source.feature_dim)
            assert not source.feature_dim  # raw audio

        source_dyn_lengths = source.get_sequence_lengths()

        # preprocessing using returnn -> more efficient
        mask = source.get_sequence_mask_broadcast(in_spatial_dim)
        source_mean = rf.reduce_mean(source, axis=[in_spatial_dim]).raw_tensor.unsqueeze(1)  # [B, 1]
        # replace padded samples by mean in order to set var-contributions to 0 (see source_var)
        source_raw = torch.where(mask, source.raw_tensor, source_mean)  # [B, T]

        # normalization denominator only over non-padded frames
        # padded frames in the sum evaluate to 0 because samples are set to the mean
        source_var = (
            1 / (source_dyn_lengths.to(source.device) - 1) * torch.sum((source_raw - source_mean) ** 2, dim=1)
        )  # [B]
        source_var = source_var.unsqueeze(1)  # [B, 1]
        # normalize to 0 mean and unit variance
        source_raw = (source_raw - source_mean) / torch.sqrt(source_var + 1e-7)
        # set padded samples to 0
        source_raw = torch.where(mask, source_raw, 0.0)
        w2v_output = self.wav2vec2(source_raw)
        enc_raw = w2v_output.last_hidden_state

        # Apply PCA
        if self.pca_enabled and self.pca_ready:
            device = enc_raw.device
            self.pca_transform = self.pca_transform.to(device)
            self.pca_bias = self.pca_bias.to(device)

            B, T, F = enc_raw.shape
            enc_raw = torch.matmul(enc_raw, self.pca_transform.T) + self.pca_bias
            self.enc_out_dim = Dim(name="pca_out", dimension=self.pca_dim, kind=Dim.Types.Feature)
            

        self._current_extracted_features = w2v_output.extract_features
        # gradient_penalty_opts = config.typed_value("gradient_penalty_opts", {})
        # if gradient_penalty_opts and rf.get_run_ctx().train_flag:
        #   self._current_extracted_features.requires_grad = True

        # get dyn seq lengths of wav2vec encoder output
        enc_dyn_lengths_raw = source_dyn_lengths
        for conv_layer in self.wav2vec2.feature_extractor.conv_layers:
            enc_dyn_lengths_raw = torch.floor(
                (enc_dyn_lengths_raw - (conv_layer.conv.kernel_size[0] - 1) - 1) / conv_layer.conv.stride[0] + 1
            )
        enc_dyn_lengths_raw = enc_dyn_lengths_raw.to(torch.int32)
        enc_dyn_lengths = rf.Tensor(
            name="wav2vec_dyn_lengths",
            dims=[batch_dim],
            dtype="int32",
            raw_tensor=enc_dyn_lengths_raw,
        )

        enc_spatial_dim = Dim(name="wav2vec_seq", dimension=enc_dyn_lengths, kind=Dim.Types.Spatial)
        enc = rf.Tensor(
            "wav2vec_states",
            dims=[batch_dim, enc_spatial_dim, self.enc_out_dim],
            dtype=rf.get_default_float_dtype(),
            raw_tensor=enc_raw,
        )

        ## Return raw features if requested
        if self.output_raw_features:
            return enc, None, enc_spatial_dim

        else:
            if isinstance(self.enc_logits, EncLogitsSubsample):
                logits, enc_spatial_dim = self.enc_logits(enc, in_spatial_dim=enc_spatial_dim)
            else:
                logits = self.enc_logits(enc)

            if config.bool("collapse_logits_segments", False):
                logits, enc_spatial_dim = collapse_logits_segment(logits, self.wb_target_dim, enc_spatial_dim)

            return logits, enc, enc_spatial_dim

    def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: incl blank
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
        """
        log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)

        # optionally apply blank penalty
        from returnn.config import get_global_config

        config = get_global_config()
        blank_penalty_opts = config.typed_value("blank_penalty_opts", {})
        if blank_penalty_opts:
            blank_penalty = blank_penalty_opts["blank_penalty"]
            blank_penalty = rf.sparse_to_dense(
                self.blank_idx, axis=self.wb_target_dim, label_value=blank_penalty, other_value=0.0
            )
            log_probs -= blank_penalty

        return log_probs


class EncLogitsSubsample(rf.Module):
    def __init__(
        self,
        *,
        in_dim: Dim,
        out_dim: Dim,
    ):
        super(EncLogitsSubsample, self).__init__()

        self.batch_norm = rf.BatchNorm(in_dim)
        self.batch_norm.gamma.initial = 30.0  # as in https://arxiv.org/pdf/2204.02492
        self.linear = rf.Linear(in_dim, in_dim)
        self.conv = rf.Conv1d(
            in_dim=in_dim,
            out_dim=out_dim,
            filter_size=9,
            strides=3,
            with_bias=False,
            padding="valid",  # no padding
        )

    def __call__(
        self,
        x: Tensor,  # [B, T, F]
        *,
        in_spatial_dim: Dim,
    ) -> Tuple[Tensor, Dim]:
        x = self.batch_norm(x)
        inter_x = self.linear(rf.dropout(x, 0.1))
        x = x + inter_x  # residual connection
        x = rf.dropout(x, 0.1)
        x, spatial_dim = self.conv(x, in_spatial_dim=in_spatial_dim)
        x = x.copy_transpose([batch_dim, spatial_dim, self.conv.out_dim])

        return x, spatial_dim


def collapse_logits_segment(logits: Tensor, vocab_dim: Dim, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
    """
    From https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/models/wav2vec_u.py#L146
    """
    logits = logits.copy_transpose([batch_dim, in_spatial_dim, vocab_dim])
    logits_raw = logits.raw_tensor
    padding_mask = ~in_spatial_dim.get_mask(dim_order=[batch_dim, in_spatial_dim]).raw_tensor

    preds = logits_raw.argmax(dim=-1)

    if padding_mask.any():
        preds[padding_mask] = -1  # mark pad
    uniques = []

    bsz, tsz, csz = logits_raw.shape

    for b, p in enumerate(preds):
        uniques.append(p.cpu().unique_consecutive(return_inverse=True, return_counts=True))

    new_tsz = max(u[0].numel() for u in uniques)
    new_logits_raw = logits_raw.new_zeros(bsz, new_tsz, csz)
    new_enc_sizes = rf.Tensor(
        "enc_collapsed_sizes", dims=[batch_dim], dtype="int32", raw_tensor=torch.zeros(bsz, dtype=torch.int32)
    )

    for b in range(bsz):
        u, idx, c = uniques[b]
        keep = u != -1

        if rf.get_run_ctx().train_flag:
            # randomly select index from segment to keep
            u[0] = 0
            u[1:] = c.cumsum(0)[:-1]
            m = c > 1
            r = torch.rand(m.sum())
            o = (c[m] * r).long()
            u[m] += o
            new_logits_raw[b, : u.numel()] = logits_raw[b, u]
        else:
            # mean pool logits over segment
            new_logits_raw[b].index_add_(dim=0, index=idx.to(new_logits_raw.device), source=logits_raw[b])
            new_logits_raw[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits_raw.device)

        new_sz = keep.sum()
        if not keep.all():
            kept_logits = new_logits_raw[b, : c.numel()][keep]
            new_logits_raw[b, :new_sz] = kept_logits

        if new_sz < new_tsz:
            pad = new_tsz - new_sz
            new_logits_raw[b, -pad:] = 0

        new_enc_sizes.raw_tensor[b] = new_sz

    new_enc_spatial_dim = Dim(new_enc_sizes)

    new_logits = rf.Tensor(
        "collapsed_logits",
        dims=[batch_dim, new_enc_spatial_dim, vocab_dim],
        raw_tensor=new_logits_raw,
        dtype=logits.dtype,
    )
    new_logits.feature_dim = vocab_dim

    return new_logits, new_enc_spatial_dim


class KMeansModel(rf.Module):
    """
    Simple K-means model storing centroids and providing fast nearest-centroid lookups.
    """

    def __init__(
        self,
        *,
        num_clusters: int,
        feature_dim: Dim,
        init_centroids: Optional[torch.Tensor] = None,
        normalize_centroids: bool = False, # TODO: Not implemented yet
    ):
        """
        :param num_clusters: number of clusters
        :param feature_dim: Dim representing feature vector size
        :param init_centroids: optional tensor [num_clusters, feature_dim] for initialization
        :param normalize_centroids: whether to keep centroids L2-normalized
        """
        super(KMeansModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.cluster_dim = Dim(name="cluster", dimension=num_clusters)

        if init_centroids is not None:
            self.centroids = rf.Parameter(
                rf.convert_to_tensor(init_centroids, dims=[self.cluster_dim, self.feature_dim]),
                auxiliary=False,
            )
            self.is_initialized = True
        else:
            self.centroids = None
            self.is_initialized = False

        self.normalize_centroids = normalize_centroids

    def update_centroids(self, new_centroids: torch.Tensor):
        """
        Update centroids in-place.
        :param new_centroids: [num_clusters, feature_dim]
        """
        if self.normalize_centroids:
            new_centroids = torch.nn.functional.normalize(new_centroids, dim=1)
            
        self.centroids = new_centroids.clone()

        self.is_initialized = True

    def __call__( # TODO: Include l2_normalize possibility
        self,
        features: rf.Tensor,  # [B, T, F]
    ) -> Tuple[rf.Tensor, Dim]:
        """
        Assign features to nearest centroid.

        :return: distances [B, T, K]
        """
        X = features                      # [B, T, F]
        C = self.centroids                # [K, F]
        B, T, F = X.shape
        K = C.shape[0]

        x2 = (X * X).sum(dim=-1, keepdim=True)       # [B, T, 1]
        c2 = (C * C).sum(dim=-1).view(1, 1, K)       # [1, 1, K]
        prod = (X.reshape(B * T, F) @ C.t()).reshape(B, T, K)  # [B, T, K]

        dist2 = x2 + c2 - 2.0 * prod                 # [B, T, K]
        dist2.clamp_min_(0.0)                        # numerical safety
        dists = dist2.sqrt()

        return dists

    def get_centroid(self, idx: int) -> torch.Tensor:
        """
        Return centroid vector for given index.
        """
        return self.centroids.raw_tensor[idx]


class Wav2VecKMeansModel(rf.Module):
    """
    A wrapper model that combines Wav2VecModel for feature extraction
    and KMeansModel for clustering the extracted features.
    """

    def __init__(
        self,
        *,
        kmeans_scale: float = 1,
        language_model_scale: float = 1,
        w2v_opts: Dict[str, Any],
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        wb_target_dim: Optional[Dim] = None,
        train_language_model: Optional[FeedForwardLm] = None,
        recog_language_model: Optional[FeedForwardLm] = None,
        rescore_language_model: Optional[FeedForwardLm] = None,
        num_clusters: int,
        init_centroids: Optional[torch.Tensor] = None,
        normalize_centroids: bool = False,
        seed: int = 42,
    ):
        super(Wav2VecKMeansModel, self).__init__()
        self.kmeans_scale = kmeans_scale
        self.language_model_scale = language_model_scale
        self.rng = random.Random(seed)

        if not wb_target_dim:
            wb_target_dim = target_dim + 1

        if target_dim.vocab and not wb_target_dim.vocab:
            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
            )
            self.wb_target_dim = wb_target_dim

        assert w2v_opts["output_raw_features"], "Wav2Vec model must output raw features."

        self.wav2vec_model = Wav2VecModel(
            w2v_opts=w2v_opts,
            target_dim=target_dim,
            wb_target_dim=wb_target_dim,
            blank_idx=blank_idx,
            eos_idx=eos_idx,
            bos_idx=bos_idx,
            train_language_model=train_language_model,
            recog_language_model=recog_language_model,
            rescore_language_model=rescore_language_model,
        )

        feature_dim_for_kmeans = self.wav2vec_model.enc_out_dim if w2v_opts.get("pca_dim", None) is None else Dim(name="pca_out", dimension=w2v_opts["pca_dim"], kind=Dim.Types.Feature)

        self.kmeans_model = KMeansModel(
            num_clusters=num_clusters,
            feature_dim=feature_dim_for_kmeans,
            init_centroids=init_centroids,
            normalize_centroids=normalize_centroids,
        )

        self.n_labels_wb = target_dim.vocab.num_labels + 1

        # Initialize mapping creates a list that relates cluster ids to label ids.
        # self.mapping is a list of lenght num_clusters, cluster n translates to label id on position n
        self.mapping = self._init_mapping()

    def _init_mapping(self):
        print(f"Initializing KMeansModel with {self.kmeans_model.cluster_dim.dimension}, {self.n_labels_wb} labels.")
        assert (
            self.kmeans_model.cluster_dim.dimension > self.n_labels_wb
        ), "Number of clusters must be greater than number of labels."

        mapping = []
        counts = [0] * self.n_labels_wb  # how many times each label has been used

        while len(mapping) < self.kmeans_model.cluster_dim.dimension:
            # minimum number of times used so far
            min_count = min(counts)

            # candidates are those that have the minimum count
            candidates = [i for i, c in enumerate(counts) if c == min_count]

            # pick randomly among candidates
            choice = self.rng.choice(candidates)
            mapping.append(choice)
            counts[choice] += 1

        return mapping

    def __call__(
        self,
        source: rf.Tensor,
        *,
        in_spatial_dim: Dim,
    ) -> Tuple[rf.Tensor, Dim]:
        """
        Performs the forward pass: feature extraction followed by clustering.
        """
        features, _, out_spatial_dim = self.wav2vec_model(source=source, in_spatial_dim=in_spatial_dim)

        distances = self.kmeans_model(features=features)

        return distances
