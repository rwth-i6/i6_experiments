# no decoder
# att_weights_inspection=False, inspection_idx=None
# window_limit_idx=None, window_size=None (length same as the num. of heads)

import copy

import i6_core.returnn as returnn

# encs_args: relative_pe=False, fixed=False, clipping=None, left_attention_only=False
# emb_dropout
class attention_for_hybrid:
    def __init__(
        self,
        target,
        num_classes,
        num_enc_layers,
        enc_args,
        type="transformer",
        second_encoder=False,
        num_sec_enc_layers=None,
        variant="same",
        share_par=True,
        normalized_loss=False,
        label_smoothing=0.0,
        focal_loss_factor=0.0,
        softmax_dropout=0.0,
        use_spec_augment=True,
        use_pos_encoding=False,
        add_to_input=True,
        src_embed_args=None,
        add_blstm_block=False,
        blstm_args=None,
        blstm_pooling_args=None,
        add_conv_block=False,
        conv_args=None,
        feature_stacking=False,
        feature_stacking_before_frontend=False,
        feature_stacking_window=None,
        feature_stacking_stride=None,
        reduction_factor=None,
        alignment_reduction=False,
        transposed_conv=False,
        transposed_conv_args=None,
        frame_repetition=False,
        loss_layer_idx=None,
        loss_scale=0.3,
        aux_loss_mlp_dim=256,
        mlp=False,
        mlp_dim=256,
        feature_repre_idx=None,
        att_weights_inspection=False,
        inspection_idx=None,
        window_limit_idx=None,
        window_size=None,
        gauss_window=False,
        was_idx=None,
        upsilon=0.5,
    ):

        assert type in ["transformer", "conformer"]

        # TODO: attention window left and right

        if type == "transformer":
            enc_args.pop("kernel_size", None)
            enc_args.pop("conv_post_dropout", None)
            enc_args.pop("normal_conv", None)
            enc_args.pop("output_channels", None)
            enc_args.pop("kernel_size_for_feature", None)

        if att_weights_inspection:
            assert isinstance(
                inspection_idx, list
            ), "please give SA layer indices to allow inspecting the attention weights."

        if window_limit_idx:
            if isinstance(window_limit_idx, int):
                window_limit_idx = [window_limit_idx]
            for idx in window_limit_idx:
                assert idx in inspection_idx, f"please include {idx} in the inspection list."
            assert len(window_size) == enc_args["num_heads"]

        if was_idx:
            if isinstance(was_idx, int):
                was_idx = [was_idx]
            for idx in was_idx:
                assert idx in inspection_idx, f"please include {idx} in the inspection list."

        if feature_repre_idx:
            if isinstance(feature_repre_idx, int):
                feature_repre_idx = [feature_repre_idx]
            for idx in feature_repre_idx:
                assert idx in inspection_idx, f"please include {idx} in the inspection list."

        if loss_layer_idx:
            if isinstance(loss_layer_idx, int):
                loss_layer_idx = [loss_layer_idx]
        if add_blstm_block and add_conv_block:
            assert not second_encoder

        if second_encoder:
            assert variant in [
                "same",
                "different",
            ], "please choose one from 'same' and 'different'."
            assert isinstance(num_sec_enc_layers, int), "please give num. of layers for the second encoder."

        if use_pos_encoding:
            if not src_embed_args:
                src_embed_args = dict(
                    dim=enc_args["model_dim"],
                    init=enc_args["initialization"],
                    with_bias=True,
                    weight=enc_args["model_dim"] ** 0.5,
                    dropout=0.0,
                )

        self.type = type
        self.was_idx = was_idx
        self.upsilon = upsilon

        self.window_limit_idx = window_limit_idx
        self.window_size = window_size
        self.gauss_window = gauss_window

        self.second_encoder = second_encoder
        self.variant = variant

        if not second_encoder or variant != "same":
            share_par = False
        else:
            assert share_par in [True, False]
        self.share_par = share_par

        self.loss_layer_idx = loss_layer_idx
        self.loss_scale = loss_scale
        self.aux_loss_mlp_dim = aux_loss_mlp_dim

        self.mlp = mlp
        self.mlp_dim = mlp_dim

        self.feature_repre_idx = feature_repre_idx

        self.add_to_input = add_to_input
        self.use_pos_encoding = use_pos_encoding

        self.att_weights_inspection = att_weights_inspection
        self.inspection_idx = inspection_idx

        ## encoder arguments
        self.num_enc_layers = num_enc_layers
        self.emb_dropout = enc_args.pop("emb_dropout", 0.0)
        self.enc_args = enc_args

        ## encoder source embedding
        self.src_embed_args = src_embed_args

        self.normalized_loss = normalized_loss
        self.label_smoothing = label_smoothing
        self.focal_loss_factor = focal_loss_factor
        self.softmax_dropout = softmax_dropout

        self.target = target
        self.num_classes = num_classes

        self.use_spec_augment = use_spec_augment

        self.add_blstm_block = add_blstm_block
        self.num_blstm_layers = len(blstm_args["dims"]) if blstm_args and "dims" in blstm_args.keys() else 2
        self.blstm_args = blstm_args
        self.blstm_pooing_args = blstm_pooling_args

        self.add_conv_block = add_conv_block
        self.conv_args = conv_args

        if feature_stacking:
            assert len(feature_stacking_window) == 2
            assert isinstance(feature_stacking_window[0], int)
            assert isinstance(feature_stacking_window[1], int)
            if not feature_stacking_stride:
                feature_stacking_stride = 1

        if reduction_factor:
            assert len(reduction_factor) == 2
            assert isinstance(reduction_factor[0], int)
            assert isinstance(reduction_factor[1], int)

        if (feature_stacking and feature_stacking_stride >= 2) or (
            reduction_factor and reduction_factor[0] * reduction_factor[1] >= 2
        ):
            assert alignment_reduction or transposed_conv or frame_repetition
            assert (alignment_reduction + transposed_conv + frame_repetition) == 1
        else:
            alignment_reduction = transposed_conv = frame_repetition = False

        if transposed_conv and not transposed_conv_args:
            transposed_conv_args = {}
        self.feature_stacking = feature_stacking
        self.feature_stacking_window = feature_stacking_window
        self.feature_stacking_stride = feature_stacking_stride
        self.feature_stacking_before_frontend = feature_stacking_before_frontend

        self.transposed_conv = transposed_conv
        self.transposed_conv_args = transposed_conv_args
        self.frame_repetition = frame_repetition
        self.alignment_reduction = alignment_reduction

        self.reduction_factor = reduction_factor

        self.network = dict()

    # two types:
    # 1. pe vector added to the weighted input embedding vector
    # 2. pe vector concat. to the input embedding vector then the concatenated vector
    #    is linearly transformed to the desired dimension
    def _positional_encoding(self, inp=None, add_to_input=True, prefix=""):
        if prefix:
            prefix = prefix + "_"

        ## source embedding layer
        self.network[f"{prefix}source_embed_raw"] = {
            "class": "linear",
            "activation": None,
            "with_bias": self.src_embed_args.get("with_bias", False),
            "n_out": self.src_embed_args["dim"],
            "forward_weights_init": self.src_embed_args["init"],
        }
        if inp is not None:
            if isinstance(inp, str):
                self.network[f"{prefix}source_embed_raw"]["from"] = [inp]
            elif isinstance(inp, list):
                self.network[f"{prefix}source_embed_raw"]["from"] = inp
            else:
                raise TypeError

        # for additive positional embedding we want to get a weighted sum of
        ## the source embedding vector and the positional embedding vector
        ## so we weight the source embedding first
        self.network[f"{prefix}source_embed_weighted"] = {
            "class": "eval",
            "from": [f"{prefix}source_embed_raw"],
            "eval": f"source(0) * {self.src_embed_args['weight']}",
        }
        if add_to_input:
            self.network[f"{prefix}source_embed_with_pos"] = {
                "class": "positional_encoding",
                "add_to_input": True,
                "from": [f"{prefix}source_embed_weighted"],
            }
        else:
            self.network[f"{prefix}pos"] = {
                "class": "positional_encoding",
                "add_to_input": False,
                "from": [f"{prefix}source_embed_raw"],
                "n_out": self.src_embed_args["dim"],
            }
            self.network[f"{prefix}source_embed_with_pos"] = {
                "class": "linear",
                "activation": None,
                "with_bias": self.src_embed_args.get("with_bias", False),
                "n_out": self.src_embed_args["dim"],
                "forward_weights_init": self.src_embed_args["init"],
                "from": [f"{prefix}source_embed_weighted", f"{prefix}pos"],
            }

        self.network[f"{prefix}source_embed"] = {
            "class": "dropout",
            "dropout": self.src_embed_args["dropout"],
            "from": [f"{prefix}source_embed_with_pos"],
        }
        return [f"{prefix}source_embed"]

    def _add_embedding(self, inp=None, prefix="", with_bias=True):

        if prefix:
            prefix = prefix + "_"

        ## linear transformation
        self.network[f"{prefix}embedding"] = {
            "class": "linear",
            "activation": None,
            "with_bias": with_bias,
            "forward_weights_init": self.enc_args["initialization"],
            "n_out": self.enc_args["model_dim"],
        }

        self.network[f"{prefix}embedding_dropout"] = {
            "class": "dropout",
            "dropout": self.emb_dropout,
            "from": [f"{prefix}embedding"],
        }

        if inp is not None:
            if isinstance(inp, str):
                self.network[f"{prefix}embedding"]["from"] = [inp]
            elif isinstance(inp, list):
                self.network[f"{prefix}embedding"]["from"] = inp
            else:
                raise TypeError

        return [f"{prefix}embedding_dropout"]

    # adjustable archtecture
    # number of blstm layers, hidden units size, l2-regularization, dropout
    def _blstm_block(self, inp=None, prefix=""):
        from . import layers

        if prefix:
            prefix = prefix + "_"

        last_layers = None
        if inp is not None:
            if isinstance(inp, str):
                last_layers = [inp]
            elif isinstance(inp, list):
                last_layers = inp
            else:
                raise TypeError

        # local variables for pooling
        pooling_layers = []
        if self.blstm_pooing_args:
            pooling_layers = self.blstm_pooing_args.get("pooling_layers", [])

        # first blstm layer
        idx = 0
        layers.add_blstm_layer(
            self.network,
            idx=idx,
            dim=self.blstm_args["dims"][idx],
            dropout=self.blstm_args.get("dropout", 0.0),
            l2=self.blstm_args.get("l2", 0.0),
            from_layers=last_layers,
            prefix=prefix,
        )
        last_layers = [f"{prefix}lstm_{idx}_fwd", f"{prefix}lstm_{idx}_bwd"]

        if idx in pooling_layers:
            if self.reduction_factor and self.reduction_factor[0] >= 2:
                layers.add_pool_layer(
                    self.network,
                    idx=idx,
                    mode="max",
                    padding="same",
                    pool_size=(self.reduction_factor[0],),
                    from_layers=last_layers,
                    trainable=False,
                )
                last_layers = [f"{prefix}lstm_pool_{idx}"]

        if self.num_blstm_layers >= 2:
            for idx in range(1, self.num_blstm_layers):
                layers.add_blstm_layer(
                    self.network,
                    idx=idx,
                    dim=self.blstm_args["dims"][idx],
                    dropout=self.blstm_args.get("dropout", 0.0),
                    l2=self.blstm_args.get("l2", 0.0),
                    from_layers=last_layers,
                    prefix=prefix,
                )

                last_layers = [f"{prefix}lstm_{idx}_fwd", f"{prefix}lstm_{idx}_bwd"]

                if (idx + 1) in pooling_layers:
                    layers.add_pool_layer(
                        self.network,
                        idx=idx,
                        mode="max",
                        padding="same",
                        pool_size=(2,),
                        from_layers=last_layers,
                        trainable=False,
                    )

                    last_layers = [f"{prefix}lstm_pool_{idx}"]

        return last_layers

    def _conv_block(self, inp=None, prefix=""):

        # input shape (batch, time, feature)

        self.network["source0"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (-1, 1),
        }  # (T,50,1)

        if inp is not None:
            if isinstance(inp, str):
                self.network["source0"]["from"] = [inp]
            elif isinstance(inp, list):
                self.network["source0"]["from"] = inp
            else:
                raise TypeError

        if prefix:
            prefix = prefix + "_"

        ## first vgg block
        self.network[f"{prefix}conv0_0"] = {
            "class": "conv",
            "from": "source0",
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32,
            "activation": None,
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:50"],
        }  # (T,50,32)
        # , "in_spatial_dims": ["T", "dim:1"]
        # , "in_spacial_dim": ["dim:1"]

        self.network[f"{prefix}conv0_1"] = {
            "class": "conv",
            "from": f"{prefix}conv0_0",
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32,
            "activation": "relu",
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:50"],
        }  # (T,50,32)

        self.network[f"{prefix}conv0p"] = {
            "class": "pool",
            "mode": "max",
            "padding": "same",
            "pool_size": (1, 2),
            "strides": (1, 2),
            "from": f"{prefix}conv0_1",
            "in_spatial_dims": ["T", "dim:50"],
        }  # (T,25,32)

        if self.reduction_factor and self.reduction_factor[0] >= 2:
            self.network[f"{prefix}conv0p"]["strides"] = (self.reduction_factor[0], 2)
            self.network[f"{prefix}conv0p"]["pool_size"] = (self.reduction_factor[0], 2)

        self.network[f"{prefix}conv1_0"] = {
            "class": "conv",
            "from": f"{prefix}conv0p",
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 64,
            "activation": None,
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"],
        }  # (T,25,64)
        self.network[f"{prefix}conv1_1"] = {
            "class": "conv",
            "from": f"{prefix}conv1_0",
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 64,
            "activation": "relu",
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"],
        }  # (T,25,64)
        self.network[f"{prefix}conv1p"] = {
            "class": "pool",
            "mode": "max",
            "padding": "same",
            "pool_size": (1, 1),
            "strides": (1, 1),  # strides': (1, 1)?
            "from": f"{prefix}conv1_1",
            "in_spatial_dims": ["T", "dim:25"],
        }  # (T,25,64)

        if self.reduction_factor and self.reduction_factor[1] >= 2:
            self.network[f"{prefix}conv1p"]["strides"] = (self.reduction_factor[1], 1)
            self.network[f"{prefix}conv1p"]["pool_size"] = (self.reduction_factor[1], 1)

        ## features from different channels are merged together stag:conv0p:conv:s1
        self.network[f"{prefix}conv_merged"] = {
            "class": "merge_dims",
            "from": f"{prefix}conv1p",
            "axes": ["stag:conv0p:conv:s1", "dim:64"],
        }  # (T,1600)

        if self.conv_args:
            for name in [
                "conv0_0",
                "conv0_1",
                "conv0p",
                "conv1_0",
                "conv1_1",
                "conv1p",
            ]:
                if self.conv_args.get(name, None):
                    self.network[name].update(self.conv_args.pop(name))
            ## for additional layers
            ## in this case the network interlinks should also be properly set
            self.network.update(self.conv_args)

        return [f"{prefix}conv_merged"]

    # to be implemented: feature representation
    def _transformer_encoder(self, inp):
        from . import layers

        separated = False
        windowing = False

        if (
            self.att_weights_inspection and 1 in self.inspection_idx
        ):  # layer 1 is inspected -> separate attention layer definition
            # with additional windowing
            separated = True
            if self.window_limit_idx and 1 in self.window_limit_idx:
                windowing = True
        layers.trafo_enc_layer_all_in_one(
            self.network,
            "enc_001",
            from_layers=inp,
            separated=separated,
            windowing=windowing,
            window_size=self.window_size,
            gauss_window=self.gauss_window,
            **self.enc_args,
        )

        if self.was_idx and 1 in self.was_idx:
            self._add_was(idx=1)

        for n in range(2, self.num_enc_layers + 1):
            prev_n = n - 1
            last_layer = f"enc_{prev_n:03d}"

            separated = False
            windowing = False
            ## eventually with feature representation
            if self.feature_repre_idx and n in self.feature_repre_idx:
                self._add_feature_representation(n, inp)
                last_layer = f"feature_repre{n}"

            if self.att_weights_inspection and n in self.inspection_idx:
                separated = True
                if self.window_limit_idx and n in self.window_limit_idx:
                    windowing = True
            layers.trafo_enc_layer_all_in_one(
                self.network,
                f"enc_{n:03d}",
                from_layers=last_layer,
                separated=separated,
                windowing=windowing,
                window_size=self.window_size,
                gauss_window=self.gauss_window,
                **self.enc_args,
            )

            if self.was_idx and n in self.was_idx:
                self._add_was(idx=n)

            # add intermediate auxillary loss of scale 0.3
            ## the two mlp layers are meant to reduce overfitting
            if self.loss_layer_idx and n in self.loss_layer_idx:
                self.add_auxiliary_loss(idx=n)

        if self.enc_args.get("end_layernorm", False):
            ## final layernorm layer
            self.network["encoder"] = {
                "class": "copy",
                "from": [f"enc_{self.num_enc_layers:03d}"],
            }
        else:
            self.network["encoder"] = {
                "class": "layer_norm",
                "from": [f"enc_{self.num_enc_layers:03d}"],
            }

        return ["encoder"]

    def _conformer_encoder(self, inp):
        from . import layers

        separated = False
        windowing = False

        if (
            self.att_weights_inspection and 1 in self.inspection_idx
        ):  # layer 1 is inspected -> separate attention layer definition
            # with additional windowing
            separated = True
            if self.window_limit_idx and 1 in self.window_limit_idx:
                windowing = True
        layers.conformer_enc_layer_all_in_one(
            self.network,
            "enc_001",
            from_layers=inp,
            separated=separated,
            windowing=windowing,
            window_size=self.window_size,
            gauss_window=self.gauss_window,
            **self.enc_args,
        )

        if self.was_idx and 1 in self.was_idx:
            self._add_was(idx=1)

        for n in range(2, self.num_enc_layers + 1):
            prev_n = n - 1
            last_layer = f"enc_{prev_n:03d}"

            separated = False
            windowing = False
            ## eventually with feature representation
            # if self.feature_repre_idx and n in self.feature_repre_idx:
            #   self._add_feature_representation(n, inp)
            #   last_layer = f"feature_repre{n}"

            if self.att_weights_inspection and n in self.inspection_idx:
                separated = True
                if self.window_limit_idx and n in self.window_limit_idx:
                    windowing = True
            layers.conformer_enc_layer_all_in_one(
                self.network,
                f"enc_{n:03d}",
                from_layers=last_layer,
                separated=separated,
                windowing=windowing,
                window_size=self.window_size,
                gauss_window=self.gauss_window,
                **self.enc_args,
            )

            if self.was_idx and n in self.was_idx:
                self._add_was(idx=n)

            if self.feature_repre_idx and n in self.feature_repre_idx:
                self._add_feature_representation(n, inp)

            # add intermediate auxillary loss of scale 0.3
            ## the two ff layers are meant to reduce overfitting
            if self.loss_layer_idx and n in self.loss_layer_idx:
                self.add_auxiliary_loss(idx=n)

        ## final layernorm layer
        if self.enc_args.get("end_layernorm", False):
            ## final layernorm layer
            self.network["encoder"] = {
                "class": "copy",
                "from": [f"enc_{self.num_enc_layers:03d}"],
            }
        else:
            self.network["encoder"] = {
                "class": "layer_norm",
                "from": [f"enc_{self.num_enc_layers:03d}"],
            }

        return ["encoder"]

    ## another kind of feature representation
    ## attention weights manipulation not implemented
    def _second_encoder(self, inp, ca_layer):
        from . import layers

        self.enc_args.pop("attention_left_only", None)

        layers.separated_trafo_ca_layer(
            self.network,
            "sec_enc_001",
            from_layers=inp,
            ca_layer=ca_layer,
            **self.enc_args,
        )

        for n in range(2, self.num_sec_enc_layers + 1):
            prev_n = n - 1

            layers.separated_trafo_ca_layer(
                self.network,
                f"sec_enc_{n:03d}",
                from_layers=f"sec_enc_{prev_n:03d}",
                ca_layer=ca_layer,
                **self.enc_args,
            )

        self.network["sec_encoder"] = {
            "class": "copy",
            "from": [f"sec_enc_{self.num_sec_enc_layers:03d}"],
        }

        return ["sec_encoder"]

    def _add_feature_representation(self, idx, from_layers):
        # type = 'concat_f'
        # type = 'add'
        type = "concat_t"
        if from_layers is None:
            from_layers = ["data"]
        elif isinstance(from_layers, str):
            from_layers = [from_layers]

        if type == "add":
            self.network[f"enc_{idx:03d}_feature_repre"] = {
                "class": "combine",
                "kind": "add",
                "from": self.network[f"enc_{idx:03d}_att_value0"]["from"] + from_layers,
            }
        if type == "concat_f":
            self.network[f"enc_{idx:03d}_feature_repre"] = {
                "class": "copy",
                "from": self.network[f"enc_{idx:03d}_att_value0"]["from"] + from_layers,
            }
        # concat along T
        else:
            assert len(self.network[f"enc_{idx:03d}_att_value0"]["from"]) == 1
            self.network[f"enc_{idx:03d}_feature_repre"] = {
                "class": "eval",
                "from": self.network[f"enc_{idx:03d}_att_value0"]["from"] + from_layers,
                "eval": "tf.concat([source(0), source(1)], 1)",
            }
            self.network[f"enc_{idx:03d}_att_weights"]["use_time_mask"] = False

        self.network[f"enc_{idx:03d}_att_value0"]["from"] = [f"enc_{idx:03d}_feature_repre"]
        self.network[f"enc_{idx:03d}_att_key0"]["from"] = [f"enc_{idx:03d}_feature_repre"]

    def _add_was(self, idx):
        ## if was should be applied
        ## use two softmax layers to do weak attention suppression
        self.network[f"enc_{idx:03d}_att_weights_aux"] = copy.deepcopy(self.network[f"enc_{idx:03d}_att_weights"])

        self.network[f"enc_{idx:03d}_att_energy_was"] = {
            "class": "eval",
            "from": [f"enc_{idx:03d}_att_energy", f"enc_{idx:03d}_att_weights_aux"],
            "eval": f"self.network.get_config().typed_value('att_weight_suppression')(source(0), source(1), upsilon={self.upsilon})",
        }

        self.network[f"enc_{idx:03d}_att_weights"]["from"] = [f"enc_{idx:03d}_att_energy_was"]

    def add_auxiliary_loss(self, idx):
        last_layers = [f"enc_{idx:03d}"]

        if self.transposed_conv:
            last_layers = self._upsampling_by_transposed_conv(last_layers, prefix=f"aux_{idx}")
            if "upsampled0" in self.network:
                self.network[f"aux_{idx}_upsampled0"]["reuse_params"] = "upsampled0"
            if "upsampled1" in self.network:
                self.network[f"aux_{idx}_upsampled1"]["reuse_params"] = "upsampled1"
            if "upsampled2" in self.network:
                self.network[f"aux_{idx}_upsampled2"]["reuse_params"] = "upsampled2"

        if self.frame_repetition:
            last_layers = self._upsampling_by_frame_repetition(last_layers, prefix=f"aux_{idx}")

        if self.aux_loss_mlp_dim:
            self.network[f"aux_{idx}_ff1"] = {
                "class": "linear",
                "activation": "relu",
                "with_bias": True,
                "from": last_layers,
                "n_out": self.aux_loss_mlp_dim,
                "forward_weights_init": self.enc_args["initialization"],
            }

            last_layers = [f"aux_{idx}_ff1"]
            self.network[f"aux_{idx}_ff2"] = {
                "class": "linear",
                "activation": None,
                "with_bias": True,
                "from": last_layers,
                "n_out": self.aux_loss_mlp_dim,
                "forward_weights_init": self.enc_args["initialization"],
            }
            last_layers = [f"aux_{idx}_ff2"]

        self.network[f"aux_{idx}_output_prob"] = {
            "class": "softmax",
            "dropout": self.softmax_dropout,
            "from": last_layers,
            "loss": "ce",
            "loss_opts": {
                "label_smoothing": self.label_smoothing,
                "use_normalized_loss": self.normalized_loss,
                "focal_loss_factor": self.focal_loss_factor,
            },
            "loss_scale": self.loss_scale,
            "target": self.target,
        }
        if self.alignment_reduction:
            self.network[f"aux_output_prob_{idx}"]["target"] = "reduced_classes"

    def _feature_stacking(self, from_layers):
        if from_layers is None:
            from_layers = ["data"]
        elif isinstance(from_layers, str):
            from_layers = [from_layers]

        # [B, T, F] -> [B, T/stride, window_size, F]
        self.network["feature_stacking_window"] = {
            "class": "window",
            "window_size": self.feature_stacking_window[0]
            + self.feature_stacking_window[1]
            + 1,  # window_size == window_left + window_right + 1
            "window_right": self.feature_stacking_window[1],
            "window_left": self.feature_stacking_window[0],
            "stride": self.feature_stacking_stride,
            "from": from_layers,
        }
        self.network["feature_stacking_merged"] = {
            "class": "merge_dims",
            "axes": (2, 3),
            "from": ["feature_stacking_window"],
        }

        return ["feature_stacking_merged"]

    def _upsampling_by_transposed_conv(self, from_layers, prefix=None):
        if not prefix:
            prefix = ""
        else:
            prefix = prefix + "_"

        if from_layers is None:
            from_layers = ["data"]
        elif isinstance(from_layers, str):
            from_layers = [from_layers]

        last_layers = from_layers

        if self.feature_stacking and self.feature_stacking_stride >= 2:
            self.network[f"{prefix}upsampled0"] = {
                "class": "transposed_conv",
                "filter_size": (self.transposed_conv_args.get("filter_size0", 3),),
                "activation": "relu",
                "strides": (self.feature_stacking_stride,),
                "with_bias": True,
                "n_out": self.enc_args["model_dim"] or self.transposed_conv_args.get("model_dim0", None),
                "from": last_layers,
            }
            last_layers = [f"{prefix}upsampled0"]

        if self.reduction_factor:
            if self.reduction_factor[0] >= 2:
                self.network[f"{prefix}upsampled1"] = {
                    "class": "transposed_conv",
                    "filter_size": (self.transposed_conv_args.get("filter_size1", 3),),
                    "activation": "relu",
                    "strides": (self.reduction_factor[0],),
                    "with_bias": True,
                    "n_out": self.enc_args["model_dim"] or self.transposed_conv_args.get("model_dim1", None),
                    "from": last_layers,
                }
                last_layers = [f"{prefix}upsampled1"]

            if self.reduction_factor[1] >= 2:
                self.network[f"{prefix}upsampled2"] = {
                    "class": "transposed_conv",
                    "filter_size": (self.transposed_conv_args.get("filter_size2", 3),),
                    "activation": "relu",
                    "strides": (self.reduction_factor[1],),
                    "with_bias": True,
                    "n_out": self.enc_args["model_dim"] or self.transposed_conv_args.get("model_dim2", None),
                    "from": last_layers,
                }
                last_layers = [f"{prefix}upsampled2"]

        time_tag_name = self.transposed_conv_args.get("time_tag_name")
        if time_tag_name is not None:
            self.network[f"{prefix}length_masked"] = {
                "class": "slice_nd",
                "from": last_layers,
                "start": 0,
                "size": returnn.CodeWrapper(time_tag_name),
                "axis": "T",
            }
        else:
            self.network[f"{prefix}length_masked"] = {
                "class": "reinterpret_data",
                "from": last_layers,
                "size_base": "data:classes",
            }
        return [f"{prefix}length_masked"]

    def _upsampling_by_frame_repetition(self, from_layers, prefix=None):

        total_ratio = 1
        if self.feature_stacking:
            total_ratio = total_ratio * self.feature_stacking_stride
        if self.reduction_factor:
            total_ratio = total_ratio * self.reduction_factor[0] * self.reduction_factor[1]

        # check if total_ratio is power of 2
        assert (total_ratio & (total_ratio - 1) == 0) and total_ratio != 0

        if not prefix:
            prefix = ""
        else:
            prefix = prefix + "_"

        if from_layers is None:
            from_layers = ["data"]
        elif isinstance(from_layers, str):
            from_layers = [from_layers]

        last_layers = from_layers
        for i in range(1, total_ratio.bit_length()):
            self.network[f"{prefix}repetition_window_{i}"] = {
                "class": "window",
                "window_size": 2,
                "window_left": 1,
                "stride": 1,
                "from": last_layers,
            }
            last_layers = [f"{prefix}repetition_window_{i}"]
            self.network[f"{prefix}repetition_merged_{i}"] = {
                "class": "merge_dims",
                "axes": (1, 2),  # merge over time and window axes
                "from": last_layers,
            }
            last_layers = [f"{prefix}repetition_merged_{i}"]

        # length masking layer
        self.network[f"{prefix}length_masked"] = {
            "class": "reinterpret_data",
            "from": last_layers,
            "size_base": "data:classes",
        }
        return [f"{prefix}length_masked"]

    def _reduce_alignment(self):

        ## (B, T) -> (B, T, 1)
        self.network["expand_classes"] = {
            "class": "eval",
            "from": "data:classes",
            "eval": "tf.expand_dims(source(0, auto_convert=False), axis=-1)",
            "out_type": {"dim": 1, "shape": (None, 1), "sparse": False},
        }
        ## for convolution operation
        self.network["type_casting"] = {
            "class": "cast",
            "from": f"expand_classes",
            "dtype": "float32",
        }

        ## label downsampling with fixed conv. kernel [1, 0,..,0]
        last_pool_layer = "type_casting"

        if self.feature_stacking and self.feature_stacking_stride >= 2:
            self.network["classes_pool0"] = {
                "class": "conv",
                "from": last_pool_layer,
                "padding": "same",
                "filter_size": (self.feature_stacking_stride,),
                "strides": (self.feature_stacking_stride,),  # (T, 1)
                "n_out": 1,
                "activation": None,
                "with_bias": False,
                "forward_weights_init": f'numpy.reshape(numpy.array([1] + [0] * {self.feature_stacking_stride-1}, dtype="float32"), '
                f"newshape=({self.feature_stacking_stride}, 1, 1))",
                "trainable": False,
            }
            last_pool_layer = "classes_pool0"

        if self.reduction_factor:
            if self.reduction_factor[0] >= 2:
                self.network[f"classes_pool1"] = {
                    "class": "conv",
                    "from": last_pool_layer,
                    "padding": "same",
                    "filter_size": (self.reduction_factor[0],),
                    "strides": (self.reduction_factor[0],),  # (T, 1)
                    "n_out": 1,
                    "activation": None,
                    "with_bias": False,
                    "forward_weights_init": f'numpy.reshape(numpy.array([1] + [0] * {self.reduction_factor[0]-1}, dtype="float32"), '
                    f"newshape=({self.reduction_factor[0]}, 1, 1))",
                    "trainable": False,
                }
                last_pool_layer = "classes_pool1"
            if self.reduction_factor[1] >= 2:
                self.network["classes_pool2"] = {
                    "class": "conv",
                    "from": last_pool_layer,
                    "padding": "same",
                    "filter_size": (self.reduction_factor[1],),
                    "strides": (self.reduction_factor[1],),  # (T, 1)
                    "n_out": 1,
                    "activation": None,
                    "with_bias": False,
                    "forward_weights_init": f'numpy.reshape(numpy.array([1] + [0] * {self.reduction_factor[1]-1}, dtype="float32"), '
                    f"newshape=({self.reduction_factor[1]}, 1, 1))",
                    "trainable": False,
                }
                last_pool_layer = "classes_pool2"

        self.network["type_casting_back"] = {
            "class": "cast",
            "from": last_pool_layer,
            "dtype": "int32",
        }
        ## squeeze and register the reduced class
        self.network["squeeze_classes"] = {
            "class": "eval",
            "from": "type_casting_back",
            "register_as_extern_data": "reduced_classes",
            "eval": "tf.squeeze(source(0, auto_convert=False), axis=[-1])",
            "out_type": {"dim": self.num_classes, "shape": (None,), "sparse": True},
        }

        # self.network['output']["target"] = "reduced_classes"
        return f"reduced_classes"

    def _build(self):

        last_layer = sec_last_layer = None
        # default 'from' layer: 'data'
        if self.use_spec_augment:
            self.network["source"] = {
                "class": "eval",
                "eval": "self.network.get_config().typed_value('transform')(source(0), network=self.network)",
                "from": "data"
                # "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
                # "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network, clip=True)",
            }
            last_layer = sec_last_layer = ["source"]

        if self.feature_stacking and self.feature_stacking_before_frontend:
            last_layer = self._feature_stacking(from_layers=last_layer)

        # pe by sinusoidal functions
        if self.use_pos_encoding:
            last_layer = sec_last_layer = self._positional_encoding(inp=last_layer, add_to_input=self.add_to_input)

        # add both blstm_block and vgg blocks
        if self.add_blstm_block and self.add_conv_block:

            if self.conv_args and self.conv_args.get("first", False):
                if self.reduction_factor:
                    self.blstm_pooing_args = None
                last_layer = self._conv_block(inp=last_layer)
                ## an embedding layer in between
                last_layer = self._add_embedding(inp=last_layer, prefix="conv_blstm")
                last_layer = self._blstm_block(inp=last_layer)
            else:
                last_layer = self._blstm_block(inp=last_layer)
                last_layer = self._add_embedding(inp=last_layer, prefix="blstm_conv")

                if self.reduction_factor:
                    reduction_factor_tmp = self.reduction_factor
                    self.reduction_factor = None

                    last_layer = self._conv_block(inp=last_layer)

                    self.reduction_factor = reduction_factor_tmp

        # not both
        else:
            # blstm block
            if self.add_blstm_block:
                last_layer = self._blstm_block(inp=last_layer)

            elif self.add_conv_block:
                last_layer = self._conv_block(inp=last_layer)

        if self.feature_stacking and not self.feature_stacking_before_frontend:
            last_layer = self._feature_stacking(from_layers=last_layer)

        last_layer = self._add_embedding(inp=last_layer)

        # encoder
        if self.type == "transformer":
            last_layer = self._transformer_encoder(inp=last_layer)
        else:
            last_layer = self._conformer_encoder(inp=last_layer)

        # second encoder
        if self.second_encoder:

            ## variant 1: use different types of feature extraction layers
            if self.variant == "different":
                if self.add_blstm_block and not self.add_conv_block:
                    sec_last_layer = self._conv_block(inp=sec_last_layer, prefix="sec")
                    sec_last_layer = self._add_embedding(inp=sec_last_layer, prefix="sec")

                elif self.add_conv_block and not self.add_blstm_block:
                    sec_last_layer = self._blstm_block(inp=sec_last_layer, prefix="sec")
                    sec_last_layer = self._add_embedding(inp=sec_last_layer, prefix="sec")

            ## variant 2: use the same type of feature extraction layers
            elif self.variant == "same":
                if self.add_blstm_block and not self.add_conv_block:
                    if self.share_par:
                        sec_last_layer = ["embedding_dropout"]
                    else:
                        sec_last_layer = self._blstm_block(inp=sec_last_layer, prefix="sec")
                        sec_last_layer = self._add_embedding(inp=sec_last_layer, prefix="sec")

                elif self.add_conv_block and not self.add_blstm_block:
                    if self.share_par:
                        sec_last_layer = ["embedding_dropout"]
                    else:
                        sec_last_layer = self._conv_block(inp=sec_last_layer, prefix="sec")

                        sec_last_layer = self._add_embedding(inp=sec_last_layer, prefix="sec")

            last_layer = self._second_encoder(inp=sec_last_layer, ca_layer=last_layer[0])

        if self.mlp:
            self.network["ff1"] = {
                "class": "linear",
                "activation": "relu",
                "with_bias": True,
                "from": [last_layer] if isinstance(last_layer, str) else last_layer,
                "n_out": self.mlp_dim,
                "forward_weights_init": self.enc_args["initialization"],
            }

            self.network["ff2"] = {
                "class": "linear",
                "activation": None,
                "with_bias": True,
                "from": ["ff1"],
                "n_out": self.mlp_dim,
                "forward_weights_init": self.enc_args["initialization"],
            }
            last_layer = ["ff2"]

        # TODO: check if down and up are matched
        if self.frame_repetition:
            last_layer = self._upsampling_by_frame_repetition(from_layers=last_layer)
        if self.transposed_conv:
            last_layer = self._upsampling_by_transposed_conv(from_layers=last_layer)

        self.network["output"] = {
            "class": "softmax",
            "dropout": self.softmax_dropout,
            "from": last_layer,
            "n_out": self.num_classes,
            "loss": "ce",
            "loss_opts": {
                "label_smoothing": self.label_smoothing,
                "use_normalized_loss": self.normalized_loss,
                "focal_loss_factor": self.focal_loss_factor,  # 2
            },
            "target": self.target,
        }

        if self.alignment_reduction:
            self.network["output"]["target"] = self._reduce_alignment()

    def get_network(self):
        self._build()
        return self.network
