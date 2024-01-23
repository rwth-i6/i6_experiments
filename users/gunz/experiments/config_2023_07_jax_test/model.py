from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, TYPE_CHECKING

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
    dataclass_json = lambda x: x
else:
    from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(eq=True, frozen=True)
class FeedForwardConfig:
    dropout: float
    hidden_dim: int
    model_dim: int

    activation: str = "swish"


@dataclass_json
@dataclass(eq=True, frozen=True)
class MhsaConfig:
    dropout: float
    model_dim: int
    num_att_heads: int


@dataclass_json
@dataclass(eq=True, frozen=True)
class ConvolutionConfig:
    dropout: float
    kernel_size: int
    model_dim: int

    activation: str = "swish"


@dataclass_json
@dataclass(eq=True, frozen=True)
class ConformerBlockConfig:
    conv: ConvolutionConfig
    ff: FeedForwardConfig
    mhsa: MhsaConfig


Stride2d = Union[int, Tuple[int, int]]


@dataclass_json
@dataclass(eq=True, frozen=True)
class Conformer1dVggConfig:
    data_xdim: int

    conv_kernel_size: Stride2d
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    pool1_kernel_size: Stride2d
    pool1_stride: Stride2d
    pool2_kernel_size: Stride2d
    pool2_stride: Stride2d

    linear_dim_in: int
    linear_dim_out: int

    activation: str = "relu"


@dataclass_json
@dataclass(eq=True, frozen=True)
class ConformerFrontendConfig:
    dropout: float
    max_num_device_types: int
    max_pos: int
    n_in: int
    n_out: int


@dataclass_json
@dataclass(eq=True, frozen=True)
class ConformerConfig:
    block_cfg: ConformerBlockConfig
    frontend_cfg: ConformerFrontendConfig
    num_blocks: int

    @classmethod
    def _config_w_model_dim(
        cls,
        dim: int,
        n_in: int,
        dropout: float,
        num_blocks: int,
        max_num_device_types: int,
        max_seq_length: int,
    ) -> "ConformerConfig":
        assert n_in > 0
        assert dim > 0
        assert num_blocks > 0
        assert max_num_device_types > 0
        assert max_seq_length > 0

        # VGG settings depend on the DFL dims and feature size
        # vgg_config = Conformer1dVggConfig(
        #     conv_kernel_size=3,
        #     conv1_channels=10,
        #     conv2_channels=7,
        #     conv3_channels=5,
        #     conv4_channels=3,
        #     data_xdim=6,
        #     linear_dim_in=468,
        #     linear_dim_out=dim,
        #     pool1_kernel_size=3,
        #     pool1_stride=3,
        #     pool2_kernel_size=3,
        #     pool2_stride=3,
        # )
        frontend_config = ConformerFrontendConfig(
            dropout=dropout,
            max_num_device_types=max_num_device_types,
            max_pos=max_seq_length,
            n_in=n_in,
            n_out=dim,
        )
        conv_config = ConvolutionConfig(
            dropout=dropout,
            kernel_size=3,
            model_dim=dim,
        )
        ff_config = FeedForwardConfig(
            dropout=dropout,
            hidden_dim=4 * dim,
            model_dim=dim,
        )
        mhsa_config = MhsaConfig(
            dropout=dropout,
            model_dim=dim,
            num_att_heads=8,
        )
        block_config = ConformerBlockConfig(
            conv=conv_config,
            ff=ff_config,
            mhsa=mhsa_config,
        )
        config = cls(
            block_cfg=block_config,
            frontend_cfg=frontend_config,
            num_blocks=num_blocks,
        )
        return config

    @classmethod
    def config_d_128(
        cls,
        n_in: int,
        num_devices: int,
        max_seq_length: int,
    ) -> "ConformerConfig":
        return cls._config_w_model_dim(
            128,
            n_in=n_in,
            dropout=0.1,
            num_blocks=8,
            max_num_device_types=num_devices,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def config_d_256(
        cls,
        n_in: int,
        num_devices: int,
        max_seq_length: int,
    ) -> "ConformerConfig":
        return cls._config_w_model_dim(
            256,
            n_in=n_in,
            dropout=0.1,
            num_blocks=12,
            max_num_device_types=num_devices,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def config_d_512(
        cls,
        n_in: int,
        num_devices: int,
        max_seq_length: int,
    ) -> "ConformerConfig":
        return cls._config_w_model_dim(
            512,
            n_in=n_in,
            dropout=0.1,
            num_blocks=12,
            max_num_device_types=num_devices,
            max_seq_length=max_seq_length,
        )


class ConformerConvolutionModule(eqx.Module):
    activation: Callable
    conv_depthwise: nn.Conv1d
    conv_pointwise1: nn.Linear
    conv_pointwise2: nn.Linear
    dropout: nn.Dropout
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    def __init__(self, cfg: ConvolutionConfig, *, key: random.KeyArray):
        k1, k2, k3 = random.split(key, 3)

        self.activation = getattr(jax.nn, cfg.activation)
        self.conv_depthwise = nn.Conv1d(
            cfg.model_dim,
            cfg.model_dim,
            cfg.kernel_size,
            padding=(cfg.kernel_size - 1) // 2,
            groups=cfg.model_dim,
            key=k2,
        )
        self.conv_pointwise1 = jax.vmap(nn.Linear(cfg.model_dim, 2 * cfg.model_dim, key=k1))
        self.conv_pointwise2 = jax.vmap(nn.Linear(cfg.model_dim, cfg.model_dim, key=k3))
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm1 = jax.vmap(nn.LayerNorm(cfg.model_dim))
        self.norm2 = jax.vmap(nn.LayerNorm(cfg.model_dim))

    def __call__(
        self,
        x: Float[Array, "T F"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        x = self.norm1(x)  # T, F
        x = self.conv_pointwise1(x)  # T, 2F
        x = jax.nn.glu(x, axis=-1)  # T, F

        x = jnp.transpose(x)  # F, T
        x = self.conv_depthwise(x)  # F, T

        x = jnp.transpose(x)  # T, F
        x = self.norm2(x)
        x = jnp.transpose(x)  # F, T

        x = jnp.transpose(x)  # T, F
        x = self.activation(x)
        x = self.conv_pointwise2(x)
        x = self.dropout(x, inference=inference, key=key)

        return x


class ConformerFeedForwardModule(eqx.Module):
    activation: Callable
    dropout: nn.Dropout
    ll1: nn.Linear
    ll2: nn.Linear
    norm: nn.LayerNorm

    def __init__(self, cfg: FeedForwardConfig, *, key: random.KeyArray):
        super().__init__()

        k1, k2 = random.split(key)

        self.activation = getattr(jax.nn, cfg.activation)
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ll1 = nn.Linear(cfg.model_dim, cfg.hidden_dim, key=k1)
        self.ll2 = nn.Linear(cfg.hidden_dim, cfg.model_dim, key=k2)

    def __call__(
        self,
        x: Float[Array, "F"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        x = self.norm(x)
        x = self.ll1(x)
        x = self.activation(x)
        x = self.ll2(x)
        x = self.dropout(x, key=key, inference=inference)

        return x


class ConformerMHSAModule(eqx.Module):
    dropout: nn.Dropout
    mhsa: nn.MultiheadAttention
    norm: nn.LayerNorm
    num_heads: int

    def __init__(self, cfg: MhsaConfig, *, key: random.KeyArray):
        super().__init__()

        self.dropout = nn.Dropout(cfg.dropout)
        self.mhsa = nn.MultiheadAttention(
            num_heads=cfg.num_att_heads,
            query_size=cfg.model_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=cfg.dropout,
            key=key,
        )
        self.norm = jax.vmap(nn.LayerNorm(cfg.model_dim))
        self.num_heads = cfg.num_att_heads

    def __call__(
        self,
        x: Float[Array, "T F"],
        seq_mask: Int[Array, "T"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        k1, k2 = random.split(key)

        mhsa_mask = self.make_self_attention_mask(seq_mask)

        x = self.norm(x)  # T, F
        x = self.mhsa(x, x, x, inference=inference, key=k1, mask=mhsa_mask)
        x = self.dropout(x, inference=inference, key=k2)

        return x

    def make_self_attention_mask(self, mask: Int[Array, "T"]) -> Float[Array, "num_heads T T"]:
        """Create self-attention mask from sequence-level mask."""

        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1),
            jnp.expand_dims(mask, axis=-2),
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class ConformerBlock(eqx.Module):
    ff1: ConformerFeedForwardModule
    mhsa: ConformerMHSAModule
    conv: ConformerConvolutionModule
    ff2: ConformerFeedForwardModule
    norm: nn.LayerNorm

    def __init__(self, cfg: ConformerBlockConfig, *, key: random.KeyArray):
        k1, k2, k3, k4 = random.split(key, 4)

        self.ff1 = eqx.filter_vmap(ConformerFeedForwardModule(cfg.ff, key=k1))
        self.mhsa = ConformerMHSAModule(cfg.mhsa, key=k2)
        self.conv = ConformerConvolutionModule(cfg.conv, key=k3)
        self.ff2 = eqx.filter_vmap(ConformerFeedForwardModule(cfg.ff, key=k4))
        self.norm = jax.vmap(nn.LayerNorm(cfg.ff.model_dim))

    def __call__(
        self,
        x: Float[Array, "T F"],
        seq_mask: Int[Array, "T"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        t = x.shape[0]
        k0, k1, k_conv, k_mhsa = random.split(key, 4)

        ff1_keys = random.split(k0, t)
        ff2_keys = random.split(k1, t)

        residual = x
        x = self.ff1(x, ff1_keys, inference)
        residual = 0.5 * x + residual
        x = self.mhsa(residual, seq_mask, k_mhsa, inference)
        residual = x + residual
        x = self.conv(residual, k_conv, inference)
        residual = x + residual
        x = self.ff2(residual, ff2_keys, inference)
        residual = 0.5 * x + residual
        x = self.norm(residual)

        return x


class Conformer1dVggModule(eqx.Module):
    """
    VGG-style convolutional model pre-processing a single DFL input down to a more
    reasonable feature dimension.

    Modules:
    - Conv1d
    - Conv1d
    - Activation
    - Pooling
    - Conv1d
    - Conv1d
    - Activation
    - Pooling
    """

    act: Callable
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv2d
    conv4: nn.Conv2d
    in_norm: nn.LayerNorm
    pool1: nn.MaxPool1d
    pool2: nn.MaxPool2d
    linear: nn.Linear

    def __init__(self, cfg: Conformer1dVggConfig, *, key: random.KeyArray):
        k0, k1, k2, k3, k4 = random.split(key, 5)

        self.act = getattr(jax.nn, cfg.activation)
        self.in_norm = jax.vmap(nn.LayerNorm(None, use_bias=False, use_weight=False))
        self.conv1 = jax.vmap(
            nn.Conv1d(
                in_channels=1,
                out_channels=cfg.conv1_channels,
                kernel_size=cfg.conv_kernel_size,
                key=k0,
            )
        )
        self.conv2 = jax.vmap(
            nn.Conv1d(
                in_channels=cfg.conv1_channels,
                out_channels=cfg.conv2_channels,
                kernel_size=cfg.conv_kernel_size,
                key=k1,
            )
        )
        self.conv3 = jax.vmap(
            nn.Conv1d(
                in_channels=cfg.conv2_channels,
                out_channels=cfg.conv3_channels,
                kernel_size=cfg.conv_kernel_size,
                key=k2,
            )
        )
        self.conv4 = jax.vmap(
            nn.Conv1d(
                in_channels=cfg.conv3_channels,
                out_channels=cfg.conv4_channels,
                kernel_size=cfg.conv_kernel_size,
                key=k3,
            )
        )
        self.linear = nn.Linear(cfg.linear_dim_in, cfg.linear_dim_out, key=k4)
        self.pool1 = jax.vmap(nn.MaxPool1d(kernel_size=cfg.pool1_kernel_size, stride=cfg.pool1_stride))
        self.pool2 = jax.vmap(nn.MaxPool1d(kernel_size=cfg.pool2_kernel_size, stride=cfg.pool2_stride))

    def __call__(
        self,
        x: Float[Array, "F D"],
        key: random.KeyArray,
        inference: bool = False,
    ) -> Float[Array, "E"]:
        # single DFL as param

        x = self.in_norm(x)  # normalize every row
        x = jnp.expand_dims(x, axis=1)  # F,1,D
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool1(x)  # F,C1,D'
        x = self.conv3(x)
        x = self.conv4(x)  # C2,C1',D'
        x = self.act(x)
        x = self.pool2(x)
        x = jnp.ravel(x)  # C*F'*B''
        x = self.linear(x)  # E

        return x


class ConformerFrontend(eqx.Module):
    dropout: nn.Dropout
    feature_embedding: nn.Linear
    norm: nn.LayerNorm
    pos_embedding: nn.Embedding
    type_embedding: nn.Embedding

    def __init__(self, cfg: ConformerFrontendConfig, *, key: random.KeyArray):
        k0, k1, k2 = random.split(key, 3)

        self.dropout = nn.Dropout(cfg.dropout)
        self.feature_embedding = jax.vmap(nn.Linear(cfg.n_in, cfg.n_out, key=k0))
        self.norm = nn.LayerNorm(cfg.n_out)
        self.pos_embedding = jax.vmap(nn.Embedding(cfg.max_pos, cfg.n_out, key=k1))

    def __call__(
        self,
        x: Float[Array, "T D"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        t = x.shape[0]

        data = self.feature_embedding(x)
        pos_emb = self.pos_embedding(jnp.arange(t))  # T F

        x = data + pos_emb  # T, F
        x = jax.vmap(self.norm)(x)
        x = self.dropout(x, inference=inference, key=key)

        return x


class ConformerEncoder(eqx.Module):
    blocks: List[ConformerBlock]
    frontend: ConformerFrontend

    def __init__(self, cfg: ConformerConfig, *, key: random.KeyArray):
        k0, *keys = random.split(key, cfg.num_blocks + 1)

        self.blocks = [ConformerBlock(cfg.block_cfg, key=k) for k in keys]
        self.frontend = ConformerFrontend(cfg.frontend_cfg, key=k0)

    def __call__(
        self,
        x: Float[Array, "T F"],
        seq_mask: Int[Array, "T"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        k, *keys = random.split(key, len(self.blocks) + 1)

        x = self.frontend(x, k, inference)
        for block, k_i in zip(self.blocks, keys):
            x = block(x, seq_mask, k_i, inference)

        return x


def mask_along_axis_iid(key: random.KeyArray, inputs: jnp.array, mask_param, axis, mask_value=0.0):
    """
    :param inputs: Input tensor of shape (batch, time, freq, 1)
    :param mask_param:
    :param mask_value:
    :param axis:
    :return:
    """
    # if axis not in [1, 2]:
    #     raise ValueError('Only Frequency and Time masking are supported')

    shape_ = (inputs.shape[0], inputs.shape[-1])

    value = random.uniform(key, shape_) * mask_param
    min_value = random.uniform(key, shape_) * (inputs.shape[axis] - value)
    mask_start = min_value.reshape(-1, 1, 1, 1)
    mask_end = (min_value + value).reshape(-1, 1, 1, 1)
    mask = jnp.arange(0, inputs.shape[axis])
    if axis == 1:
        inputs = inputs.transpose(0, 2, 3, 1)
    else:
        inputs = inputs.transpose(0, 1, 3, 2)
    o = (mask >= mask_start) & (mask < mask_end)
    o = jnp.repeat(o, inputs.shape[1], axis=1)
    inputs = jnp.where(o, 0, inputs)  # inputs.at[o].set(mask_value)

    if axis == 1:
        inputs = inputs.transpose(0, 3, 1, 2)
    else:
        inputs = inputs.transpose(0, 1, 3, 2)
    return inputs


class SpecAugment:
    def __init__(self, freq_param: int, time_param: int, num_masks: int = 2):
        self.freq_param = freq_param
        self.time_param = time_param
        self.num_masks = num_masks

    def augment(self, inputs, rng):
        outputs = inputs
        if self.freq_param != 0:
            rng, k0 = random.split(rng)
            outputs = mask_along_axis_iid(k0, outputs, self.freq_param, 2)
        if self.time_param != 0:
            rng, k0 = random.split(rng)
            outputs = mask_along_axis_iid(k0, outputs, self.time_param, 1)
        return outputs, rng

    def __call__(self, inputs, rng):
        outputs = inputs
        for _ in range(self.num_masks):
            outputs, rng = self.augment(outputs, rng)
        return outputs


class Classifier(eqx.Module):
    enc: ConformerEncoder
    sa: SpecAugment

    def __init__(self, cfg: ConformerConfig, *, key: random.KeyArray):
        self.enc = eqx.filter_vmap(ConformerEncoder(cfg, key=key))
        self.sa = SpecAugment(freq_param=20, time_param=10, num_masks=2)

    def __call__(
        self,
        x: Float[Array, "B T F"],
        seq_mask: Int[Array, "B T"],
        key: random.KeyArray,
        inference: bool = False,
    ):
        k0, k1 = random.split(key)

        if not inference:
            x = jnp.expand_dims(x, axis=3)
            x = self.sa(x, k0)
            x = jnp.squeeze(x)

        x = self.enc(x, seq_mask, k1, inference)

        return x
