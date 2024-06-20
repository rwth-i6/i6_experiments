# Importing i6_models conformer into returnn frontend setup

## Structural differences
| Difference                                             | * | i6_models | rf |
|--------------------------------------------------------|---|-----------|--|
| input_projection uses bias                             |   | yes       | no |
| self attention qkv uses bias                           |   | yes       | no |
| convolutional kernel size                              |   | 31        | 32 |
| positional encoding in mhsa                            | * | no        | yes |
| batch norm                                             | * | no        | yes |
| ff_activation                                          |   | silu      | rf.relu(x) ** 2.0 |
| vgg frontend (several differences)                     |   |           |  |
| feature extraction (not sure of the exact difference ) |   |           |  |

VGG Frontend configuration for rf:
```
ConformerConvSubsampleV2(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                    Dim(32, name="conv4"),  # Changed: Dim(64, name="conv4")
                ],
                filter_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],  # Changed
                activation_times=[False, True, False, True],  # Changed
                pool_sizes=[(1, 1), (3, 1), (1, 1), (2, 1)],  # Changed
                strides=[(1, 1), (1, 1), (1, 1), (1, 1)],  # Changed
                padding="same",  # Changed: padding="valid"
                pool_padding="valid",  # Changed
                swap_merge_dim_order=True,  # Changed
                # Note: uses relu activation by default
            ),
```

## Adjust weights

Transpose:

```
_transpose_list = [
    "encoder_out_linear.weight",
    "encoder.input_projection.weight",
    "joiner.linear.weight",
    "predictor.linear.weight",
]

for layer_idx in range(12):
    _transpose_list.append(f"encoder.layers.{layer_idx}.ffn1.linear_ff.weight")
    _transpose_list.append(f"encoder.layers.{layer_idx}.ffn1.linear_out.weight")
    _transpose_list.append(f"encoder.layers.{layer_idx}.ffn2.linear_ff.weight")
    _transpose_list.append(f"encoder.layers.{layer_idx}.ffn2.linear_out.weight")

    _transpose_list.append(
        f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
    )
    _transpose_list.append(
        f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
    )

```

Adjust self att weights:
```
    if name.endswith(".self_att.qkv.weight"):
        value = ckpt["model"][
            f"conformer.module_list.{layer_idx}.mhsa.mhsa.in_proj_weight"
        ]
        value = value.reshape(3, num_heads, self_att_dim // num_heads, self_att_dim).permute(3, 1, 0, 2).reshape(self_att_dim, -1)
        return value.numpy()

    if name.endswith(".self_att.qkv.bias"):
        value = ckpt["model"][
            f"conformer.module_list.{layer_idx}.mhsa.mhsa.in_proj_bias"
        ]
        value = value.reshape(3, num_heads, self_att_dim // num_heads).permute(1, 0, 2).reshape(-1)
        return value.numpy()

    if name.endswith(".self_att.proj.weight"):
        value = ckpt["model"][
            f"conformer.module_list.{layer_idx}.mhsa.mhsa.out_proj.weight"
        ]
        value = value.reshape(self_att_dim, num_heads, self_att_dim // num_heads).permute(1, 2, 0).reshape(self_att_dim, -1)
        return value.numpy()
```