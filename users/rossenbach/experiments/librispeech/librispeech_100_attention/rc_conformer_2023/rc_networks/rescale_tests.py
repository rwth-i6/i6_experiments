from returnn_common import nn

class BF16Linear(nn.Linear):
    """
    Linear transformation.
    """

    def __call__(self, source: nn.Tensor) -> nn.Tensor:
        if not isinstance(source, nn.Tensor):
            raise TypeError(f"{self}: source must be a Tensor but got {type(source)}")
        if self.in_dim not in source.dims_set:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")
        source_bf16 = nn.cast(source, dtype="bfloat16")
        weights_bf16 = nn.cast(self.weight, dtype="bfloat16")
        bias_bf16 = nn.cast(self.bias, dtype="bfloat16")
        out = nn.dot(source_bf16, weights_bf16, reduce=self.in_dim)
        if self.with_bias:
            out += bias_bf16
        out = nn.cast(out, dtype="float32")
        return out


class Memristor(nn.Linear):
    """
    Linear transformation.
    """

    def __call__(self, source: nn.Tensor) -> nn.Tensor:
        if not isinstance(source, nn.Tensor):
            raise TypeError(f"{self}: source must be a Tensor but got {type(source)}")
        if self.in_dim not in source.dims_set:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")
        scaled_weights = self.weight * 100
        printed_weights = nn.print(scaled_weights)
        weights_quantized = nn.make_layer(
            layer_dict={
                "class": "eval",
                "eval": "tf.quantization.fake_quant_with_min_max_args(source(0), num_bits=4)",
                # "from": self.weight,
                "from": printed_weights,
            },
            name="quantize",
        )
        weights_quantized = nn.print(weights_quantized)
        out = nn.dot(source, weights_quantized, reduce=self.in_dim)
        if self.with_bias:
            out += self.bias
        out = nn.cast(out, dtype="float32")
        return out