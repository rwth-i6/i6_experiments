# test whether the torch and returnn implementation of the mhsa layer are equivalent

import torch

import returnn.frontend as rf

from returnn.tensor import Dim

import itertools

torch.backends.mha.set_fastpath_enabled(False)

rf.select_backend_torch()
rf.init_forward_step_run_ctx()

spatial_dim = Dim(3, name="spatial_dim")
out_dim = Dim(512, name="out_dim")

random_opts = {
    "distribution": "normal",
    "dtype": "float32",
}

rf.set_random_seed(1)

rf_input = rf.random(dims=[Dim(1), spatial_dim, out_dim], **random_opts)

qkv_weight = rf.random(dims=[out_dim, 3 * out_dim], **random_opts)
# qkv_weight = rf.full(dims=[out_dim, 3*out_dim], fill_value=0.0, dtype="float32")
# qkv_weight.raw_tensor[:,1024:] = 0.0
# qkv_weight.raw_tensor[:,0:1023] = 0.0
#
# eye = torch.eye(512)
# zeros = torch.zeros(512, 512)
#
# qkv_raw = torch.cat([eye, eye, zeros], dim=1)
#
# qkv_weight.raw_tensor = qkv_raw

qkv_bias = rf.random(dims=[3 * out_dim], **random_opts)
# qkv_bias = rf.full(dims=[3*out_dim], fill_value=0.0, dtype="float32")
# qkv_bias.raw_tensor[1024:] = 0.0

proj_weight = rf.random(dims=[out_dim, out_dim], **random_opts)
proj_bias = rf.random(dims=[out_dim], **random_opts)

torch_input = rf_input.raw_tensor

rf_mhsa = rf.SelfAttention(
    in_dim=out_dim,
    proj_dim=out_dim,
    key_dim_total=out_dim,
    value_dim_total=out_dim,
    num_heads=8,
    att_dropout=0.1,
)

rf_mhsa.qkv.weight._raw_backend.set_parameter_initial_value(
    rf_mhsa.qkv.weight, qkv_weight
)
rf_mhsa.qkv.bias._raw_backend.set_parameter_initial_value(rf_mhsa.qkv.bias, qkv_bias)

rf_mhsa.proj.weight._raw_backend.set_parameter_initial_value(
    rf_mhsa.proj.weight, proj_weight.raw_tensor
)
rf_mhsa.proj.bias._raw_backend.set_parameter_initial_value(rf_mhsa.proj.bias, proj_bias)

torch_mhsa = torch.nn.MultiheadAttention(
    512,
    8,
    dropout=0.1,
    batch_first=True,
)

state_dict = torch_mhsa.state_dict()

num_heads = 8

state_dict["in_proj_weight"] = (
    qkv_weight.raw_tensor.reshape(
        out_dim.dimension, num_heads, 3, out_dim.dimension // num_heads
    )
    .permute(2, 1, 3, 0)
    .reshape(-1, out_dim.dimension)
)
state_dict["in_proj_bias"] = (
    qkv_bias.raw_tensor.reshape(num_heads, 3, out_dim.dimension // num_heads)
    .permute(1, 0, 2)
    .reshape(-1)
)
state_dict["out_proj.weight"] = (
    proj_weight.raw_tensor.reshape(
        num_heads, out_dim.dimension // num_heads, out_dim.dimension
    )
    .permute(2, 0, 1)
    .reshape(-1, out_dim.dimension)
)
state_dict["out_proj.bias"] = proj_bias.raw_tensor

torch_mhsa.load_state_dict(state_dict)
torch_mhsa.eval()

rf_output = rf_mhsa(rf_input, axis=spatial_dim)
torch_output, _ = torch_mhsa(
    torch_input, torch_input, torch_input, key_padding_mask=None, need_weights=False
)

print("RF output")
print(rf_output.raw_tensor)
print(rf_output.raw_tensor.shape)
print("---------------------------")
print("Torch output")
print(torch_output)
print(torch_output.shape)
print("Same: ", torch.allclose(rf_output.raw_tensor, torch_output, atol=1e-3))

## extended checks

if False:
    q_raw, k_raw, v_raw = torch.split(qkv_weight.raw_tensor, 512, dim=1)

    weights = [q_raw, k_raw, v_raw]

    any = False

    for perm in itertools.permutations(weights):
        for q_inv, k_inv, v_inv in itertools.product([0, 1], [0, 1], [0, 1]):
            q_raw, k_raw, v_raw = perm

            if q_inv == 1:
                q_raw = q_raw.T
            if k_inv == 1:
                k_raw = k_raw.T
            if v_inv == 1:
                v_raw = v_raw.T

            qkv_weight_adj_raw = torch.cat([q_raw.T, k_raw.T, v_raw.T], dim=1)

            state_dict = torch_mhsa.state_dict()
            state_dict["in_proj_weight"] = qkv_weight_adj_raw.T
            state_dict["in_proj_bias"] = qkv_bias.raw_tensor
            state_dict["out_proj.weight"] = proj_weight.raw_tensor
            state_dict["out_proj.bias"] = proj_bias.raw_tensor

            torch_mhsa.load_state_dict(state_dict)
            torch_mhsa.eval()

            rf_output = rf_mhsa(rf_input, axis=spatial_dim)
            torch_output, _ = torch_mhsa(
                torch_input,
                torch_input,
                torch_input,
                key_padding_mask=None,
                need_weights=False,
            )

            print("RF output")
            print(rf_output.raw_tensor)
            print(rf_output.raw_tensor.shape)
            print("---------------------------")
            print("Torch output")
            print(torch_output)
            print(torch_output.shape)

            if torch.allclose(rf_output.raw_tensor, torch_output, atol=1e-6):
                print("Match with perm: ", perm)
                print("Match with inversions: ", q_inv, k_inv, v_inv)
                any = True
                break

        if any:
            break

    print("Done")
