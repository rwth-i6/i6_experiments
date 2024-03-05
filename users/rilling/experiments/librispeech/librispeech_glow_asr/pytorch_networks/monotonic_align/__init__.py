import numpy as np
import torch
try:
    from .core import maximum_path_c
except:
    import subprocess
    import sys
    import os 
    dir = os.path.realpath(os.path.dirname(__file__))
    subprocess.call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=dir)
    from .core import maximum_path_c
    
def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


# def maximum_path_noC(paths, values, t_xs, t_ys, max_neg_value=-1e9):
#     for i in range(values.shape[0]):
#         maximum_path_each


# def maximum_path_each(path, value, t_x, t_y, max_neg_value):
#     for y in range(t_y):
#         for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
#             if x == y:
#                 v_cur = max_neg_value
#             else:
#                 v_cur = value[x, y - 1]
            
#             if x == 0:
#                 if y == 0:
#                     v_prev = max_neg_value
#                 else:
#                     v_prev = max_neg_value
#             else:
#                 v_prev = value[x-1, y-1]

#             value[x, y] = max(v_cur, v_prev) + value[x, y]
    
#     index = t_x -1
#     for y in range(t_y - 1, -1, -1):
#         path[index, y] = 1
        
#         if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
#             index = index - 1
