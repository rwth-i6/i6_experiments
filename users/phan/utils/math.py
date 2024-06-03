"""
Some specific tensor operations
"""

import torch

def log_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Log matrix multiplication, i.e.
    A = log X, B = log Y
    -> log_matmul(A, B) = log (X @ Y)
    https://stackoverflow.com/questions/36467022/handling-matrix-multiplication-in-log-space-in-python

    :param A: first matrix in log scale
    :param B: second matrix in log scale
    :returns: matrix product of the two matrices in log scale
    """
    m, n = A.shape
    n, r = B.shape
    A_expand = A.unsqueeze(0).expand(r, -1, -1).transpose(0, 1) # (m, r, n)
    B_expand = B.unsqueeze(0).expand(m, -1, -1).transpose(1, 2) # (m, r, n)
    return (A_expand + B_expand).logsumexp(dim=-1)


def modified_log_matmul(A: torch.Tensor, B: torch.Tensor, log_zero=-1e15):
    """
    Special case of log_matmul to calculate
    Z_ij = sum{k != j} X_{ik}*Y_{kj}
    where A = log X, B = log Y,
    B is square matrix
    """
    m, n = A.shape
    A_expand = A.unsqueeze(0).expand(n, -1, -1).transpose(0, 1) # (m, n, n)
    B_expand = B.unsqueeze(0).expand(m, -1, -1).transpose(1, 2) # (m, n, n)
    C = A_expand + B_expand
    # to exclude k = j from the summation, apply some masking here
    index = torch.arange(n).unsqueeze(0).expand(m, -1).unsqueeze(-1).to(A.device)
    # write log zero to C[i][j][j] for all i, j
    C_masked = torch.scatter(C, 2, index, log_zero)
    return C_masked.logsumexp(dim=-1)
