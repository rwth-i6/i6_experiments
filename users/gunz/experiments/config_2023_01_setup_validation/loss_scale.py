import typing

from ...setups.fh.factored import PhoneticContext
from .config import FH_LOSS_SHARE_FINAL_OUT

FINAL_LOSS_SCALE = 1.0


def get_int_loss_scale(
    num_final_losses: int,
    aux_losses: typing.List[typing.Tuple[int, PhoneticContext, bool]],
    final_loss_share: float = FH_LOSS_SHARE_FINAL_OUT,
) -> float:
    total_num_int_losses = float(sum([1 if center_only else 3 for _, _, center_only in aux_losses]))
    total_num_final_losses = float(num_final_losses)

    total_loss_weight = (1.0 / final_loss_share) * (total_num_final_losses * FINAL_LOSS_SCALE)
    remaining_int_loss_weight = total_loss_weight - (total_num_final_losses * FINAL_LOSS_SCALE)

    return remaining_int_loss_weight / total_num_int_losses if total_num_int_losses > 0 else remaining_int_loss_weight
