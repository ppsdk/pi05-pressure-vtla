from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenAddress:
    time_id: torch.Tensor
    finger_id: torch.Tensor
    pos_xyz: torch.Tensor


def canonicalize(address, batch_size):
    return TokenAddress(
        time_id=address.time_id.unsqueeze(0).expand(batch_size, -1),
        finger_id=address.finger_id.unsqueeze(0).expand(batch_size, -1),
        pos_xyz=address.pos_xyz,
    )
