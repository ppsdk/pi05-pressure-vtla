import torch
from torch import nn


class TimeFingerXYZRoPE(nn.Module):
    def __init__(self, head_dim, config):
        super().__init__()
        self.head_dim = head_dim
        self.axis_dim = 2 * ((head_dim // 2) // 5)
        self.config = dict(config)
        for name, theta in self.frequency_specs():
            self.register_buffer(f"{name}_inv_freq", self.inverse_frequency(theta), persistent=False)

    def frequency_specs(self):
        return tuple((name, self.config[field]) for name, field in (("time", "time_theta"), ("finger", "finger_theta"), ("x", "spatial_theta"), ("y", "spatial_theta"), ("z", "spatial_theta")))

    def inverse_frequency(self, theta):
        theta_fp32 = torch.tensor(theta, dtype=torch.float32)
        exponent = -torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim
        return torch.pow(theta_fp32, exponent)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, theta in self.frequency_specs():
            buffer_name = f"{name}_inv_freq"
            device = getattr(self, buffer_name).device
            setattr(self, buffer_name, self.inverse_frequency(theta).to(device=device))
        return self

    @staticmethod
    def rotate(values, cos, sin):
        values = values.float()
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, q, k, address):
        positions = (address.time_id, address.finger_id, address.pos_xyz[..., 0], address.pos_xyz[..., 1], address.pos_xyz[..., 2])
        scale_fp32 = float(torch.tensor(self.config["spatial_coordinate_scale"], dtype=torch.float32).item())
        scales = (1.0, 1.0, scale_fp32, scale_fp32, scale_fp32)
        q_chunks, k_chunks = [], []
        start = 0
        for (name, _), position, scale in zip(self.frequency_specs(), positions, scales):
            end = start + self.axis_dim
            phase = position.float()[:, None, :, None] * float(scale) * getattr(self, f"{name}_inv_freq").float()[None, None, None, :]
            cos, sin = phase.cos(), phase.sin()
            q_chunks.append(self.rotate(q[..., start:end], cos, sin))
            k_chunks.append(self.rotate(k[..., start:end], cos, sin))
            start = end
        if start < self.head_dim:
            q_chunks.append(q[..., start:].float())
            k_chunks.append(k[..., start:].float())
        return torch.cat(q_chunks, dim=-1).to(q.dtype), torch.cat(k_chunks, dim=-1).to(k.dtype)
