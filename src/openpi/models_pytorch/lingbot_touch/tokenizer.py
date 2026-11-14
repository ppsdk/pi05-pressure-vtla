import torch
from torch import nn

from .address import TokenAddress


class PointTokenizer(nn.Module):
    def __init__(self, embed_dim, tubelet_size):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.signal_mlp = nn.Sequential(nn.Linear(tubelet_size * 3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, signal, pos_xyz, coverage_mask):
        batch, frames, fingers, taxels, channels = signal.shape
        keep = coverage_mask.reshape(-1).nonzero().flatten().to(signal.device)
        spatial = keep.numel()
        compact_signal = signal.reshape(batch, frames, fingers * taxels, channels).index_select(2, keep)
        pos = pos_xyz.to(device=signal.device, dtype=torch.float32)
        if pos.ndim == 3:
            pos = pos.unsqueeze(0).expand(batch, -1, -1, -1)
        compact_pos = pos.reshape(batch, fingers * taxels, 3).index_select(1, keep)
        finger = torch.arange(fingers, dtype=torch.long, device=signal.device).view(fingers, 1).expand(fingers, taxels).reshape(-1).index_select(0, keep)
        times = frames // self.tubelet_size
        tubelets = compact_signal.reshape(batch, times, self.tubelet_size, spatial, channels).permute(0, 1, 3, 2, 4).contiguous().reshape(batch, times, spatial, self.tubelet_size * channels)
        tokens = self.signal_mlp(tubelets)
        address = TokenAddress(
            time_id=torch.arange(times, dtype=torch.long, device=signal.device).repeat_interleave(spatial),
            finger_id=finger.repeat(times),
            pos_xyz=compact_pos[:, None, :, :].expand(batch, times, spatial, 3).reshape(batch, times * spatial, 3),
        )
        return tokens.reshape(batch, times * spatial, -1), address


class ModalityRoutedTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_by_source = dict(config.group_by_source)
        self.tokenizers = nn.ModuleDict({group: PointTokenizer(config.embed_dim, config.tubelet_size) for group in dict.fromkeys(config.group_by_source.values())})

    def forward(self, signal, pos_xyz, coverage_mask, modality):
        return self.tokenizers[self.group_by_source[modality]](signal, pos_xyz, coverage_mask)
