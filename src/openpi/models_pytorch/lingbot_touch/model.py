from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .address import TokenAddress, canonicalize
from .config import EncoderConfig
from .rope import TimeFingerXYZRoPE
from .tokenizer import ModalityRoutedTokenizer


@dataclass(frozen=True)
class EncoderOutput:
    latent: torch.Tensor
    address: TokenAddress


class AddressAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.address_rope = TimeFingerXYZRoPE(self.head_dim, config.address_rope)

    def forward(self, x, address):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.address_rope(q, k, address)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class AddressBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embed_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AddressAttention(config)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * config.mlp_ratio)), nn.GELU(), nn.Linear(int(dim * config.mlp_ratio), dim))

    def forward(self, x, address):
        x = x + self.attn(self.norm1(x), address)
        return x + self.mlp(self.norm2(x))


class AddressTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_layers = config.output_layers
        self.blocks = nn.ModuleList([AddressBlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norms_block = nn.ModuleList([nn.LayerNorm(config.embed_dim, eps=1e-6) for _ in config.output_layers])

    def forward(self, x, address):
        address = canonicalize(address, x.shape[0])
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x, address)
            if index in self.output_layers:
                outputs.append(self.norms_block[self.output_layers.index(index)](x))
        return torch.cat(outputs, dim=-1)


class SharedPointwiseProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, tokens):
        return tokens + self.fc2(self.act(self.fc1(self.norm(tokens))))


class TargetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, EncoderConfig):
            raise TypeError("config must be EncoderConfig")
        self.config = EncoderConfig.from_dict(config.to_dict())
        self.embed_dim = config.embed_dim
        self.target_dim = config.embed_dim * len(config.output_layers)
        self.tokenizer = ModalityRoutedTokenizer(config)
        self.transformer = AddressTransformer(config)
        self.shared_projector = SharedPointwiseProjector(config.embed_dim) if config.use_shared_projector else None
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def validate_inputs(self, signal, pos_xyz, coverage_mask, modality):
        if not isinstance(modality, str) or modality not in self.config.group_by_source:
            raise ValueError(f"modality must be one of {tuple(self.config.group_by_source)}")
        if not isinstance(signal, torch.Tensor) or not signal.is_floating_point() or signal.ndim != 5 or signal.shape[2] != 5 or signal.shape[-1] != 3:
            raise ValueError("signal must be floating [B,T,5,M,3]")
        batch, frames, _, taxels, _ = signal.shape
        if batch < 1 or taxels < 1 or frames < 1 or frames % self.config.tubelet_size:
            raise ValueError("B, T and M must be positive and T divisible by tubelet_size")
        if signal.device != next(self.parameters()).device:
            raise ValueError("signal device must match encoder device")
        if not isinstance(pos_xyz, torch.Tensor) or not pos_xyz.is_floating_point() or tuple(pos_xyz.shape) not in ((5, taxels, 3), (batch, 5, taxels, 3)):
            raise ValueError("pos_xyz must be floating [5,M,3] or [B,5,M,3]")
        if not isinstance(coverage_mask, torch.Tensor) or coverage_mask.dtype != torch.bool or tuple(coverage_mask.shape) != (5, taxels) or not bool(coverage_mask.any()):
            raise ValueError("coverage_mask must be shared nonempty bool [5,M]")
        if not bool(torch.isfinite(signal).all()) or not bool(torch.isfinite(pos_xyz).all()) or not bool(torch.isfinite(pos_xyz.float()).all()):
            raise ValueError("signal and pos_xyz must contain finite values representable in runtime precision")

    @torch.no_grad()
    def encode(self, signal, pos_xyz, coverage_mask, *, modality):
        self.validate_inputs(signal, pos_xyz, coverage_mask, modality)
        tokens, address = self.tokenizer(signal, pos_xyz, coverage_mask, modality)
        if self.shared_projector is not None:
            tokens = self.shared_projector(tokens)
        latent = self.transformer(tokens, address)
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("encoder produced nonfinite latent values")
        return EncoderOutput(latent=latent, address=address)

    def forward(self, signal, pos_xyz, coverage_mask, *, modality):
        return self.encode(signal, pos_xyz, coverage_mask, modality=modality)
