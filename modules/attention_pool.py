import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    """Learned temporal downsampling, a drop-in replacement for nn.MaxPool1d.

    Max-pooling keeps the single strongest frame per window and drops the rest, so
    only that frame receives gradient and the motion inside the window is thrown
    away. Glosses are told apart by how the hands travel through a window, not by
    any one peak frame, which makes that the wrong summary on this task.

    Windows overlap (window > stride), so each output position sees more context
    than the two frames a stride-2 max-pool gives it. Output length stays
    floor(T / stride) -- identical to nn.MaxPool1d(stride, ceil_mode=False) -- so
    TemporalConv.update_lgt() keeps working unchanged.

    out_proj is zero-initialised and the window mean is added back as a residual,
    so before training this block computes exactly average pooling and learns to
    deviate from there. On ~900 training clips that is a much safer starting point
    than random attention weights.
    """

    def __init__(self, channels, window=8, stride=2, dim=256, num_heads=4,
                 dropout=0.1):
        super(AttentionPool1d, self).__init__()
        if window < stride:
            raise ValueError(f'window ({window}) must be >= stride ({stride})')
        if (window - stride) % 2:
            raise ValueError(f'window - stride must be even, got {window - stride}')
        if dim % num_heads:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')

        self.channels = channels
        self.window = window
        self.stride = stride
        self.pad = (window - stride) // 2
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(channels, dim)
        self.k_proj = nn.Linear(channels, dim)
        self.v_proj = nn.Linear(channels, dim)
        self.out_proj = nn.Linear(dim, channels)

        # The query is the window mean plus a learned offset.
        self.pos_embed = nn.Parameter(torch.zeros(channels))
        # One bias per (head, slot). The window is a fixed size, so a slot's offset
        # from the window centre takes exactly `window` distinct values.
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, window))

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, lgt=None):
        """(B, C, T) -> (B, C, floor(T / stride)).

        lgt: (B,) valid input lengths, float or int. Slots at or past a clip's
        length are masked out. Clips here range 85-265 frames and collate_fn pads
        by replicating the last frame, so at batch 2 a short clip can be over half
        padding -- an unmasked softmax would just return that frozen frame.
        """
        B, C, T = x.shape
        x_pad = F.pad(x, (self.pad, self.pad), mode='replicate')
        win = x_pad.unfold(dimension=2, size=self.window, step=self.stride)
        T_out = win.size(2)

        win = win.permute(0, 2, 3, 1)                       # (B, T_out, window, C)
        ctx = win.mean(dim=2)                               # (B, T_out, C)

        q = self.q_proj(ctx + self.pos_embed)               # (B, T_out, D)
        k = self.k_proj(win)                                # (B, T_out, window, D)
        v = self.v_proj(win)

        q = q.view(B, T_out, self.num_heads, self.head_dim)
        k = k.view(B, T_out, self.window, self.num_heads, self.head_dim)
        v = v.view(B, T_out, self.window, self.num_heads, self.head_dim)

        scores = torch.einsum('bthd,btwhd->bthw', q, k) * self.scale
        scores = scores + self.rel_pos_bias.view(1, 1, self.num_heads, self.window)

        if lgt is not None:
            scores = scores.masked_fill(
                self._pad_mask(lgt, T_out, x.device), float('-inf'))

        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.einsum('bthw,btwhd->bthd', attn, v)
        out = out.reshape(B, T_out, -1)
        out = self.dropout(self.out_proj(out))

        out = self.norm(ctx + out)
        return out.transpose(1, 2)                          # (B, C, T_out)

    def _pad_mask(self, lgt, T_out, device):
        """True where a window slot points past the end of its clip.

        Returns (B, T_out, 1, window) to broadcast over heads.
        """
        lgt = lgt.to(device=device, dtype=torch.long)
        t = torch.arange(T_out, device=device).view(T_out, 1)
        w = torch.arange(self.window, device=device).view(1, self.window)
        idx = t * self.stride + w - self.pad                # (T_out, window)
        invalid = idx.unsqueeze(0) >= lgt.view(-1, 1, 1)    # (B, T_out, window)
        # Leave fully-invalid windows unmasked: an all -inf row makes softmax NaN,
        # and those output positions sit past feat_len so nothing downstream reads
        # them anyway.
        invalid = invalid & ~invalid.all(dim=-1, keepdim=True)
        return invalid.unsqueeze(2)

    def extra_repr(self):
        return (f'channels={self.channels}, window={self.window}, '
                f'stride={self.stride}, heads={self.num_heads}, '
                f'head_dim={self.head_dim}')
