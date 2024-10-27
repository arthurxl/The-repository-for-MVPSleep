import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


def zigzag_path(M, N):
    def zigzag_path_lr(M, N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(M):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else M - 1 - j
                path.append((start_row + dir_row * i) * M + start_col + dir_col * col)
        return path

    def zigzag_path_tb(M, N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(M):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * M + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, M - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, M - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(M, N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(M, N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)

    _zz_paths = paths[:8]

    return _zz_paths


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 14, emb_size: int = 768, img_size: int = 129 * 29,
                 register: int = 5):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.c_token = nn.Parameter(torch.randn(1, 1 + register, emb_size), requires_grad=True)
        self.positions = nn.Parameter(torch.randn((img_size // (patch_size * patch_size)), emb_size))
        self._zz_paths = zigzag_path(9, 2)

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x_old = self.projection(x)
        x = x_old[:, self._zz_paths[4], :]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上

        queries, keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 14,
                 emb_size: int = 256,  # 768
                 img_size: int = 129 * 29,
                 depth: int = 12,
                 register: int = 5,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size, register),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
        )


class MLP_Proj(nn.Module):
    def __init__(self):
        super(MLP_Proj, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(4864, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4864)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class SubModel(nn.Module):
    def __init__(self, args):
        super(SubModel, self).__init__()
        self.register: int = 5
        self.encoder = ViT(register=self.register)
        self.args = args
        self.projection = MLP_Proj()
        self.sequence_transformer = Seq_Transformer(patch_size=256, dim=128, depth=4, heads=4, mlp_dim=128)

    def forward(self, x):
        x = x.split(1, 1)
        output_list = []
        for data in x:
            data = self.encoder(data)[:, 0:-self.register - 1].permute(0, 2, 1)
            data = data.contiguous().view(data.size()[0], -1)
            data = self.projection(data)
            out = data.reshape(data.size(0), -1, 256)
            output_list.append(out)
        x_output = torch.cat(output_list, 1)
        out = self.sequence_transformer(x_output)
        out = out.contiguous().view(out.size()[0], self.args.seq_len, -1)
        return out


class SiamModelTWO(nn.Module):
    def __init__(self, args, is_train):
        super(SiamModelTWO, self).__init__()
        self.online = SubModel(args)
        self.target = SubModel(args)
        self.args = args
        self.is_train = is_train
        self.prediction = nn.Sequential(
            nn.Linear(2432, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2432),
            nn.ReLU()
        )

        self.m = 0.4
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)

    def trian(self, x1, x2):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        x1_online = self.online(x1)
        x1_online = self.prediction(x1_online)

        x1_target = self.target(x1)

        x2_online = self.online(x2)
        x2_online = self.prediction(x2_online)

        x2_target = self.target(x2)
        return x1_online, x1_target, x2_online, x2_target

    def test(self, x1):
        x1_online = self.online(x1)
        return x1_online

    def forward(self, x1, x2):
        if (self.is_train):
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()
            x1_online = self.online(x1)
            x1_online = self.prediction(x1_online)

            x1_target = self.target(x1)

            x2_online = self.online(x2)
            x2_online = self.prediction(x2_online)

            x2_target = self.target(x2)
            return x1_online, x1_target, x2_online, x2_target
        else:
            x1_online = self.online(x1)
            return x1_online


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        x = self.transformer(x)
        return x


class Seq_GRU(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Seq_GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out
