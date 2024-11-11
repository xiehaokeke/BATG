import torch
import torch.nn as nn
import math
import numpy as np
from GEN.helper import trunc_normal_, to_2tuple, DropPath


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def updown(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=4, mode='bicubic')
    x = nn.AvgPool2d(4)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        self.act_layer = gelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
        # self.noise_strength_2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        # x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_2
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, que_dim, key_dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.que_dim = que_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        head_dim = que_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_transform = nn.Linear(que_dim, que_dim, bias=qkv_bias)
        self.k_transform = nn.Linear(key_dim, que_dim, bias=qkv_bias)
        self.v_transform = nn.Linear(key_dim, que_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(que_dim, que_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, embedding):
        B, N, C = x.shape
        B, E_N, E_C = embedding.shape

        # transform
        q = self.q_transform(x)
        k = self.k_transform(embedding)
        v = self.v_transform(embedding)
        # reshape
        q = q.reshape(B, N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)
        k = k.reshape(B, E_N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)
        v = v.reshape(B, E_N, self.num_heads, self.que_dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, C)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.size(-1) == v.size(-2), f"attn.size: {attn.size()}, v.size:{v.size()}"
        output = self.mat(attn, v).transpose(1, 2).reshape(B, N, self.que_dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        return x + output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        self.norm = PixelNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.window_size = window_size
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cross_attention = CrossAttention(que_dim=dim, key_dim=embedding_dim, num_heads=num_heads)

    def forward(self, inputs):
        x, embedding = inputs
        x = self.cross_attention(x, embedding)
        B, N, C = x.size()
        H = W = int(np.sqrt(N))
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, N, C)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return [x, embedding]


class StageBlock(nn.Module):
    def __init__(self, depth, dim, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=window_size
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x, embedding):
        x = self.block([x, embedding])
        return x


class Generator(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=200, embed_dim=384, depth=4,
                 num_heads=3, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()
        self.ch = embed_dim
        self.bottom_width = 7
        self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm
        act_layer = gelu

        self.linear = nn.Linear(512, (self.bottom_width ** 2) * self.embed_dim)
        self.embedding_transform = nn.Linear(512, (self.bottom_width ** 2) * self.embed_dim)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim//4))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, (self.bottom_width * 16) ** 2, embed_dim//16))
        self.pos_embed_6 = nn.Parameter(torch.zeros(1, (self.bottom_width * 32) ** 2, embed_dim//64))
        self.embed_pos = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6
        ]
        self.blocks_0 = StageBlock(
            depth=depth,
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=7
        )
        self.blocks_1 = StageBlock(
            depth=depth,
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=14
        )
        self.blocks_2 = StageBlock(
            depth=depth,
            dim=embed_dim,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=28
        )
        self.blocks_3 = StageBlock(
            depth=depth,
            dim=embed_dim//4,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=56
        )
        self.blocks_4 = StageBlock(
            depth=depth,
            dim=embed_dim//16,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=56
        )
        self.blocks_5 = StageBlock(
            depth=depth,
            dim=embed_dim//64,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=56
        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//64, 3, 3, 1, 1)
        )

    def forward(self, z):
        latent_size = z.size(-1)
        z = (z / z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        x = self.linear(z).view(-1, self.bottom_width ** 2, self.embed_dim)

        # input noise
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size(0)
        H, W = self.bottom_width, self.bottom_width

        # embedding
        embedding = self.embedding_transform(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        embedding = embedding + self.embed_pos.to(embedding.get_device())

        '''
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size()
        H, W = self.bottom_width, self.bottom_width
        x, _ = self.blocks_0(x, embedding)
        x, H, W = updown(x, H, W) # 7 384
        '''

        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[1].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_1(x, embedding)
        x, H, W = updown(x, H, W) # 14 384

        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[2].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_2(x, embedding)
        x, H, W = updown(x, H, W) # 28 384

        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[3].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_3(x, embedding)
        x, H, W = updown(x, H, W) # 56 96

        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[4].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_4(x, embedding)
        x, H, W = updown(x, H, W) # 112 24

        
        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[5].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_5(x, embedding)
        x, H, W = updown(x, H, W) # 224 6
        

        x = x.permute(0, 2, 1).view(B, C, 224, 224)
        output = self.deconv(x)
        return output
