"""Compact StyleGAN2: mapping network, modulated convs, skip generator, residual discriminator."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 512
W_DIM = 512
CLAMP = 256  # keeps fp16 activations finite
GAIN = math.sqrt(2)


def channels(res, base=16384, cap=512):
    # 4..32 px: 512, 64: 256, 128: 128, 256: 64, 512: 32
    return min(cap, base // res)


class BiasAct(torch.autograd.Function):
    """x + bias -> optional leaky ReLU * sqrt(2) -> clamp, saving only the output for backward."""

    @staticmethod
    def forward(ctx, x, b, act):
        y = x + b.view(1, -1, 1, 1).to(x.dtype)
        if act:
            y = F.leaky_relu(y, 0.2).mul_(GAIN)
        y = y.clamp_(-CLAMP, CLAMP)
        ctx.act = act
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, gy):
        (y,) = ctx.saved_tensors
        slope = torch.where(y > 0, GAIN, 0.2 * GAIN) if ctx.act else 1.0
        gx = gy * ((y.abs() < CLAMP) * slope).to(gy.dtype)
        return gx, gx.sum([0, 2, 3]).float(), None


def bias_act(x, b, act):
    return BiasAct.apply(x, b, act)


class EqLinear(nn.Module):
    """Linear layer with equalized learning rate."""

    def __init__(self, cin, cout, bias_init=0.0, lr_mul=1.0, act=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin) / lr_mul)
        self.bias = nn.Parameter(torch.full((cout,), float(bias_init)))
        self.scale = lr_mul / math.sqrt(cin)
        self.lr_mul = lr_mul
        self.act = act

    def forward(self, x):
        x = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        return F.leaky_relu(x, 0.2) * GAIN if self.act else x


class EqConv(nn.Module):
    """Conv with equalized learning rate, optional 2x average-pool downsample."""

    def __init__(self, cin, cout, k, bias=True, act=True, down=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None
        self.scale = 1 / math.sqrt(cin * k * k)
        self.act, self.down = act, down

    def forward(self, x):
        x = F.conv2d(x, self.weight * self.scale, padding=self.weight.shape[-1] // 2)
        if self.down:
            x = F.avg_pool2d(x, 2)
        if self.bias is None:
            return x
        return bias_act(x, self.bias, self.act)


class ModConv(nn.Module):
    """Style-modulated conv. Modulates activations instead of weights so it maps to a plain conv."""

    def __init__(self, cin, cout, k, demod=True, up=False, act=True, noise_res=None):
        super().__init__()
        self.affine = EqLinear(W_DIM, cin, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale = 1 / math.sqrt(cin * k * k)
        self.demod, self.up, self.act = demod, up, act
        if noise_res:
            self.noise_strength = nn.Parameter(torch.zeros([]))
            self.register_buffer("noise_const", torch.randn(noise_res, noise_res))
        else:
            self.noise_strength = None

    def forward(self, x, w, noise="random"):
        s = self.affine(w).float()
        weight = self.weight * self.scale
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x * s.to(x.dtype)[:, :, None, None]
        x = F.conv2d(x, weight, padding=weight.shape[-1] // 2)
        if self.demod:
            d = torch.rsqrt((weight.float().square()[None] * s.square()[:, None, :, None, None]).sum([2, 3, 4]) + 1e-8)
            x = x * d.to(x.dtype)[:, :, None, None]
        if self.noise_strength is not None and noise != "none":
            if noise == "const":
                n = self.noise_const[None, None]
            else:
                n = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            x = x + (n * self.noise_strength).to(x.dtype)
        return bias_act(x, self.bias, self.act)


class Mapping(nn.Module):
    def __init__(self, layers=8):
        super().__init__()
        dims = [Z_DIM] + [W_DIM] * layers
        self.net = nn.Sequential(*(EqLinear(a, b, lr_mul=0.01, act=True) for a, b in zip(dims, dims[1:])))
        self.register_buffer("w_avg", torch.zeros(W_DIM))

    def forward(self, z, track=False):
        z = z * torch.rsqrt(z.square().mean(1, keepdim=True) + 1e-8)
        w = self.net(z)
        if track:
            self.w_avg.copy_(w.detach().float().mean(0).lerp(self.w_avg, 0.995))
        return w


class Generator(nn.Module):
    def __init__(self, res=256):
        super().__init__()
        self.res = res
        self.n_blocks = int(math.log2(res)) - 1  # one block per resolution, 4 -> res
        self.num_ws = 2 * self.n_blocks
        self.mapping = Mapping()
        c4 = channels(4)
        self.const = nn.Parameter(torch.randn(c4, 4, 4))
        self.convs = nn.ModuleList()
        self.torgbs = nn.ModuleList()
        cin = c4
        for i in range(self.n_blocks):
            r = 4 * 2**i
            cout = channels(r)
            if i > 0:
                self.convs.append(ModConv(cin, cout, 3, up=True, noise_res=r))
            self.convs.append(ModConv(cout, cout, 3, noise_res=r))
            self.torgbs.append(ModConv(cout, 3, 1, demod=False, act=False))
            cin = cout

    def styles(self, z, psi=1.0, mix_prob=0.0, track=False):
        w = self.mapping(z, track=track)
        ws = w[:, None].repeat(1, self.num_ws, 1)
        if mix_prob > 0 and torch.rand([]) < mix_prob:
            cut = int(torch.randint(1, self.num_ws, []))
            ws[:, cut:] = self.mapping(torch.randn_like(z))[:, None]
        if psi != 1:
            ws = self.mapping.w_avg.lerp(ws, psi)
        return ws

    def synthesis(self, ws, noise="random"):
        x = self.const[None].repeat(ws.shape[0], 1, 1, 1)
        img = None
        k = 0
        convs = iter(self.convs)
        for i, torgb in enumerate(self.torgbs):
            if i > 0:
                x = next(convs)(x, ws[:, k], noise)
                k += 1
            x = next(convs)(x, ws[:, k], noise)
            y = torgb(x, ws[:, k + 1]).float()
            img = y if img is None else F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False) + y
            k += 1
        return img

    def forward(self, z, psi=1.0, mix_prob=0.0, track=False, noise="random"):
        return self.synthesis(self.styles(z, psi, mix_prob, track), noise)


class DBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv0 = EqConv(cin, cin, 3)
        self.conv1 = EqConv(cin, cout, 3, down=True)
        self.skip = EqConv(cin, cout, 1, bias=False, act=False, down=True)

    def forward(self, x):
        return (self.skip(x) + self.conv1(self.conv0(x))) * (1 / math.sqrt(2))


class Discriminator(nn.Module):
    def __init__(self, res=256):
        super().__init__()
        self.fromrgb = EqConv(3, channels(res), 1)
        blocks = []
        r = res
        while r > 4:
            blocks.append(DBlock(channels(r), channels(r // 2)))
            r //= 2
        self.blocks = nn.Sequential(*blocks)
        c = channels(4)
        self.conv = EqConv(c + 1, c, 3)
        self.fc = EqLinear(c * 16, c, act=True)
        self.out = EqLinear(c, 1)

    def forward(self, img):
        x = self.blocks(self.fromrgb(img))
        # minibatch standard deviation
        b, c, h, w = x.shape
        g = min(4, b)
        while b % g:
            g -= 1
        y = x.float().reshape(g, -1, c, h, w)
        y = (y.var(0, unbiased=False) + 1e-8).sqrt().mean([1, 2, 3])
        y = y.reshape(-1, 1, 1, 1).repeat(g, 1, h, w).to(x.dtype)
        x = self.conv(torch.cat([x, y], 1))
        return self.out(self.fc(x.flatten(1).float())).view(-1)


def to_uint8(img):
    """(B, 3, H, W) in [-1, 1] -> list of (H, W, 3) uint8 arrays."""
    img = (img.float().clamp(-1, 1) + 1) * 127.5
    return list(img.round().byte().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
