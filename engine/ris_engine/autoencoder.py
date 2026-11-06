# Image autoencoder: 8x VAE + 2x2 patchify = 16x; latents are 128-channel tokens with (t, h, w, l) ids.
#
# Portions derived from black-forest-labs/flux-3-image-klein (src/flux3/autoencoder.py),
# Copyright Black Forest Labs, licensed under the Apache License 2.0; see NOTICE.

import torch
import torch.nn.functional as F
from torch import nn

EMBED_DIM = 32
CH = 128
CH_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128


def swish(x):
    return x * torch.sigmoid(x)


def normalize(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, height, width = h.shape
        flat = lambda t: t.reshape(b, c, height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
        h = F.scaled_dot_product_attention(flat(self.q(h)), flat(self.k(h)), flat(self.v(h)))
        return x + self.proj_out(h.squeeze(1).permute(0, 2, 1).reshape(b, c, height, width))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * EMBED_DIM, 2 * EMBED_DIM, 1)
        self.conv_in = nn.Conv2d(3, CH, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = CH
        for level, mult in enumerate(CH_MULT):
            down = nn.Module()
            down.block = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS):
                down.block.append(ResnetBlock(block_in, CH * mult))
                block_in = CH * mult
            if level != len(CH_MULT) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * EMBED_DIM, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level, down in enumerate(self.down):
            for block in down.block:
                h = block(h)
            if level != len(CH_MULT) - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))

    def forward_low_memory(self, x):
        """forward() in horizontal strips down to the middle blocks, like
        Decoder.forward_low_memory (about 0.6 GB per megapixel instead of 2)."""
        rows = ConvRows(self.conv_in, Rows(x))
        for level, down in enumerate(self.down):
            for block in down.block:
                rows = Rows(block_in_strips(block, rows))
            if level != len(CH_MULT) - 1:
                rows = Rows(downsample_in_strips(down.downsample.conv, rows))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(rows.tensor)))
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, EMBED_DIM, 1)
        block_in = CH * CH_MULT[-1]
        self.conv_in = nn.Conv2d(EMBED_DIM, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.up = nn.ModuleList()
        for level in reversed(range(len(CH_MULT))):
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS + 1):
                up.block.append(ResnetBlock(block_in, CH * CH_MULT[level]))
                block_in = CH * CH_MULT[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(CH_MULT))):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))

    def forward_low_memory(self, z):
        """forward() for GPUs without room for the full-resolution activations (about 2 GB per
        megapixel): after the middle blocks every convolution runs in horizontal strips, each
        upsampled tensor is recomputed strip by strip instead of being stored, and GroupNorm
        statistics are gathered over the whole image, so the result matches forward() up to
        rounding (there is no tiling and so no seams).  Needs about 0.6 GB per megapixel."""
        h = self.conv_in(self.post_quant_conv(z))
        rows = Rows(self.mid.block_2(self.mid.attn_1(self.mid.block_1(h))))
        for level in reversed(range(len(CH_MULT))):
            for block in self.up[level].block:
                rows = Rows(block_in_strips(block, rows))
            if level != 0:
                rows = UpsampledRows(self.up[level].upsample.conv, rows)
        x = rows.tensor
        out = x.new_empty(1, self.conv_out.out_channels, *x.shape[2:])
        conv_in_strips(self.conv_out, NormedRows(self.norm_out, rows), out)
        return out


# --- strip-wise evaluation for forward_low_memory ---------------------------
STRIP_BYTES = 1 << 28   # working memory per strip


def strip_height(channels, width):
    return max(8, STRIP_BYTES // (channels * width * 16))


class Rows:
    """A stored (1, C, H, W) tensor, read in row ranges."""

    def __init__(self, tensor):
        self.tensor = tensor
        _, self.channels, self.height, self.width = tensor.shape

    def get(self, r0, r1):
        return self.tensor[:, :, r0:r1]


class UpsampledRows(Rows):
    """conv(2x nearest upsample(source)), computed on demand for each row range."""

    def __init__(self, conv, source):
        self.conv, self.source = conv, source
        self.channels, self.height, self.width = conv.out_channels, 2 * source.height, 2 * source.width

    def get(self, r0, r1):
        a, b = max(r0 - 1, 0), min(r1 + 1, self.height)   # upsampled rows the conv reads
        x = self.source.get(a // 2, (b + 1) // 2)
        x = x.repeat_interleave(2, dim=2)[:, :, a - a // 2 * 2:][:, :, :b - a].repeat_interleave(2, dim=3)
        x = F.pad(x, (0, 0, int(r0 == 0), int(r1 == self.height)))
        return F.conv2d(x, self.conv.weight, self.conv.bias, padding=(0, 1))


class ConvRows(Rows):
    """conv(source) for a 3x3, padding 1 conv, computed on demand for each row range."""

    def __init__(self, conv, source):
        self.conv, self.source = conv, source
        self.channels, self.height, self.width = conv.out_channels, source.height, source.width

    def get(self, r0, r1):
        x = self.source.get(max(r0 - 1, 0), min(r1 + 1, self.height))
        x = F.pad(x, (0, 0, int(r0 == 0), int(r1 == self.height)))
        return F.conv2d(x, self.conv.weight, self.conv.bias, padding=(0, 1))


def downsample_in_strips(conv, source):
    """Downsample: conv(pad(source, right and bottom by 1)), 3x3 with stride 2, in strips."""
    height, width = source.height // 2, source.width // 2
    out = torch.empty((1, conv.out_channels, height, width), dtype=conv.weight.dtype, device=conv.weight.device)
    step = max(1, strip_height(source.channels, source.width) // 2)
    for r0 in range(0, height, step):
        r1 = min(r0 + step, height)
        x = source.get(2 * r0, min(2 * r1 + 1, source.height))   # output row r reads input rows 2r..2r+2
        x = F.pad(x, (0, 1, 0, 2 * r1 + 1 - min(2 * r1 + 1, source.height)))
        out[:, :, r0:r1] = F.conv2d(x, conv.weight, conv.bias, stride=2)
    return out


class NormedRows(Rows):
    """swish(GroupNorm(source)) with statistics over the whole source."""

    def __init__(self, norm, source):
        self.norm, self.source = norm, source
        self.channels, self.height, self.width = source.channels, source.height, source.width
        groups = norm.num_groups
        count, mean, m2 = 0, 0.0, 0.0   # Chan et al.'s parallel mean/variance, per group
        step = strip_height(self.channels, self.width)
        for r0 in range(0, self.height, step):
            x = source.get(r0, min(r0 + step, self.height)).float().reshape(groups, -1)
            n = x.shape[1]
            var, m = torch.var_mean(x, dim=1, unbiased=False)
            delta = m - mean
            mean = mean + delta * n / (count + n)
            m2 = m2 + var * n + delta * delta * count * n / (count + n)
            count += n
        rstd = torch.rsqrt(m2 / count + norm.eps)
        per_channel = lambda t: t.repeat_interleave(self.channels // groups).view(1, -1, 1, 1)
        self.scale = per_channel(rstd) * norm.weight.float().view(1, -1, 1, 1)
        self.shift = norm.bias.float().view(1, -1, 1, 1) - per_channel(mean) * self.scale

    def get(self, r0, r1):
        return self.apply(self.source.get(r0, r1))

    def apply(self, x):
        return F.silu(x.float() * self.scale + self.shift).to(self.norm.weight.dtype)


def conv_in_strips(conv, source, out, add=None):
    """out = conv(source) (+ add), for a 3x3, padding 1 conv, in strips of rows."""
    step = strip_height(source.channels, source.width)
    for r0 in range(0, source.height, step):
        r1 = min(r0 + step, source.height)
        x = source.get(max(r0 - 1, 0), min(r1 + 1, source.height))
        x = F.pad(x, (0, 0, int(r0 == 0), int(r1 == source.height)))
        y = F.conv2d(x, conv.weight, conv.bias, padding=(0, 1))
        if add is not None:
            y += add(r0, r1)
        out[:, :, r0:r1] = y


def conv_in_place(conv, norm, x, add):
    """x = conv(swish(norm(x))) + add(rows), for a 3x3, padding 1 conv, overwriting x strip by
    strip; each strip keeps a copy of the row above it from before that row was overwritten."""
    normed = NormedRows(norm, Rows(x))
    height = x.shape[2]
    step = strip_height(x.shape[1], x.shape[3])
    above = None
    for r0 in range(0, height, step):
        r1 = min(r0 + step, height)
        rows = x[:, :, r0:min(r1 + 1, height)]
        if above is not None:
            rows = torch.cat([above, rows], dim=2)
        rows = F.pad(normed.apply(rows), (0, 0, int(r0 == 0), int(r1 == height)))
        y = F.conv2d(rows, conv.weight, conv.bias, padding=(0, 1))
        y += add(r0, r1)
        above = x[:, :, r1 - 1:r1].clone()
        x[:, :, r0:r1] = y


def block_in_strips(block, rows):
    """ResnetBlock.forward on Rows, holding only one new tensor: the hidden one, which becomes
    the output."""
    hidden = torch.empty((1, block.out_channels, rows.height, rows.width),
                         dtype=block.conv1.weight.dtype, device=block.conv1.weight.device)
    conv_in_strips(block.conv1, NormedRows(block.norm1, rows), hidden)
    if block.in_channels != block.out_channels:
        shortcut = lambda r0, r1: block.nin_shortcut(rows.get(r0, r1))
    else:
        shortcut = rows.get
    conv_in_place(block.conv2, block.norm2, hidden, shortcut)
    return hidden


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn_eps = 1e-4
        self.bn = nn.BatchNorm2d(LATENT_CHANNELS, eps=self.bn_eps, affine=False, track_running_stats=True)

    @torch.inference_mode()
    def encode(self, image):
        """image (3, H, W) in [-1, 1] -> tokens (h*w, 128), plus h and w."""
        image = image.unsqueeze(0).to(self.encoder.conv_in.weight.dtype)
        try:
            moments = self.encoder(image)
        except torch.OutOfMemoryError:
            moments = None   # retried outside the handler, so the failed attempt's tensors are freed
        if moments is None:
            torch.cuda.empty_cache()
            moments = self.encoder.forward_low_memory(image)
        mean, _ = moments.chunk(2, dim=1)
        mean = mean.float()  # the latent statistics stay fp32
        z = mean.reshape(1, EMBED_DIM, mean.shape[2] // 2, 2, mean.shape[3] // 2, 2).permute(0, 1, 3, 5, 2, 4)
        z = self.bn(z.reshape(1, LATENT_CHANNELS, mean.shape[2] // 2, mean.shape[3] // 2))
        h, w = z.shape[2:]
        return z[0].permute(1, 2, 0).reshape(h * w, LATENT_CHANNELS), h, w

    @torch.inference_mode()
    def decode(self, tokens, h, w):
        """tokens (h*w, 128) in row-major order -> image (3, H, W) in [-1, 1]."""
        z = tokens.float().reshape(h, w, LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
        z = z * torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps) + self.bn.running_mean.view(1, -1, 1, 1)
        z = z.reshape(1, EMBED_DIM, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, EMBED_DIM, 2 * h, 2 * w)
        z = z.to(self.decoder.conv_in.weight.dtype)
        try:
            return self.decoder(z)[0]
        except torch.OutOfMemoryError:
            pass   # outside the handler, so the failed attempt's tensors are freed
        torch.cuda.empty_cache()
        return self.decoder.forward_low_memory(z)[0]


def latent_ids(h, w, device="cpu"):
    """(h*w, 4) ids = (0, row, col, 0), row-major."""
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)
