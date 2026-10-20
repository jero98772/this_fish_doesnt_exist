"""Multi-GPU StyleGAN2 training on the fish dataset.

    torchrun --nproc_per_node=2 train.py      # both GPUs
    python train.py                           # single GPU / CPU
"""
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.utils import save_image

from model import Z_DIM, Discriminator, Generator

RES = int(os.environ.get("RES", 256))
DATA = os.environ.get("DATA", f"data/fish{RES}.npy")
BATCH = int(os.environ.get("BATCH", 32))  # global batch, split across GPUs
BATCH_GPU = int(os.environ.get("BATCH_GPU", 8))  # images per forward pass; gradients accumulate up to BATCH
KIMG = int(os.environ.get("KIMG", 10000))  # total training length in thousands of real images
KIMG_SNAP = int(os.environ.get("KIMG_SNAP", 50))
GAMMA = float(os.environ.get("GAMMA", 1.0 if RES <= 256 else 4.0))  # R1 weight
LR = float(os.environ.get("LR", 0.0025))
AMP = os.environ.get("AMP", "1") == "1"
WORKERS = int(os.environ.get("WORKERS", 8))
CKPT = os.environ.get("CKPT", "checkpoint.pt")
R1_EVERY = 16
EMA_KIMG = 10
MIX_PROB = 0.9


class FishNpy(Dataset):
    """uint8 (N, RES, RES, 3) memory-mapped array, opened lazily in each loader worker."""

    def __init__(self, path):
        self.path = path
        self.n = len(np.load(path, mmap_mode="r"))
        self.data = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.data is None:
            self.data = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self.data[i])).permute(2, 0, 1)


# DiffAugment (Zhao et al. 2020): differentiable augmentations applied to everything D sees,
# so D cannot memorise the training set and G still learns the un-augmented distribution.
def diff_augment(x):
    b = x.shape[0]
    dev = x.device
    # colour
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    mean = x.mean(1, keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) * 2) + mean
    mean = x.mean([1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) + 0.5) + mean
    # translation by up to 1/8 of the image
    h, w = x.shape[2:]
    sh, sw = h // 8, w // 8
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=dev)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=dev), torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij"
    )
    gx = (gx + tx + 1).clamp(0, h + 1)
    gy = (gy + ty + 1).clamp(0, w + 1)
    x = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1)[gb, gx, gy].permute(0, 3, 1, 2)
    # cutout of half the image size
    ch, cw = h // 2, w // 2
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=dev), torch.arange(ch, device=dev), torch.arange(cw, device=dev), indexing="ij"
    )
    gx = (gx + ox - ch // 2).clamp(0, h - 1)
    gy = (gy + oy - cw // 2).clamp(0, w - 1)
    mask = torch.ones(b, h, w, device=dev, dtype=x.dtype)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


def all_reduce_grads(model, world, sanitize=False):
    # one flat all-reduce per step; inf/nan from fp16 overflow reach every rank, so all GradScalers skip together
    params = [p for p in model.parameters() if p.grad is not None]
    if (world > 1 or sanitize) and params:
        flat = torch.cat([p.grad.flatten() for p in params])
        if world > 1:
            dist.all_reduce(flat)
            flat /= world
        if sanitize:
            torch.nan_to_num_(flat, nan=0, posinf=1e5, neginf=-1e5)
        for p, g in zip(params, flat.split([p.numel() for p in params])):
            p.grad.copy_(g.view_as(p))


def infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def main():
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        dev = torch.device("cuda", local_rank)
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", device_id=dev)
    lead = rank == 0
    amp = AMP and dev.type == "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1234 + rank)

    local = BATCH // world
    gpu = min(BATCH_GPU, local)
    rounds = local // gpu
    ds = FishNpy(DATA)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(
        ds, local, shuffle=sampler is None, sampler=sampler, num_workers=WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=WORKERS > 0,
    )
    batches = infinite(loader, sampler)

    G = Generator(RES).to(dev)
    D = Discriminator(RES).to(dev)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    c = R1_EVERY / (R1_EVERY + 1)  # lazy regularization correction for D's optimizer
    optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.99), eps=1e-8)
    optD = torch.optim.Adam(D.parameters(), lr=LR * c, betas=(0.0, 0.99**c), eps=1e-8)
    scalerG = torch.amp.GradScaler("cuda", enabled=amp)
    scalerD = torch.amp.GradScaler("cuda", enabled=amp)

    nimg = 0
    if Path(CKPT).exists():
        ck = torch.load(CKPT, map_location=dev)
        if ck["res"] != RES:
            raise SystemExit(f"{CKPT} is for RES={ck['res']}, not {RES}; move it away or set RES")
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        G_ema.load_state_dict(ck["G_ema"])
        optG.load_state_dict(ck["optG"])
        optD.load_state_dict(ck["optD"])
        scalerG.load_state_dict(ck["scalerG"])
        scalerD.load_state_dict(ck["scalerD"])
        nimg = ck["nimg"]
    if world > 1:
        for t in [*G.state_dict().values(), *D.state_dict().values(), *G_ema.state_dict().values()]:
            dist.broadcast(t, 0)

    if lead:
        Path("samples").mkdir(exist_ok=True)
        n_g = sum(p.numel() for p in G.parameters()) / 1e6
        n_d = sum(p.numel() for p in D.parameters()) / 1e6
        print(f"{len(ds)} images at {RES}px | {world} GPU(s) x {rounds} x {gpu} imgs | G {n_g:.1f}M D {n_d:.1f}M params"
              f" | AMP {amp} | start at {nimg / 1000:.0f} kimg", flush=True)
    fixed = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(0)).to(dev)

    def load_real():
        x = next(batches).to(dev, non_blocking=True).float() / 127.5 - 1
        flip = torch.rand(x.shape[0], 1, 1, 1, device=dev) < 0.5
        return torch.where(flip, x.flip(3), x)

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16, enabled=amp)

    step = nimg // BATCH
    next_snap = (nimg // (KIMG_SNAP * 1000) + 1) * KIMG_SNAP * 1000
    t_start, nimg_start = time.time(), nimg
    while nimg < KIMG * 1000:
        # --- generator: non-saturating loss ---
        D.requires_grad_(False)
        optG.zero_grad(set_to_none=True)
        for _ in range(rounds):
            z = torch.randn(gpu, Z_DIM, device=dev)
            with autocast():
                fake = G(z, mix_prob=MIX_PROB, track=True)
                g_loss = F.softplus(-D(diff_augment(fake))).mean()
            scalerG.scale(g_loss / rounds).backward()
        all_reduce_grads(G, world)
        scalerG.step(optG)
        scalerG.update()
        D.requires_grad_(True)

        # --- discriminator: logistic loss ---
        reals = load_real().split(gpu)
        optD.zero_grad(set_to_none=True)
        for real in reals:
            with torch.no_grad(), autocast():
                fake = G(torch.randn(gpu, Z_DIM, device=dev), mix_prob=MIX_PROB)
            with autocast():
                d_fake = D(diff_augment(fake))
                d_real = D(diff_augment(real))
                d_loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
            scalerD.scale(d_loss / rounds).backward()
        all_reduce_grads(D, world)
        scalerD.step(optD)
        scalerD.update()

        # --- lazy R1 penalty on reals, in fp32 ---
        r1 = None
        if step % R1_EVERY == 0:
            optD.zero_grad(set_to_none=True)
            for real in reals:
                real = real.detach().requires_grad_(True)
                out = D(diff_augment(real))
                (grad,) = torch.autograd.grad(out.sum(), real, create_graph=True)
                r1 = grad.square().sum([1, 2, 3]).mean()
                ((r1 * (GAMMA / 2) * R1_EVERY + out.mean() * 0) / rounds).backward()
            all_reduce_grads(D, world, sanitize=True)
            optD.step()

        # --- EMA of the generator, used for all samples ---
        nimg += BATCH
        step += 1
        ema_nimg = min(EMA_KIMG * 1000, nimg * 0.05)
        beta = 0.5 ** (BATCH / max(ema_nimg, 1e-8))
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.copy_(p.lerp(pe, beta))
            for be, b in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b)

        if lead and step % 100 == 0:
            speed = (time.time() - t_start) / max(nimg - nimg_start, 1) * 1000
            r1_txt = f"  R1 {r1.item():.3f}" if r1 is not None else ""
            print(f"kimg {nimg / 1000:8.1f}  G {g_loss.item():.3f}  D {d_loss.item():.3f}{r1_txt}"
                  f"  {speed:.1f} s/kimg  mem {torch.cuda.max_memory_allocated(dev) / 2**30 if amp else 0:.1f}GB",
                  flush=True)

        if nimg >= next_snap or nimg >= KIMG * 1000:
            next_snap += KIMG_SNAP * 1000
            if lead:
                with torch.no_grad():
                    imgs = torch.cat([G_ema(z, noise="const") for z in fixed.split(16)])
                save_image(imgs, f"samples/kimg_{nimg // 1000:05d}.png", nrow=8, normalize=True, value_range=(-1, 1))
                tmp = CKPT + ".tmp"
                torch.save({
                    "res": RES, "nimg": nimg,
                    "G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "scalerG": scalerG.state_dict(), "scalerD": scalerD.state_dict(),
                }, tmp)
                os.replace(tmp, CKPT)
                print(f"snapshot at {nimg / 1000:.0f} kimg -> samples/, {CKPT}", flush=True)
            if world > 1:
                dist.barrier()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
