"""Sample fish from a trained checkpoint.

    python generate.py --n 16              # 16 single PNGs
    python generate.py --grid 8            # one 8x8 sheet
    python generate.py --interp 120        # latent walk GIF with 120 frames
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from model import Z_DIM, Generator, to_uint8

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoint.pt")
p.add_argument("--n", type=int, default=1, help="number of fish to save")
p.add_argument("--psi", type=float, default=0.7, help="truncation: lower = cleaner, less varied")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--grid", type=int, default=0, help="save one grid x grid sheet instead")
p.add_argument("--interp", type=int, default=0, help="save a latent-walk GIF with this many frames")
p.add_argument("--out", default="generated")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=dev)
G = Generator(ck["res"]).to(dev).eval()
G.load_state_dict(ck["G_ema"])
out = Path(args.out)
out.mkdir(exist_ok=True)
seed = args.seed if args.seed is not None else int(time.time())
rng = torch.Generator().manual_seed(seed)


@torch.no_grad()
def render(z):
    return [im for zz in z.split(16) for im in to_uint8(G(zz.to(dev), psi=args.psi, noise="const"))]


if args.interp:
    keys = torch.randn(max(2, args.interp // 30) + 1, Z_DIM, generator=rng)
    keys[-1] = keys[0]  # loop back to the start
    ws = G.styles(keys.to(dev), psi=args.psi)
    t = torch.linspace(0, len(keys) - 1, args.interp + 1)[:-1]
    i, f = t.long(), (t - t.floor()).to(dev)[:, None, None]
    f = (1 - torch.cos(f * torch.pi)) / 2  # ease in/out between keyframes
    with torch.no_grad():
        frames = [im for chunk in ws[i].lerp(ws[i + 1], f).split(16) for im in to_uint8(G.synthesis(chunk, "const"))]
    path = out / f"walk_{seed}.gif"
    Image.fromarray(frames[0]).save(path, save_all=True, append_images=[Image.fromarray(x) for x in frames[1:]],
                                    duration=40, loop=0)
elif args.grid:
    ims = render(torch.randn(args.grid**2, Z_DIM, generator=rng))
    r = ck["res"]
    sheet = Image.new("RGB", (args.grid * r, args.grid * r))
    for k, im in enumerate(ims):
        sheet.paste(Image.fromarray(im), ((k % args.grid) * r, (k // args.grid) * r))
    path = out / f"grid_{seed}.png"
    sheet.save(path)
else:
    for k, im in enumerate(render(torch.randn(args.n, Z_DIM, generator=rng))):
        path = out / f"fish_{seed}_{k:03d}.png"
        Image.fromarray(im).save(path)
print("saved", path if not args.n > 1 or args.grid or args.interp else f"{args.n} fish to {out}/")
