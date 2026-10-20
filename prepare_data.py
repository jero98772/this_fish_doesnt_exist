import hashlib
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image, ImageFilter

SIZE = int(os.environ.get("RES", 256))
FV_COUNT = int(os.environ.get("FV_COUNT", 0))  # 0 = every fish-vista image
LSF = os.environ.get("LSF", "1") == "1"  # include the Large-Scale Fish dataset
BG = 245  # every background is normalized to this grey level
BAR = np.array([124, 116, 103])  # letterbox colour in phylo-fish
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def fish_vista_paths():
    repo = "imageomics/fish-vista"
    names = [
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith("Images/")
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "_dor_" not in f
        and "_ven" not in f
    ]
    if not FV_COUNT:
        root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["Images/*"], max_workers=16))
        return sorted(str(root / n) for n in names if (root / n).exists())
    random.Random(0).shuffle(names)
    names = names[:FV_COUNT]

    def get(name):
        try:
            return hf_hub_download(repo, name, repo_type="dataset")
        except Exception:
            return None

    with ThreadPoolExecutor(16) as ex:
        return [p for p in ex.map(get, names) if p]


def phylo_paths():
    root = Path(
        snapshot_download(
            repo_id="imageomics/phylo-fish",
            repo_type="dataset",
            allow_patterns=["train/*", "test/*"],
        )
    )
    return sorted(
        str(p) for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def large_scale_fish_shards():
    # Ulucan et al. 2020, fish on a white tray with segmentation masks (CC-BY-NC-4.0)
    root = Path(
        snapshot_download(
            repo_id="FriedParrot/a-large-scale-fish-dataset",
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
            max_workers=8,
        )
    )
    return sorted(str(p) for p in root.glob("data/*.parquet"))


def clean_shard(path):
    """Cut every fish out with its mask, paste it on the background grey, then clean as usual."""
    out = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=32, columns=["image", "mask", "class_name"]):
        for row in batch.to_pylist():
            if row["class_name"].lower().startswith("shrimp"):
                continue
            try:
                a = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"), dtype=np.float32)
            except Exception:
                continue
            m = np.asarray(row["mask"], dtype=np.float32)
            if m.shape != a.shape[:2]:
                m = m.T if m.T.shape == a.shape[:2] else np.asarray(
                    Image.fromarray((m > 0).astype(np.uint8) * 255).resize(a.shape[1::-1]), np.float32)
            # shrink the mask a little: its edge carries a halo of the blue tray
            m = Image.fromarray((m > 0).astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))
            m = (np.asarray(m) > 0).astype(np.float32)[..., None]
            arr = clean_array((a * m + BG * (1 - m)).astype(np.int16))
            if arr is not None:
                out.append(arr)
    return out


def drop_letterbox(a):
    # phylo-fish pads every image with flat grey bars above and below the photo
    rows = np.abs(np.median(a, axis=1) - BAR).max(axis=1) > 12
    if rows.all() or rows.sum() <= 16:
        return a
    # the strip's outer pixels carry a thin dark edge line
    return a[rows][3:-3, 4:-4]


def clean(path):
    try:
        im = Image.open(path)
        im.load()
        a = np.asarray(im.convert("RGB"), dtype=np.int16)
    except Exception:
        return None
    return clean_array(drop_letterbox(a))


def clean_array(a):
    h, w, _ = a.shape
    if min(h, w) < 32:
        return None

    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                             a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.abs(a - bg).max(axis=2) > 40

    # bbox from rows/cols that hold a real share of foreground (ignores specks)
    cols = np.where(fg.sum(0) > max(2, 0.02 * h))[0]
    rows = np.where(fg.sum(1) > max(2, 0.02 * w))[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 64 or not 1.3 <= bw / max(bh, 1) <= 6:
        return None
    m = int(0.06 * bw)
    x0, x1, y0, y1 = max(0, x0 - m), min(w, x1 + m), max(0, y0 - m), min(h, y1 + m)
    crop = a[y0:y1, x0:x1].astype(np.float32)

    # map the background colour to near-white, then flatten what is left of it
    # (shading, strip edges) with a soft mask so pale fins survive
    crop = np.clip(crop * (BG / np.maximum(bg, 1)), 0, 255)
    alpha = np.clip((np.abs(crop - BG).max(axis=2, keepdims=True) - 15) / 25, 0, 1)
    crop = (alpha * crop + (1 - alpha) * BG).astype(np.uint8)
    if crop.std() < 8:
        return None
    ch, cw, _ = crop.shape
    side = max(ch, cw)
    canvas = np.full((side, side, 3), BG, dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw] = crop
    return np.asarray(Image.fromarray(canvas).resize((SIZE, SIZE), Image.LANCZOS))


def main():
    paths = fish_vista_paths() + phylo_paths()
    shards = large_scale_fish_shards() if LSF else []
    print(len(paths), "source images +", len(shards), "Large-Scale Fish shards", flush=True)

    seen = set()
    images = []

    def keep(arr):
        if arr is not None:
            digest = hashlib.md5(arr.tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                images.append(arr)

    with Pool(max(1, (os.cpu_count() or 2) - 4)) as pool:
        for i, arr in enumerate(pool.imap(clean, paths, chunksize=16)):
            keep(arr)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(paths)} processed, {len(images)} kept", flush=True)
        for i, arrs in enumerate(pool.imap_unordered(clean_shard, shards)):
            for arr in arrs:
                keep(arr)
            print(f"  shard {i + 1}/{len(shards)}, {len(images)} kept", flush=True)

    out = OUT / f"fish{SIZE}.npy"
    data = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.uint8, shape=(len(images), SIZE, SIZE, 3)
    )
    for i, arr in enumerate(images):
        data[i] = arr
    data.flush()

    pick = random.Random(1).sample(range(len(images)), min(64, len(images)))
    sheet = Image.new("RGB", (8 * 128, 8 * 128))
    for k, j in enumerate(pick):
        sheet.paste(Image.fromarray(images[j]).resize((128, 128)), ((k % 8) * 128, (k // 8) * 128))
    sheet.save(OUT / "preview.jpg", quality=90)
    print(len(images), "images kept ->", out)


if __name__ == "__main__":
    main()
