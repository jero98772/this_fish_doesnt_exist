# This Fish Doesn't Exist

A StyleGAN2 trained on ~80K side-view fish photos, trained on all local GPUs in parallel.

## Data
`prepare_data.py` downloads, cleans (crop to the fish, flat grey background, square, resize) and deduplicates:

| Dataset | Images | License |
|---|---|---|
| [imageomics/fish-vista](https://huggingface.co/datasets/imageomics/fish-vista) | ~60K museum photos, 1,900 species | see dataset card |
| [imageomics/phylo-fish](https://huggingface.co/datasets/imageomics/phylo-fish) | ~5K | CC0 |
| [Large-Scale Fish Dataset](https://huggingface.co/datasets/FriedParrot/a-large-scale-fish-dataset) (Ulucan et al. 2020) | ~8K, cut out with their masks | CC-BY-NC-4.0 |

## Run
```bash
python -m venv env && . env/bin/activate
pip install -r requirements.txt

RES=256 python prepare_data.py                    # -> data/fish256.npy, check data/preview.jpg
RES=256 torchrun --nproc_per_node=2 train.py      # one process per GPU; Ctrl-C and rerun to resume
python generate.py --n 16                         # -> generated/fish_*.png
python generate.py --grid 8                       # 8x8 sheet
python generate.py --interp 240                   # looping latent-walk GIF
```
Progress grids land in `samples/kimg_*.png` every 50 kimg; `checkpoint.pt` is rewritten at the same time.

## Settings (environment variables for `train.py`)
| Var | Default | Meaning |
|---|---|---|
| `RES` | 256 | resolution (must match the prepared data) |
| `BATCH` | 32 | global batch across all GPUs |
| `BATCH_GPU` | 8 | images per forward pass; lower it on OOM |
| `KIMG` | 10000 | training length in thousands of images shown |
| `GAMMA` | 1 (256px) / 4 (512px) | R1 regularization strength |
| `KIMG_SNAP` | 50 | snapshot interval |
| `AMP` | 1 | fp16 mixed precision |

Use `--psi` in `generate.py` (default 0.7) to trade variety (1.0) for cleaner fish (0.5).
For 512px: `RES=512 python prepare_data.py`, then `RES=512 BATCH_GPU=4 torchrun --nproc_per_node=2 train.py` (~4x slower).
