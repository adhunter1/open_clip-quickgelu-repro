# open_clip QuickGELU README check

Reproduction and verification for [mlfoundations/open_clip#1254](https://github.com/mlfoundations/open_clip/pull/1254),
which corrects the open_clip README note claiming that OpenAI pretrained weights "will always default to QuickGELU."

Since open_clip commit [`84f7d2f`](https://github.com/mlfoundations/open_clip/commit/84f7d2f7cf2b2235a7b61bb3be1f919bda4493e6)
(Oct 2024), `create_model_and_transforms('ViT-B-32', pretrained='openai')` builds the model with `nn.GELU`
and only emits a `QuickGELU mismatch` warning. See also issue [#1109](https://github.com/mlfoundations/open_clip/issues/1109).

## Results

CPU, torch 2.15.0, OpenAI ViT-B-32 weights, zero-shot on the first 1,000 CIFAR-10 test images
(prompt `"a photo of a {class}."`). Identical on open_clip 3.3.0 and 4.2.2.

| Load | Activation | Warning | Zero-shot accuracy |
| --- | --- | --- | --- |
| `ViT-B-32`, `pretrained='openai'` (original README guidance) | GELU | QuickGELU mismatch | 85.0% |
| `ViT-B-32-quickgelu`, `pretrained='openai'` (corrected README snippet) | QuickGELU | none | 89.5% |

Image-embedding cosine similarity between the two loads: mean 0.936, min 0.834.
Raw outputs are in [`results/`](results/).

## Files

| File | What it does |
| --- | --- |
| `repro_quickgelu_readme.py` | Loads OpenAI ViT-B-32 both ways; prints activation, warnings, zero-shot CIFAR-10 accuracy, embedding similarity |
| `verify_readme_fix.py` | Extracts the Python snippet from the fixed README and runs it exactly as written, plus the `force_quick_gelu=True` alternative |
| `results/result_open_clip_3.3.0.txt` | `repro_quickgelu_readme.py` output on open_clip 3.3.0 (before) |
| `results/result_open_clip_4.2.2.txt` | `repro_quickgelu_readme.py` output on open_clip 4.2.2 (before) |
| `results/result_after_fix_4.2.2.txt` | `verify_readme_fix.py` output on the fixed branch (after) |

## Reproduce

```bash
pip install open_clip_torch torchvision
python repro_quickgelu_readme.py --n 1000 --data ./cifar10   # downloads ~600 MB weights + ~170 MB CIFAR-10

git clone -b docs/openai-quickgelu-note https://github.com/adhunter1/open_clip.git
python verify_readme_fix.py --readme open_clip/README.md
```

Expected if the original README were correct: same activation, identical accuracy, cosine similarity 1.0000.

AI assistance (Claude) was used to write these scripts; results were produced by running them locally.
