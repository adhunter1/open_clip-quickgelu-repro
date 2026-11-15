"""Check the open_clip README claim: "All OpenAI pretrained weights will always default to QuickGELU."

Compares two ways of loading OpenAI CLIP ViT-B-32 weights:
  (1) as the README note implies is safe:  create_model_and_transforms("ViT-B-32", pretrained="openai")
  (2) explicit QuickGELU variant:           create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")

Reports, for each: activation class actually used, any warnings, and zero-shot accuracy
on the first N CIFAR-10 test images. Also reports cosine similarity between the two
models' image embeddings. CPU-only, deterministic.

Usage: python repro_quickgelu_readme.py [--n 1000] [--data DIR]
"""
import argparse
import warnings

import open_clip
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load(model_name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval()
    acts = sorted({type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__})
    msgs = [str(w.message) for w in caught if "GELU" in str(w.message)]
    return model, preprocess, acts, msgs


@torch.no_grad()
def evaluate(model, preprocess, tokenizer, data_dir, n):
    ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=preprocess)
    loader = DataLoader(Subset(ds, range(n)), batch_size=100, shuffle=False)
    text = tokenizer([f"a photo of a {c}." for c in CLASSES])
    text_feat = F.normalize(model.encode_text(text), dim=-1)
    correct, feats = 0, []
    for x, y in loader:
        f = F.normalize(model.encode_image(x), dim=-1)
        feats.append(f)
        correct += ((f @ text_feat.T).argmax(-1) == y).sum().item()
    return correct / n, torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data", default="./cifar10")
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"open_clip {open_clip.__version__} | torch {torch.__version__}")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    results = {}
    for name in ["ViT-B-32", "ViT-B-32-quickgelu"]:
        model, preprocess, acts, msgs = load(name)
        acc, feats = evaluate(model, preprocess, tokenizer, args.data, args.n)
        results[name] = feats
        print(f"\n[{name}, pretrained='openai']")
        print(f"  activation modules : {acts}")
        print(f"  GELU warnings      : {msgs or 'none'}")
        print(f"  zero-shot CIFAR-10 : {acc:.2%} on first {args.n} test images")

    cos = (results["ViT-B-32"] * results["ViT-B-32-quickgelu"]).sum(-1)
    print(f"\nImage-embedding cosine similarity between the two loads: "
          f"mean {cos.mean():.4f}, min {cos.min():.4f}")
    print("Expected if README claim held: same activation, identical accuracy, cosine = 1.0000")


if __name__ == "__main__":
    main()
