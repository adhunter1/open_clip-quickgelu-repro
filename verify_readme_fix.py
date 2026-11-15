"""Run the corrected README QuickGELU snippet verbatim, then the force_quick_gelu=True alternative it mentions.

Usage: python verify_readme_fix.py --readme path/to/open_clip/README.md
(the README from branch docs/openai-quickgelu-note, i.e. PR mlfoundations/open_clip#1254)
"""
import argparse
import re
import warnings
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--readme", required=True, help="README.md from the fixed open_clip branch")
args = ap.parse_args()

blocks = re.findall(r"```python\n(.*?)```", Path(args.readme).read_text(encoding="utf-8"), re.S)
snippet = next(b for b in blocks if "quickgelu" in b.lower())
print("README snippet run verbatim:\n" + snippet)


def report(label, model, caught):
    acts = sorted({type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__})
    msgs = [str(w.message) for w in caught if "GELU" in str(w.message)]
    print(f"[{label}] activation modules: {acts} | GELU warnings: {msgs or 'none'}")


import open_clip  # noqa: E402  (the snippet assumes it is imported, as elsewhere in the README)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ns = {"open_clip": open_clip}
    exec(snippet, ns)
report("README snippet", ns["model"], caught)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
report("ViT-B-32 + force_quick_gelu=True", model, caught)
