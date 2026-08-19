# yolo-heatmaps

Explanatory heatmaps for [YOLO26](https://github.com/ultralytics/ultralytics) object detection results, using Layer-wise Relevance Propagation (LRP) and its contrastive variant (CRP) — [Bach et al.](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf). YOLOv8/YOLOv9/YOLOv10/YOLO11 are also fully supported — see "Supported YOLO versions" below.

Given a detector and an image, `YOLOLRP` explains *why* the model predicted a given class at a given location: LRP produces a per-pixel relevance map for one class on its own, and CRP contrasts that class against every other class, so the resulting map shows what counts *for* the class (positive) as well as *against* it (negative).

## LRP heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/db92eacd-b6d2-4b6f-86a2-cc3fe3d8fad8)

## CRP heatmaps

![image](https://github.com/akarasman/yolo-heatmaps/assets/56434833/140e8e6a-e589-450f-8c09-6b05f94fbeeb)

## Installation

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is the minimal runtime install (what you need to just use `YOLOLRP`/`explain.py`). If you're developing this repo — running the test suite, the notebook, or the formatter/linters — use `requirements-dev.txt` instead, which layers exact-pinned dev tooling (pytest, ipykernel/nbclient, black, isort, flake8, mypy) on top:

```
pip install -r requirements-dev.txt
```

Weights aren't bundled — the first `YOLO(...)` call downloads the requested checkpoint (e.g. `yolo26n.pt`) automatically via `ultralytics`.

## Quick start

### Command line

```
python explain.py riksi.jpg --classes person cat --contrastive
```

Writes one heatmap PNG per requested class (plus the raw relevance as `.npy`) to an auto-named output directory. Run `python explain.py --help` for the full list of knobs — model/weights, propagation-rule parameters (`power`/`eps`/`positive`), contrastive weighting, colormap, etc.

### Python API

```python
from ultralytics import YOLO
from src.yolo.explainer import YOLOLRP

# `image` is a (3, H, W) float tensor, H and W divisible by 32
yolo = YOLO('yolo26n.pt')
lrp = YOLOLRP(yolo, power=2, eps=1e-05)

explanation_lrp = lrp.explain(image, cls='person', contrastive=False)
explanation_crp = lrp.explain(image, cls='person', contrastive=True)
```

Don't call `yolo(...)`/`yolo.predict(...)` on the wrapped model directly — that fuses it for standalone inference and breaks the internal structure `YOLOLRP` reads from. Construct `YOLOLRP` from a freshly-loaded model and only ever call it through `YOLOLRP`/`explain()` afterward.

See `example.ipynb` for a full worked example (image loading, LRP and CRP heatmaps, plotting) against a real image.

## Supported YOLO versions

| Version | Checkpoint tested |
|---|---|
| YOLOv8 | `yolov8n.pt` |
| YOLOv9 | `yolov9t.pt` |
| YOLO26 | `yolo26n.pt` |
| YOLOv10 | `yolov10n.pt` |
| YOLO11 | `yolo11n.pt` |

## Project layout

```
src/
  lrp/            Architecture-independent LRP machinery
    relevance.py    RelevanceMessage / LayerRelevance - the scatter/gather/cache data structures
    rules.py         PropRule (Conv/Linear/MaxPool/Upsample) - Strategy-pattern relevance rules
    inverter.py      Inverter - per-layer-type backward dispatch
    fwd_hooks.py     Default forward hooks (cache in/out tensors per primitive layer type)
    utils.py
  yolo/           YOLO26-specific wiring
    explainer.py     YOLOLRP - the main entry point
    block_rules.py   Relevance rules for YOLO26's composite blocks (C3k2, C2PSA, SPPF, Attention, ...)
    fwd_hooks.py     Forward hooks for YOLO-specific blocks (SPPF, Concat)
explain.py        CLI (see above)
example.ipynb     Notebook walkthrough
tests/            pytest suite (see below)
```

`lrp/` has no dependency on `ultralytics` or YOLO at all; `yolo/` is where every YOLO26-architecture assumption lives. A different YOLO version whose block types are a straightforward registry swap doesn't need a new `YOLOLRP` subclass — `prop_registry`/`fwd_hook_registry` are constructor parameters, defaulting to YOLO26's own (`block_rules.PROP_REGISTRY`/`fwd_hooks.FWD_HOOK_REGISTRY`).

## Testing

```
pip install -r requirements-dev.txt
pytest
```

Everything except `tests/test_integration.py` runs offline against stubbed/synthetic modules. `test_integration.py` (marked `integration`) downloads a real checkpoint per supported YOLO version on first run (see "Supported YOLO versions" above) and exercises the full `explain()` pipeline against each — deselect it with `pytest -m "not integration"` if you want the fast, network-free subset. `pytest` reports coverage automatically (`pyproject.toml`'s `addopts`).

`mypy .` runs in strict mode across `src/` and `explain.py`; `black`, `isort`, and `flake8` are also part of CI (`.github/workflows/ci.yml`) — run them all together with:

```
black --check src tests explain.py
isort --check src tests explain.py
flake8 src
mypy .
```

## Notes on the LRP rules

- **Attention** (`C2PSA`, and the `attn=True` branch of `C3k2`) has a genuine relevance rule, not an identity placeholder — a from-scratch implementation in the spirit of [AttnLRP](https://proceedings.mlr.press/v235/achtibat24a.html) (Achtibat et al., 2024): a bilinear split for the two matrix products in self-attention (`Q·Kᵀ`, `attention·V`), and a dedicated softmax rule derived from its own Jacobian, rather than treating attention as opaque. It's a principled derivation done for this project, not a verified reproduction of the paper's exact published rules.
- BatchNorm is treated as transparent to relevance throughout (a standard LRP-for-CNNs simplification), not explicitly decomposed.
- Contrastive (CRP) relevance intentionally isn't globally conserved the way plain LRP roughly is — the dual (primal + "everything else") path expands the relevance total as it flows backward, which is expected, not a bug.

If you are planning to utilize this repo in your research kindly cite the following work:

```
@INPROCEEDINGS{9827744,
  author={Karasmanoglou, Apostolos and Antonakakis, Marios and Zervakis, Michalis},
  booktitle={2022 IEEE International Conference on Imaging Systems and Techniques (IST)}, 
  title={Heatmap-based Explanation of YOLOv5 Object Detection with Layer-wise Relevance Propagation}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/IST55454.2022.9827744}
}
```

## License

[MIT](LICENSE)
