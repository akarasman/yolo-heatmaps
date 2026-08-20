#!/usr/bin/env python
"""
CLI for generating LRP/CRP relevance heatmaps from a YOLO model, for one
or more requested classes, on a single input image.

Installed as the `yolo-lrp` console script (see pyproject.toml's
[project.scripts]); also runnable in a checkout without installing via
`python -m yolo_lrp.cli`.

Example
-------

    yolo-lrp dog_bicycle_truck.jpg --classes dog bicycle --contrastive

    yolo-lrp dog_bicycle_truck.jpg --classes truck --weights yolo26s.pt \\
        --power 2 --eps 1e-5 -o out/truck_explanation
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional, Sequence, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

from .lrp.viz import gaussian_blur_2d, overlay_heatmaps
from .yolo.explainer import YOLOLRP


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Defines and parses the script's command-line interface.

    Arguments
    ---------

    argv : Sequence[str], optional
        Argument list to parse instead of sys.argv (mainly for testing).

    Returns
    -------

    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load a YOLO model, run it on an image, and generate LRP/CRP "
            "relevance heatmaps explaining one or more requested classes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help=(
            "Class name(s) or index/indices to explain (e.g. person cat, "
            "or 0 15). One heatmap is generated per class. If omitted, "
            "generates a single heatmap for the model's winning class."
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument("--weights", default="yolo26n.pt", help="YOLO weights to load.")
    model.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(640, 640),
        help="Resize the image to HEIGHT WIDTH before explaining (should "
        "be divisible by 32).",
    )
    model.add_argument(
        "--device", default="cpu", help="Device to run inference/LRP on."
    )

    rule = parser.add_argument_group("propagation rule")
    rule.add_argument(
        "--power",
        type=int,
        default=1,
        help="Exponent applied to inputs/weights by the propagation rules.",
    )
    rule.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Stabilizing epsilon for the propagation rules.",
    )
    rule.add_argument(
        "--positive",
        dest="positive",
        action="store_true",
        default=True,
        help="Truncate negative activations to zero.",
    )
    rule.add_argument(
        "--no-positive",
        dest="positive",
        action="store_false",
        help="Don't truncate negative activations to zero.",
    )

    explain = parser.add_argument_group("explanation")
    explain.add_argument(
        "--contrastive",
        action="store_true",
        help="Compute contrastive (CRP) relevance instead of plain LRP.",
    )
    explain.add_argument(
        "--max-class-only",
        dest="max_class_only",
        action="store_true",
        default=True,
        help="Zero out all but the winning class's activations before "
        "seeding relevance.",
    )
    explain.add_argument(
        "--no-max-class-only",
        dest="max_class_only",
        action="store_false",
        help="Don't zero out non-winning classes before seeding relevance.",
    )
    explain.add_argument(
        "--primal-intensity",
        type=float,
        default=0.5,
        help="Weight for primal relevance when combining contrastive output "
        "(only used with --contrastive).",
    )
    explain.add_argument(
        "--dual-intensity",
        type=float,
        default=0.5,
        help="Weight for dual relevance when combining contrastive output "
        "(only used with --contrastive).",
    )
    explain.add_argument(
        "--suppress-outliers",
        action="store_true",
        help="Zero out the top 2%% of relevance values before saving.",
    )
    explain.add_argument(
        "--localize-bbox",
        action="store_true",
        help="Gate relevance to the class's own post-NMS bounding box "
        "before propagation. No-op for a class not detected in the "
        "image (logged, not an error).",
    )
    explain.add_argument(
        "--debug-layers",
        action="store_true",
        help="Also save one heatmap per network layer (upsampled and "
        "overlaid on the input, at each layer's own resolution) to a "
        "'debug/' subdirectory per class - for investigating where a "
        "strange final heatmap actually comes from.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory. If omitted, one is generated from the "
        "image name, requested classes, and the current timestamp.",
    )
    output.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap for the heatmap PNGs. Defaults to "
        "'seismic' - a diverging map, since relevance can be negative "
        "even outside --contrastive (any model with an attention block: "
        "YOLO26/YOLOv10/YOLO11).",
    )

    return parser.parse_args(argv)


def load_image(path: Path, size: Sequence[int]) -> torch.Tensor:
    """
    Loads and preprocesses an image for `YOLOLRP.explain`.

    Arguments
    ---------

    path : Path
        Path to the image file.

    size : Sequence[int]
        (height, width) to resize to.

    Returns
    -------

    torch.Tensor
        The image as a (3, height, width) float tensor.
    """

    image = Image.open(path).convert("RGB")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(tuple(size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    return cast(torch.Tensor, transform(image).float())


def resolve_class(raw: str) -> Union[int, str]:
    """
    Interprets a `--classes` argument as a class index if it looks like
    one, otherwise leaves it as a class name for `YOLOLRP.explain` to
    resolve against the model's own class list.

    Arguments
    ---------

    raw : str
        A single `--classes` value, as given on the command line.

    Returns
    -------

    int or str
        The class index, or the class name unchanged.
    """

    try:
        return int(raw)
    except ValueError:
        return raw


def default_output_dir(image_path: Path) -> Path:
    """
    Builds a default output directory name from the image name, the
    requested classes, and the current timestamp, for when `-o` isn't
    given.

    Arguments
    ---------

    image_path : Path
        The input image path.

    Returns
    -------

    Path
        A directory path (not yet created) under the current directory.
    """

    stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(f"{image_path.stem}_{stamp}")


def save_heatmap(relevance: torch.Tensor, path: Path, cmap: str) -> None:
    """
    Renders a single relevance heatmap to a PNG file, lightly blurred for
    display (see `gaussian_blur_2d`; doesn't touch the values). Centers
    the color scale on zero if the data has any real negative value,
    not based on `--contrastive` - plain LRP can also produce negative
    relevance for any model with an attention block (YOLO26/YOLOv10/
    YOLO11's C2PSA/PSA), since AttnLRP's rules aren't sign-preserving
    the way the rest are.

    Arguments
    ---------

    relevance : torch.Tensor
        2D relevance map, as returned by `YOLOLRP.explain`.

    path : Path
        File to write the PNG to.

    cmap : str
        Matplotlib colormap name. Should be a diverging map (e.g. the
        default `seismic`) unless the caller has confirmed this
        relevance can never be negative.

    Returns
    -------

        None
    """

    array = gaussian_blur_2d(relevance)
    max_abs = float(np.abs(array).max())
    symmetric = bool((array < 0).any())

    fig, ax = plt.subplots(figsize=(6, 6))
    if symmetric:
        ax.imshow(array, cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        ax.imshow(array, cmap=cmap, vmin=0, vmax=max_abs)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_layer_debug(
    image: torch.Tensor, relevance: torch.Tensor, path: Path, cmap: str = "seismic"
) -> bool:
    """
    Renders one layer's own-input relevance snapshot (at that layer's own,
    usually much smaller, resolution), upsampled and overlaid on the
    input image via `overlay_heatmaps`, for `--debug-layers`.

    Arguments
    ---------

    image : torch.Tensor
        The (3, H, W) input image being explained.

    relevance : torch.Tensor
        A single layer's relevance snapshot, as returned by
        `YOLOLRP.get_layer_relevance` - shape (B, C, h, w), B possibly
        doubled under contrastive relevance. May be empty - not every
        layer has gathered its own-input relevance yet at snapshot time.

    path : Path
        File to write the PNG to.

    cmap : str
        Matplotlib colormap. Diverging by default - see `save_heatmap`
        for why relevance can be signed outside contrastive mode too.

    Returns
    -------

    bool
        Whether a file was actually written - False if `relevance` was
        empty (e.g. Detect, the first layer inverted).
    """

    if relevance.numel() == 0:
        return False

    # Primal only (index 0) even under contrastive - dual would double
    # every debug image for little insight.
    layer_map = relevance[0].sum(dim=0, keepdim=True).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        layer_map, size=image.shape[-2:], mode="nearest"
    )[0, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    overlay_heatmaps(ax, image, [(upsampled, cmap, None)])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _save_layer_debug_images(
    explainer: YOLOLRP, image: torch.Tensor, debug_dir: Path, cmap: str
) -> None:
    """
    Saves one `save_layer_debug` image per entry in
    `explainer.get_layer_relevance()` into `debug_dir`, numbered in
    propagation order - 0 is the output end (Detect), the last number is
    the network's own input.

    Arguments
    ---------

    explainer : YOLOLRP
        The explainer that was just run with `save_r_values=True`.

    image : torch.Tensor
        The (3, H, W) input image being explained.

    debug_dir : Path
        Directory to write the numbered PNGs into (created if needed).

    cmap : str
        Matplotlib colormap, forwarded to `save_layer_debug`.

    Returns
    -------

        None
    """

    r_values = explainer.get_layer_relevance()
    if not r_values:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, (layer, relevance) in enumerate(r_values):
        if layer is None:
            name = "input"
        else:
            reg_num = getattr(layer, "reg_num", "?")
            name = f"{type(layer).__name__}_{reg_num}"
        path = debug_dir / f"{i:03d}_{name}.png"
        if save_layer_debug(image, relevance, path, cmap=cmap):
            saved += 1

    print(f"[debug] {saved}/{len(r_values)} layer heatmaps saved to {debug_dir}")


def explain_class(
    explainer: YOLOLRP,
    image: torch.Tensor,
    raw_cls: Optional[str],
    output_dir: Path,
    cmap: str,
    *,
    contrastive: bool,
    max_class_only: bool,
    primal_intensity: float,
    dual_intensity: float,
    suppress_outliers: bool,
    debug_layers: bool,
    localize_bbox: bool,
) -> None:
    """
    Generates one relevance heatmap for a single requested class and
    writes it (plus the raw relevance array) to `output_dir`.

    Arguments
    ---------

    explainer : YOLOLRP
        The explainer to run.

    image : torch.Tensor
        Preprocessed input image, as returned by `load_image`.

    raw_cls : str, optional
        A single `--classes` value as given on the command line, or None
        to explain the model's winning class.

    output_dir : Path
        Directory to write the `.npy`/`.png` outputs into.

    cmap : str
        Matplotlib colormap for the heatmap PNG.

    contrastive : bool
        Whether to compute contrastive (CRP) relevance instead of plain
        LRP.

    max_class_only : bool
        Whether to zero out all but the winning class's activations
        before seeding relevance.

    primal_intensity : float
        Weight for primal relevance when combining contrastive output.

    dual_intensity : float
        Weight for dual relevance when combining contrastive output.

    suppress_outliers : bool
        Whether to zero out the top 2% of relevance values before saving.

    debug_layers : bool
        Also save one upsampled, overlaid heatmap per network layer to
        `output_dir/debug/<label>/` (see `save_layer_debug`); sets
        `explainer.save_r_values` for the duration of this call.

    localize_bbox : bool
        Gate relevance to the class's own post-NMS bounding box before
        propagation - see `YOLOLRP.explain`.

    Returns
    -------

        None
    """

    cls = resolve_class(raw_cls) if raw_cls is not None else None
    explainer.save_r_values = debug_layers

    heatmap = explainer.explain(
        image,
        cls=cls,
        contrastive=contrastive,
        max_class_only=max_class_only,
        primal_intensity=primal_intensity,
        dual_intensity=dual_intensity,
        localize_bbox=localize_bbox,
    )
    if suppress_outliers:
        heatmap = explainer._suppress_outliers(heatmap)

    label = str(raw_cls) if raw_cls is not None else "top"
    safe_label = label.replace("/", "_").replace(" ", "_")

    np.save(output_dir / f"{safe_label}.npy", heatmap.detach().cpu().numpy())
    save_heatmap(heatmap, output_dir / f"{safe_label}.png", cmap=cmap)
    print(f"[{label}] heatmap saved to {output_dir / f'{safe_label}.png'}")

    if debug_layers:
        _save_layer_debug_images(
            explainer, image, output_dir / "debug" / safe_label, cmap
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point: loads the model and image, generates one relevance
    heatmap per requested class, and writes them (plus the raw relevance
    arrays and a copy of the resized input) to the output directory.

    Arguments
    ---------

    argv : Sequence[str], optional
        Argument list to parse instead of sys.argv (mainly for testing).

    Returns
    -------

    int
        Process exit code.
    """

    args = parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    if any(dim % 32 != 0 for dim in args.size):
        print(
            f"Warning: --size {tuple(args.size)} is not divisible by 32; "
            "the model may warn about or mishandle this input shape."
        )

    device = torch.device(args.device)
    image = load_image(args.image, args.size).to(device)

    yolo = YOLO(args.weights)
    explainer = YOLOLRP(yolo, power=args.power, positive=args.positive, eps=args.eps)
    explainer.to(device)

    output_dir = args.output or default_output_dir(args.image)
    output_dir.mkdir(parents=True, exist_ok=True)

    # seismic (diverging), not Reds: relevance can be negative even
    # outside --contrastive - see save_heatmap.
    cmap = args.cmap or "seismic"
    torchvision.utils.save_image(image, output_dir / "input.png")

    requested = args.classes if args.classes is not None else [None]
    for raw_cls in requested:
        explain_class(
            explainer,
            image,
            raw_cls,
            output_dir,
            cmap,
            contrastive=args.contrastive,
            max_class_only=args.max_class_only,
            primal_intensity=args.primal_intensity,
            dual_intensity=args.dual_intensity,
            suppress_outliers=args.suppress_outliers,
            debug_layers=args.debug_layers,
            localize_bbox=args.localize_bbox,
        )

    print(f"Done. Output written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
