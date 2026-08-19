#!/usr/bin/env python
"""
CLI for generating LRP/CRP relevance heatmaps from a YOLO model, for one
or more requested classes, on a single input image.

Example
-------

    py explain.py riksi.jpg --classes person cat --contrastive

    py explain.py riksi.jpg --classes person --weights yolo26s.pt \\
        --power 2 --eps 1e-5 -o out/person_explanation
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

from src.yolo.explainer import YOLOLRP


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
        "'seismic' with --contrastive (relevance can be negative), "
        "'Reds' otherwise.",
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


def save_heatmap(
    relevance: torch.Tensor, path: Path, cmap: str, symmetric: bool
) -> None:
    """
    Renders a single relevance heatmap to a PNG file.

    Arguments
    ---------

    relevance : torch.Tensor
        2D relevance map, as returned by `YOLOLRP.explain`.

    path : Path
        File to write the PNG to.

    cmap : str
        Matplotlib colormap name.

    symmetric : bool
        Whether to center the color scale on zero (for contrastive
        relevance, which can be negative) instead of starting at zero.

    Returns
    -------

        None
    """

    array = relevance.detach().cpu().numpy()
    max_abs = float(np.abs(array).max())

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

    Returns
    -------

        None
    """

    cls = resolve_class(raw_cls) if raw_cls is not None else None

    heatmap = explainer.explain(
        image,
        cls=cls,
        contrastive=contrastive,
        max_class_only=max_class_only,
        primal_intensity=primal_intensity,
        dual_intensity=dual_intensity,
    )
    if suppress_outliers:
        heatmap = explainer._suppress_outliers(heatmap)

    label = str(raw_cls) if raw_cls is not None else "top"
    safe_label = label.replace("/", "_").replace(" ", "_")

    np.save(output_dir / f"{safe_label}.npy", heatmap.detach().cpu().numpy())
    save_heatmap(
        heatmap,
        output_dir / f"{safe_label}.png",
        cmap=cmap,
        symmetric=contrastive,
    )
    print(f"[{label}] heatmap saved to {output_dir / f'{safe_label}.png'}")


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

    cmap = args.cmap or ("seismic" if args.contrastive else "Reds")
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
        )

    print(f"Done. Output written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
