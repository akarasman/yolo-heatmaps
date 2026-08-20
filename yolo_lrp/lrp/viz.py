"""
Shared display-only rendering for relevance heatmaps, used by both the CLI
(`yolo_lrp.cli`) and `example.ipynb`. Rendering only - blur, alpha, color -
applied at draw time; never touches the tensors `explain()`/
`get_layer_relevance()` return or writes to `.npy`. Centralized so a
rendering fix benefits both callers at once.
"""

from typing import Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Patch

ArrayLike = Union[torch.Tensor, np.ndarray]


def gaussian_blur_2d(
    heatmap: ArrayLike, sigma: float = 0.5, kernel_size: Optional[int] = None
) -> np.ndarray:
    """
    Lightly Gaussian-blurs a 2D relevance map - display-only, never
    applied to the values `explain()`/`get_layer_relevance()` return or
    written to `.npy`.

    Exists because `ConvRule`'s backward step inverts strided convs via
    `conv_transpose2d`, producing a checkerboard artifact whenever kernel
    size doesn't evenly divide stride (this codebase's default `Conv`
    does, at kernel_size=3/stride=2 - confirmed on a real explain() call
    as a ~3.4x alternating parity pattern). `sigma=2.0` is a tradeoff:
    it doesn't fully cancel every stride-2 layer's checkerboard period the
    way a heavier blur (sigma=4) would, but keeps more of the map's real
    texture visible.

    Arguments
    ---------

    heatmap : torch.Tensor or numpy.ndarray
        2D relevance map to blur.

    sigma : float
        Gaussian standard deviation, in pixels.

    kernel_size : int, optional
        Size of the square blur kernel (odd). Defaults to `6 * sigma`
        rounded up to odd, wide enough to cover the Gaussian's tails.

    Returns
    -------

    numpy.ndarray
        The blurred map, same shape as `heatmap`.
    """

    tensor = (
        heatmap
        if isinstance(heatmap, torch.Tensor)
        else torch.as_tensor(heatmap, dtype=torch.float32)
    )

    if kernel_size is None:
        kernel_size = int(6 * sigma) | 1

    coords = (
        torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
        - kernel_size // 2
    )
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)

    padded = tensor.view(1, 1, *tensor.shape)
    blurred = torch.nn.functional.conv2d(padded, kernel_2d, padding=kernel_size // 2)
    return blurred.view(tensor.shape).detach().cpu().numpy()


def alpha_from_percentile(
    heatmap: np.ndarray, floor_pct: float = 40, ceil_pct: float = 95
) -> np.ndarray:
    """
    Maps |heatmap| to an alpha channel via a percentile ramp: transparent
    at or below `floor_pct`, fully opaque at or above `ceil_pct`, linear
    between. Controls what *fraction* of a map's pixels are visible
    directly, unlike a fixed power-law curve - an earlier attempt at that
    looked diffuse for one map and razor-thin for another at the same
    exponent.

    Arguments
    ---------

    heatmap : numpy.ndarray
        Relevance map (already blurred, if desired) to derive alpha from.

    floor_pct : float
        Percentile of |heatmap| below which alpha is 0.

    ceil_pct : float
        Percentile of |heatmap| at or above which alpha is 1.

    Returns
    -------

    numpy.ndarray
        Alpha channel in [0, 1], same shape as `heatmap`.
    """

    magnitude = np.abs(heatmap)
    floor_value = np.percentile(magnitude, floor_pct)
    ceil_value = np.percentile(magnitude, ceil_pct)
    alpha = (magnitude - floor_value) / max(ceil_value - floor_value, 1e-12)
    return cast(np.ndarray, np.clip(alpha, 0, 1))


def colors_from_heatmap(
    heatmap: np.ndarray, cmap: str, color_gamma: float = 1.0
) -> np.ndarray:
    """
    Maps a (possibly signed) heatmap to RGBA colors. `color_gamma`
    controls how quickly color saturates from pale to deep as magnitude
    rises, independent of alpha (see `alpha_from_percentile`); values < 1
    pull mid-magnitude pixels toward full saturation. `color_gamma=1` is
    plain linear.

    Always scaled symmetrically around zero, not [0, max]: LRP relevance
    can be negative even outside contrastive mode for models with an
    attention block (C2PSA/PSA - YOLO26/v10/11 all have one), since
    AttnLRP's bilinear/softmax rules aren't sign-preserving like the
    z+-style Conv/Linear rules used elsewhere.

    Arguments
    ---------

    heatmap : numpy.ndarray
        Relevance map (already blurred, if desired) to colorize.

    cmap : str
        Matplotlib colormap name. Should be diverging unless this
        relevance is known to always be non-negative.

    color_gamma : float
        Compression exponent applied to normalized magnitude before
        colorizing.

    Returns
    -------

    numpy.ndarray
        RGBA array, shape `heatmap.shape + (4,)`.
    """

    max_abs = np.abs(heatmap).max() or 1.0  # all-zero map: avoid a 0/0 norm
    normed = np.clip(heatmap / max_abs, -1, 1)  # signed, in [-1, 1]
    shaped = np.sign(normed) * (np.abs(normed) ** color_gamma)  # still [-1, 1]
    return plt.get_cmap(cmap)((shaped + 1) / 2)  # cmap wants [0, 1]


def overlay_heatmaps(
    ax: Axes,
    image: torch.Tensor,
    layers: Sequence[Tuple[ArrayLike, str, Optional[str]]],
    floor_pct: float = 70,
    ceil_pct: float = 95,
    color_gamma: float = 1.0,
    sigma: float = 0.5,
    kernel_size: Optional[int] = None,
) -> None:
    """
    Draws `image` with one or more relevance heatmaps alpha-blended on
    top - transparent where relevance is near zero, opaque and colored by
    that layer's own cmap where it's large. All layers are composited
    into a single RGBA image drawn with one `imshow` call, rather than
    one `imshow` per layer stacked in z-order: two locally-noisy alpha
    fields (see `gaussian_blur_2d`) would otherwise flip whichever layer
    is drawn last to fully opaque pixel-by-pixel wherever its alpha spikes
    above the other's, producing a speckled mess instead of a genuine
    blend where regions overlap.

    `floor_pct`/`ceil_pct` (see `alpha_from_percentile`) control *where*
    anything is visible; `color_gamma` (see `colors_from_heatmap`)
    controls how deep the color gets within that region.

    Arguments
    ---------

    ax : matplotlib.axes.Axes
        Axes to draw onto.

    image : torch.Tensor
        The (3, H, W) input image being explained.

    layers : Sequence[(torch.Tensor or numpy.ndarray, str, str or None)]
        One `(heatmap, cmap, label)` entry per map to overlay, in draw
        order. `label=None` omits that layer from the legend; if every
        layer omits it, no legend is drawn.

    floor_pct : float
        Forwarded to `alpha_from_percentile` for every layer.

    ceil_pct : float
        Forwarded to `alpha_from_percentile` for every layer.

    color_gamma : float
        Forwarded to `colors_from_heatmap` for every layer.

    sigma : float
        Forwarded to `gaussian_blur_2d` for every layer.

    kernel_size : int, optional
        Forwarded to `gaussian_blur_2d` for every layer.

    Returns
    -------

        None
    """

    ax.imshow(image.permute(1, 2, 0))

    handles = []
    per_layer_rgb = []
    per_layer_alpha = []
    for heatmap, cmap, label in layers:
        blurred = gaussian_blur_2d(heatmap, sigma=sigma, kernel_size=kernel_size)
        per_layer_rgb.append(colors_from_heatmap(blurred, cmap, color_gamma)[..., :3])
        per_layer_alpha.append(alpha_from_percentile(blurred, floor_pct, ceil_pct))
        if label is not None:
            handles.append(Patch(color=plt.get_cmap(cmap)(0.85), label=label))

    if per_layer_rgb:
        rgb_stack = np.stack(per_layer_rgb, axis=0)  # (N, H, W, 3)
        alpha_stack = np.stack(per_layer_alpha, axis=0)  # (N, H, W)

        # Each layer's RGB weighted by its own alpha, so overlapping
        # pixels blend proportionally instead of picking one layer.
        weight = alpha_stack[..., None]
        weight_sum = weight.sum(axis=0)
        composite_rgb = np.divide(
            (rgb_stack * weight).sum(axis=0),
            weight_sum,
            out=np.zeros(rgb_stack.shape[1:]),
            where=weight_sum > 0,
        )
        # P(at least one layer "on") per pixel - not a plain sum, which
        # could exceed 1 where layers overlap.
        composite_alpha = 1 - np.prod(1 - alpha_stack, axis=0)

        composite = np.concatenate([composite_rgb, composite_alpha[..., None]], axis=-1)
        ax.imshow(composite)

    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=12, framealpha=0.5)
