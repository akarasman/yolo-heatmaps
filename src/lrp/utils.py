import torch


def get_dummy_summation_conv_layer(out_channels: int) -> torch.nn.Conv2d:
    """
    Builds a fixed-weight 1x1 Conv2d that reproduces `a + b` from
    `cat(a, b)` along the channel dimension - used throughout
    block_rules.py to split relevance for a residual/branch addition via
    ConvRule's own LRP rule (see `_split_additive_relevance`), instead of
    a hand-rolled ratio.

    Arguments
    ---------

    out_channels : int
        Number of channels in each addend (`a`/`b`); the conv takes
        `2 * out_channels` input channels and produces `out_channels`.

    Returns
    -------

    torch.nn.Conv2d
        A bias-free 1x1 conv whose weight is 1.0 at `[c, c]` and
        `[c, out_channels + c]` for every output channel `c`, and 0
        elsewhere.
    """

    in_channels = 2 * out_channels

    conv = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    weight = torch.zeros((out_channels, in_channels, 1, 1))

    for c in range(out_channels):
        weight[c, c, 0, 0] = 1.0
        weight[c, out_channels + c, 0, 0] = 1.0

    conv.weight = torch.nn.Parameter(weight)

    return conv
