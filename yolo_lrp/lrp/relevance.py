from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class RelevanceMessage:
    """
    A single relevance message flowing between layers during backward
    propagation.

    Attributes
    ----------

    from_ : int, optional
        Registration number of the originating layer, if known.

    to : int
        Registration number of the destination layer, or a reserved
        negative sentinel (see `scale_key`) for "this layer's own input".

    relevance : torch.Tensor
        The relevance payload - always a single tensor, never a list.
        YOLO's per-scale seed relevance is multiple messages (one per
        scale, see `scale_key`) rather than one message holding a list,
        since the scales have differing spatial shapes.
    """

    from_: Optional[int]
    to: int
    relevance: torch.Tensor

    def __post_init__(self) -> None:
        """
        Enforces that `relevance` is an actual tensor, never a list
        smuggled through.

        Arguments
        ---------

            None

        Returns
        -------

            None
        """

        if not isinstance(self.relevance, torch.Tensor):
            raise TypeError(
                "RelevanceMessage.relevance must be a torch.Tensor, got "
                f"{type(self.relevance)} (to={self.to}, from_={self.from_})"
            )


def scale_key(scale: int) -> int:
    """
    Cache key for the seed relevance of one YOLO detection scale.

    `-1` is reserved elsewhere in this package for "this layer's own
    input", but Detect's own input is really N feature maps (one per
    scale) with differing spatial shapes that can't merge into a single
    message - so each scale gets seeded under its own `scale_key(i)`,
    read back the same way by `prop_Detect`.

    Arguments
    ---------

    scale : int
        Index of the detection scale (0 = highest resolution).

    Returns
    -------

    int
        Cache key for that scale, distinct from -1 and every real layer
        registration number (always >= 0).
    """

    return -(scale + 2)


class LayerRelevance:
    """
    Does exactly three things for one layer's relevance during the LRP
    backward pass:

    1. Scatter/gather relevance messages (`scatter`, `gather`).
    2. Cache messages addressed to layers not yet visited (`cache`,
       `pop_cache`).
    3. Print its own state (`__str__`).

    Internally everything - including this layer's own relevance - is a
    `RelevanceMessage` in `cache`, keyed by `to` (`-1` for "this layer's
    own input"). Not a `torch.Tensor` subclass: nothing relies on
    tensor-like duck typing, and the payload isn't reliably a single
    tensor anyway (see `RelevanceMessage`).

    Attributes
    ----------

    cache : Dict[int, RelevanceMessage]
        Every currently-held message, keyed by destination layer (`-1`
        included).

    contrastive : bool
        Whether this relevance represents contrastive (primal vs. dual)
        propagation. Purely descriptive - concatenating primal/dual
        batches is the caller's responsibility.

    print_decimals : int
        Decimals used when printing.
    """

    def __init__(
        self,
        contrastive: bool = False,
        print_decimals: int = 5,
    ) -> None:
        """
        Starts empty - relevance only ever enters via `gather()`.

        Arguments
        ---------

        contrastive : bool
            See class docstring.

        print_decimals : int
            See class docstring.

        Returns
        -------

            None
        """

        self.contrastive = contrastive
        self.print_decimals = print_decimals
        self.cache: Dict[int, RelevanceMessage] = {}

    def gather(self, *messages: RelevanceMessage) -> None:
        """
        Gathers incoming relevance messages into the cache. If this layer
        already has relevance cached for a message's `to`, the new
        relevance is added to it rather than replacing it - a target can
        receive contributions from several upstream paths (e.g. a Concat
        feeding two branches) before it's consumed.

        Arguments
        ---------

        *messages : RelevanceMessage
            Messages to gather.

        Returns
        -------

            None

        Raises
        ------

        RuntimeError
            If a message's relevance can't be added to what's already
            cached for its `to` (almost always a shape mismatch between
            two upstream contributions). Re-raised with target and both
            shapes for context.
        """

        for message in messages:

            if message.to not in self.cache:
                self.cache[message.to] = message
                continue

            existing = self.cache[message.to].relevance
            try:
                accumulated = existing + message.relevance
            except RuntimeError as e:
                raise RuntimeError(
                    f"Could not accumulate relevance for layer {message.to}: "
                    f"existing shape {tuple(existing.shape)}, incoming shape "
                    f"{tuple(message.relevance.shape)} ({e})"
                ) from e
            self.cache[message.to] = RelevanceMessage(
                from_=message.from_, to=message.to, relevance=accumulated
            )

    def scatter(self, which: Optional[int] = None, destroy: bool = True) -> Any:
        """
        Scatters cached relevance back out.

        Arguments
        ---------

        which : int, optional
            Destination layer to scatter relevance for (`-1` for this
            layer's own input). None returns every cached message.

        destroy : bool
            Whether to remove the scattered message(s) from the cache
            afterwards.

        Returns
        -------

        RelevanceMessage.relevance payload
            When `which` is given: the raw payload cached for that target
            (an empty tensor if `which=-1` and nothing's been gathered
            for it yet - a normal state, not a bug).

        List[RelevanceMessage]
            When `which` is None: every currently cached message.

        Raises
        ------

        KeyError
            If `which` is given, isn't -1, and nothing is cached for it.
            Unlike -1, every other key (a real layer registration number
            or a `scale_key(i)` slot) should always have been gathered by
            the time it's requested - silently returning zeros here would
            turn a real bug into a heatmap that's just quietly wrong.
        """

        if which is None:
            messages = list(self.cache.values())
            if destroy:
                self.cache.clear()
            return messages

        if which == -1:
            message = self.cache.get(which)
            payload = message.relevance if message is not None else torch.tensor([])
        elif which in self.cache:
            payload = self.cache[which].relevance
        else:
            raise KeyError(
                f"No relevance has been gathered for layer {which} - "
                "this is a bug (e.g. a stale prop_to entry or scale "
                "index), not a legitimate empty state."
            )

        if destroy and which in self.cache:
            del self.cache[which]
        return payload

    def pop_cache(self, rev_idx: int) -> None:
        """
        Moves relevance cached for a specific layer into this layer's own
        (`-1`) relevance, accumulating with whatever's already there.
        No-op if nothing is cached for `rev_idx`.

        Arguments
        ---------

        rev_idx : int
            Registration number of the layer of interest, numbered from
            top layer to back layer.

        Returns
        -------

            None
        """

        if rev_idx not in self.cache:
            return

        incoming = self.cache.pop(rev_idx)
        self.gather(
            RelevanceMessage(from_=incoming.from_, to=-1, relevance=incoming.relevance)
        )

    def __str__(self) -> str:
        """
        Arguments
        ---------

            None

        Returns
        -------

        str
            This layer's relevance state as a fraction of total relevance
            held, own vs. cached-per-target.
        """

        def value(message: Optional[RelevanceMessage]) -> Any:
            """Sums a message's relevance - (primal, dual) if contrastive,
            a single float otherwise. Absent -> zero."""
            if message is None:
                return (0.0, 0.0) if self.contrastive else 0.0
            r = message.relevance
            return (
                (r[0].sum().item(), r[1].sum().item())
                if self.contrastive
                else r.sum().item()
            )

        def fraction(v: Any) -> Any:
            """Expresses a value() result as a fraction of `total`,
            rounded to print_decimals; zero if there's no total."""
            if self.contrastive:
                p, d = v
                p = round(p / total, self.print_decimals) if total else 0.0
                d = round(d / total, self.print_decimals) if total else 0.0
                return "P:{}/D:{}".format(p, d)
            return round(v / total, self.print_decimals) if total else 0.0

        own_value = value(self.cache.get(-1))
        other_values = {to: value(msg) for to, msg in self.cache.items() if to != -1}

        total = (
            sum(own_value) + sum(p + d for p, d in other_values.values())
            if self.contrastive
            else own_value + sum(other_values.values())
        )

        cache_str = " ".join(
            "({}, {})".format(to, fraction(v)) for to, v in other_values.items()
        )

        return "LayerRelevance({}, cache={}, contrastive={})".format(
            fraction(own_value), cache_str, self.contrastive
        )
