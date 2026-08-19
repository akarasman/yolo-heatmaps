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
        Registration number of the layer the relevance originated from, if
        known. None where the origin isn't tracked by the caller.

    to : int
        Registration number of the layer this message is addressed to, or
        a reserved negative sentinel (see `scale_key`) for "this layer's
        own input" (the convention used throughout this package).

    relevance : torch.Tensor
        The relevance payload itself, always a single tensor - YOLO's
        per-detection-scale seed relevance is multiple messages (one per
        scale, see `scale_key`), never one message holding a list, since
        the scales have differing spatial shapes and can't be merged into
        one tensor.
    """

    from_: Optional[int]
    to: int
    relevance: torch.Tensor

    def __post_init__(self) -> None:
        """
        Enforces the one real invariant this class exists to guarantee:
        `relevance` is an actual tensor, never a list smuggled through
        (see the class docstring for why that distinction matters).

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

    `-1` is reserved for "this layer's own input" everywhere else in this
    package, but Detect's own input is really N separate feature maps (one
    per detection scale) with differing spatial shapes that can't be
    merged into a single message - so `_RelevanceInitializer` seeds one
    message per scale under `scale_key(i)`, and `prop_Detect` reads them
    back the same way, instead of either flattening the scales together or
    smuggling a list through a single `to=-1` message.

    Arguments
    ---------

    scale : int
        Index of the detection scale (0 = highest resolution).

    Returns
    -------

    int
        Cache key for that scale's seed message, distinct from -1 and from
        every real layer registration number (which are always >= 0).
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

    Internally, everything - including this layer's own relevance - is
    just a `RelevanceMessage` in `cache`, keyed by `to` (`-1` for "this
    layer's own input"). Not a `torch.Tensor` subclass: nothing outside
    `scatter`/`gather` ever relied on tensor-like duck typing, and the
    payload isn't even reliably a single tensor (see `RelevanceMessage`),
    so the old inheritance was never an honest fit.

    Attributes
    ----------

    cache : Dict[int, RelevanceMessage]
        Every currently-held relevance message, keyed by destination layer
        (`-1` included, for this layer's own input).

    contrastive : bool
        Whether this relevance represents contrastive (primal vs. dual)
        propagation. Purely descriptive - concatenating primal/dual
        batches, if needed, is the caller's responsibility (see
        `_RelevanceInitializer`).

    print_decimals : int
        Amount of decimals to use in printing.
    """

    def __init__(
        self,
        contrastive: bool = False,
        print_decimals: int = 5,
    ) -> None:
        """
        Starts empty. Seed initial relevance with `gather()` - there's no
        separate construction-time shortcut, so there's exactly one way
        relevance ever enters a LayerRelevance.

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
        Gather incoming relevance messages into the cache. Every target
        accumulates: if this layer already has relevance cached for a
        message's `to`, the new message's relevance is added to it rather
        than replacing it, since any target can receive contributions
        from several upstream paths (e.g. a Concat feeding two branches,
        or two branches both eventually routing relevance back to the
        same earlier layer) before it's consumed.

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
            two upstream contributions that should have lined up).
            Re-raised with the target and both shapes, since the bare
            torch error alone gives no indication which layer or messages
            were involved.
        """

        for message in messages:

            # Add to cache if nothing's there yet for this target
            if message.to not in self.cache:
                self.cache[message.to] = message
                continue

            # Accumulate to existing cache entry for this target
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
        Scatter cached relevance back out.

        Arguments
        ---------

        which : int, optional
            Destination layer to scatter relevance for (`-1` for this
            layer's own input). If None, every cached message is
            returned.

        destroy : bool
            Whether to remove the scattered message(s) from the cache
            afterwards.

        Returns
        -------

        RelevanceMessage.relevance payload
            When `which` is given: the raw relevance payload cached for
            that target (an empty tensor if `which=-1` and nothing has
            been gathered for it yet - a layer legitimately having no
            own-input relevance is a normal state, not a bug).

        List[RelevanceMessage]
            When `which` is None: every currently cached message.

        Raises
        ------

        KeyError
            If `which` is given, isn't -1, and nothing is cached for it.
            Unlike -1, every other key (a real layer registration number,
            or a `scale_key(i)` seed slot) should always have been
            gathered by the time anything asks for it - silently handing
            back zeros here would turn a real bug (e.g. an off-by-one in
            scale indexing) into a heatmap that's just quietly wrong.
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
        Move relevance cached for a specific layer into this layer's own
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
            """Sums a message's relevance - a (primal, dual) pair if
            contrastive, a single float otherwise. Absent -> zero."""
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
            rounded to print_decimals; zero if there's no total to divide by.
            """
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
