"""Packing policies.

Packing is a training decision, not a storage detail.  Every unused slot in a
window is a token the model did not learn from, and every unrelated document
sharing an attention window is a chance to learn a transition that does not
exist.  The right trade-off differs by data type, so the policy is chosen per
capability lane rather than globally:

| policy                | used by            | co-packs | may split a document |
|-----------------------|--------------------|----------|----------------------|
| pad_only              | (baseline)         | no       | no (truncates)       |
| concat_chop           | general_web        | yes      | yes, anywhere        |
| greedy                | math_science       | yes      | at document boundary |
| best_fit              | code, indic        | yes      | at document boundary |
| structure_preserving  | agentic            | no       | at turn boundaries   |
| long_context          | reasoning          | no       | at turn boundaries   |

`pad_only` is kept even though no lane uses it, because the packing report
compares every policy's utilisation against it - that comparison is the whole
argument for doing anything more complicated.

These functions work on abstract `Item`s so they can be unit tested without a
tokenizer, a corpus or a shard on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

POLICIES = (
    "pad_only",
    "concat_chop",
    "greedy",
    "best_fit",
    "structure_preserving",
    "long_context",
)

# A long window is wasted on a short document.  Under long_context packing a
# document must fill at least this fraction of the window to be scheduled.
LONG_CONTEXT_MIN_FILL = 0.45


@dataclass
class Item:
    """One packable unit: a document, or one piece of a split document."""

    item_id: str
    size: int
    # Offsets where this item may be split, relative to its own start.
    # For a structured document these are turn boundaries.
    split_points: Tuple[int, ...] = ()
    min_context: int = 0
    order_key: str = ""

    def __post_init__(self):
        if not self.order_key:
            self.order_key = self.item_id


@dataclass
class Placement:
    item_id: str
    offset_in_item: int      # where in the source item this piece starts
    length: int
    start_in_window: int

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "offset_in_item": self.offset_in_item,
            "length": self.length,
            "start_in_window": self.start_in_window,
        }


@dataclass
class Window:
    placements: List[Placement] = field(default_factory=list)
    used: int = 0

    def add(self, item_id: str, offset: int, length: int) -> None:
        self.placements.append(Placement(item_id, offset, length, self.used))
        self.used += length


@dataclass
class PackResult:
    windows: List[Window]
    capacity: int
    splits: int = 0
    truncated_tokens: int = 0
    deferred: List[str] = field(default_factory=list)
    input_tokens: int = 0

    @property
    def utilisation(self) -> float:
        """Share of allocated window positions holding a real token."""
        if not self.windows:
            return 0.0
        return sum(w.used for w in self.windows) / (len(self.windows) * self.capacity)

    @property
    def retention(self) -> float:
        """Share of the *input* corpus that survived packing.

        Reported alongside utilisation because utilisation alone can be gamed.
        `pad_only` scores 1.0 utilisation on an oversized document by throwing
        away everything past the window - perfect packing of the part it kept,
        and a corpus that lost half its tokens.  Retention is what catches that.
        """
        if self.input_tokens <= 0:
            return 0.0
        return sum(w.used for w in self.windows) / self.input_tokens

    @property
    def wasted(self) -> int:
        return len(self.windows) * self.capacity - sum(w.used for w in self.windows)

    def stats(self) -> dict:
        return {
            "windows": len(self.windows),
            "capacity": self.capacity,
            "input_tokens": self.input_tokens,
            "used_tokens": sum(w.used for w in self.windows),
            "wasted_positions": self.wasted,
            "utilisation": round(self.utilisation, 4),
            "retention": round(self.retention, 4),
            "document_splits": self.splits,
            "truncated_tokens": self.truncated_tokens,
            "deferred_items": sorted(self.deferred),
            "deferred_tokens": self.input_tokens
            - sum(w.used for w in self.windows)
            - self.truncated_tokens,
            "boundary_crossings": sum(
                max(0, len(w.placements) - 1) for w in self.windows
            ),
        }


# --------------------------------------------------------------------------
# Splitting helper
# --------------------------------------------------------------------------


def _split_at_boundaries(item: Item, capacity: int) -> List[Tuple[int, int]]:
    """Cut an oversized item into (offset, length) pieces.

    Preference order: cut at a declared split point (a turn boundary) that
    leaves the piece as full as possible; if no split point fits, cut at the
    capacity.  This is what "structure preserving" means in practice - not that
    a document is never divided, but that it is never divided mid-turn, so each
    piece is still a coherent stretch of one conversation.
    """
    pieces: List[Tuple[int, int]] = []
    offset = 0
    boundaries = sorted(set(item.split_points))
    while offset < item.size:
        remaining = item.size - offset
        if remaining <= capacity:
            pieces.append((offset, remaining))
            break
        limit = offset + capacity
        usable = [b for b in boundaries if offset < b <= limit]
        cut = usable[-1] if usable else limit
        pieces.append((offset, cut - offset))
        offset = cut
    return pieces


# --------------------------------------------------------------------------
# The policies
# --------------------------------------------------------------------------


def pack_pad_only(items: Sequence[Item], capacity: int) -> PackResult:
    """One item per window; anything over capacity is truncated away."""
    result = PackResult([], capacity)
    for item in items:
        window = Window()
        length = min(item.size, capacity)
        window.add(item.item_id, 0, length)
        result.truncated_tokens += item.size - length
        result.windows.append(window)
    return result


def pack_concat_chop(items: Sequence[Item], capacity: int) -> PackResult:
    """Join everything into one stream and cut fixed windows out of it.

    Efficient, and correct for plain pretraining where a sequence boundary is
    an engineering boundary.  Documents are split wherever the cut lands.
    """
    result = PackResult([], capacity)
    window = Window()
    for item in items:
        offset = 0
        while offset < item.size:
            room = capacity - window.used
            if room == 0:
                result.windows.append(window)
                window = Window()
                room = capacity
            take = min(room, item.size - offset)
            window.add(item.item_id, offset, take)
            if offset + take < item.size:
                result.splits += 1
            offset += take
    if window.used:
        result.windows.append(window)
    return result


def pack_greedy(items: Sequence[Item], capacity: int) -> PackResult:
    """First window with enough room wins.  Fast, order dependent, leaves holes."""
    result = PackResult([], capacity)
    for item in items:
        cuts = _split_at_boundaries(item, capacity)
        # one split event per *extra* piece, so an unsplit item counts zero
        result.splits += len(cuts) - 1
        for offset, length in cuts:
            placed = False
            for window in result.windows:
                if capacity - window.used >= length:
                    window.add(item.item_id, offset, length)
                    placed = True
                    break
            if not placed:
                window = Window()
                window.add(item.item_id, offset, length)
                result.windows.append(window)
    return result


def pack_best_fit(items: Sequence[Item], capacity: int) -> PackResult:
    """Longest first, into the tightest window that still fits.

    Costs a sort and a scan per item, and returns noticeably fewer windows than
    greedy on a corpus with many short documents and a few long ones.
    """
    result = PackResult([], capacity)
    pieces: List[Tuple[str, int, int]] = []
    for item in items:
        cuts = _split_at_boundaries(item, capacity)
        if len(cuts) > 1:
            result.splits += len(cuts) - 1
        for offset, length in cuts:
            pieces.append((item.item_id, offset, length))

    pieces.sort(key=lambda p: (-p[2], p[0], p[1]))

    for item_id, offset, length in pieces:
        best_index: Optional[int] = None
        best_slack: Optional[int] = None
        for index, window in enumerate(result.windows):
            slack = capacity - window.used - length
            if slack < 0:
                continue
            if best_slack is None or slack < best_slack:
                best_index, best_slack = index, slack
                if slack == 0:
                    break
        if best_index is None:
            window = Window()
            window.add(item_id, offset, length)
            result.windows.append(window)
        else:
            result.windows[best_index].add(item_id, offset, length)
    return result


def pack_structure_preserving(items: Sequence[Item], capacity: int) -> PackResult:
    """Never co-pack two documents; split only at declared turn boundaries.

    Used for agentic traces.  Two trajectories in one attention window teaches
    the model that an unrelated tool observation is a natural continuation of
    the previous task, which is a specific and hard-to-unlearn failure.
    """
    result = PackResult([], capacity)
    for item in items:
        cuts = _split_at_boundaries(item, capacity)
        if len(cuts) > 1:
            result.splits += len(cuts) - 1
        for offset, length in cuts:
            window = Window()
            window.add(item.item_id, offset, length)
            result.windows.append(window)
    return result


def pack_long_context(items: Sequence[Item], capacity: int) -> PackResult:
    """One document per window, and only documents worth a long window.

    Two filters.  A document declaring `min_context` above the current window
    is deferred rather than truncated - a reasoning trace cut off before its
    verification step teaches the model to stop reasoning early.  A document
    too short to fill LONG_CONTEXT_MIN_FILL of the window is also deferred,
    because long-context batches are expensive and every unused position in one
    is a high-value training opportunity spent on padding.
    """
    result = PackResult([], capacity)
    threshold = int(capacity * LONG_CONTEXT_MIN_FILL)
    for item in items:
        if item.min_context > capacity:
            result.deferred.append(item.item_id)
            continue
        if item.size < threshold:
            result.deferred.append(item.item_id)
            continue
        cuts = _split_at_boundaries(item, capacity)
        if len(cuts) > 1:
            result.splits += len(cuts) - 1
        for offset, length in cuts:
            window = Window()
            window.add(item.item_id, offset, length)
            result.windows.append(window)
    return result


PACKERS: Dict[str, Callable[[Sequence[Item], int], PackResult]] = {
    "pad_only": pack_pad_only,
    "concat_chop": pack_concat_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
    "structure_preserving": pack_structure_preserving,
    "long_context": pack_long_context,
}


def pack(policy: str, items: Sequence[Item], capacity: int) -> PackResult:
    if policy not in PACKERS:
        raise ValueError(f"unknown packing policy {policy!r}; expected one of {POLICIES}")
    items = list(items)
    result = PACKERS[policy](items, capacity)
    result.input_tokens = sum(item.size for item in items)
    return result


def compare_policies(items: Sequence[Item], capacity: int) -> Dict[str, dict]:
    """Run every policy over the same items.

    This is what turns "best-fit is better" from an assertion into a measured
    claim in the packing report.
    """
    return {policy: pack(policy, items, capacity).stats() for policy in POLICIES}
