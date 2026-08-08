"""Loss masks, attention masks and position ids - and the checks that they are right.

The batch has to carry more than token ids; it has to carry the *training
meaning* of those ids.  Three arrays do that:

*   `loss_mask[i] == 1` means position i is a **target**: the model is graded on
    predicting token i given everything before it.  Position 0 of a segment is
    never a target, because there is nothing before it to condition on.  Padding
    is never a target.  For SFT-shaped data the prompt is never a target; for
    agentic data the user turn and the tool observation are never targets.

*   `segment_ids[i]` names which packed document position i belongs to.  The
    attention mask is causal **and** block-diagonal over these ids, so a
    document co-packed after another cannot attend back into it.  Without that,
    the model learns that unrelated text is a natural continuation.

*   `position_ids[i]` restarts at 0 for each segment, so a document packed
    second does not appear to have begun at position 900.

`validate_masks` is not decoration.  Every one of these arrays is easy to get
subtly wrong in a way that still trains and still produces a falling loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

PAD_SEGMENT = -1


@dataclass
class MaskValidation:
    ok: bool
    problems: List[str]

    def as_dict(self) -> dict:
        return {"ok": self.ok, "problems": self.problems}


def build_position_ids(segment_ids: Sequence[int]) -> List[int]:
    """Segment-relative positions: each packed document restarts at 0."""
    positions: List[int] = []
    counters: dict = {}
    for segment in segment_ids:
        if segment == PAD_SEGMENT:
            positions.append(0)
            continue
        counters[segment] = counters.get(segment, -1) + 1
        positions.append(counters[segment])
    return positions


def build_attention_mask(segment_ids: Sequence[int]) -> List[List[bool]]:
    """Dense causal, segment-confined mask.

    `mask[q][k]` is True when query position q may attend to key position k.
    Materialised densely only for validation and for the small toy model - a
    real implementation would pass segment ids to a fused kernel instead of
    building an L x L boolean array.
    """
    length = len(segment_ids)
    mask = [[False] * length for _ in range(length)]
    for q in range(length):
        if segment_ids[q] == PAD_SEGMENT:
            continue
        for k in range(q + 1):
            if segment_ids[k] == segment_ids[q]:
                mask[q][k] = True
    return mask


def first_positions(segment_ids: Sequence[int]) -> List[int]:
    """Indices that begin a segment - the positions that cannot be targets."""
    out: List[int] = []
    seen = set()
    for index, segment in enumerate(segment_ids):
        if segment == PAD_SEGMENT or segment in seen:
            continue
        seen.add(segment)
        out.append(index)
    return out


def validate_masks(
    token_ids: Sequence[int],
    loss_mask: Sequence[int],
    segment_ids: Sequence[int],
    position_ids: Sequence[int],
    pad_token_id: int,
    graded_flags: Optional[Sequence[bool]] = None,
) -> MaskValidation:
    problems: List[str] = []
    length = len(token_ids)

    if not (len(loss_mask) == len(segment_ids) == len(position_ids) == length):
        return MaskValidation(False, ["array lengths disagree"])

    # 1. Padding never carries loss, and is never inside a segment.
    for i in range(length):
        if token_ids[i] == pad_token_id and segment_ids[i] == PAD_SEGMENT:
            if loss_mask[i]:
                problems.append(f"position {i}: loss on a pad token")

    # 2. The first position of every segment is not a target.
    for i in first_positions(segment_ids):
        if loss_mask[i]:
            problems.append(f"position {i}: loss on the first token of a segment")

    # 3. Positions are segment-relative, contiguous and monotonic.
    counters: dict = {}
    for i, segment in enumerate(segment_ids):
        if segment == PAD_SEGMENT:
            if position_ids[i] != 0:
                problems.append(f"position {i}: pad should have position id 0")
            continue
        expected = counters.get(segment, -1) + 1
        counters[segment] = expected
        if position_ids[i] != expected:
            problems.append(
                f"position {i}: position id {position_ids[i]}, expected {expected}"
            )

    # 4. Segments are contiguous - a segment must not reappear after another
    #    segment interrupted it, or block-diagonal attention would be wrong.
    order: List[int] = []
    for segment in segment_ids:
        if segment == PAD_SEGMENT:
            continue
        if not order or order[-1] != segment:
            order.append(segment)
    if len(order) != len(set(order)):
        problems.append("segments are interleaved rather than contiguous")

    # 5. Padding is a suffix.  Holes in the middle would silently break the
    #    "pad is never attended" assumption the kernels rely on.
    seen_pad = False
    for i, segment in enumerate(segment_ids):
        if segment == PAD_SEGMENT:
            seen_pad = True
        elif seen_pad:
            problems.append(f"position {i}: real token after padding began")
            break

    # 6. Loss only where the data type says it is allowed.
    if graded_flags is not None:
        for i in range(length):
            if loss_mask[i] and not graded_flags[i]:
                problems.append(f"position {i}: loss on a context-only token")

    return MaskValidation(not problems, problems)


def check_attention_confinement(segment_ids: Sequence[int]) -> MaskValidation:
    """Assert no attention edge crosses a document boundary or looks forward."""
    mask = build_attention_mask(segment_ids)
    problems: List[str] = []
    for q, row in enumerate(mask):
        for k, allowed in enumerate(row):
            if not allowed:
                continue
            if k > q:
                problems.append(f"attention {q}<-{k} looks into the future")
            elif segment_ids[k] != segment_ids[q]:
                problems.append(f"attention {q}<-{k} crosses a document boundary")
            elif segment_ids[q] == PAD_SEGMENT:
                problems.append(f"attention {q}<-{k} originates in padding")
    return MaskValidation(not problems, problems[:20])
