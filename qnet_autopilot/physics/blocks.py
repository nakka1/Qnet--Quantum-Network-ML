"""Where to cut the key blocks, when the error rate moves during the run.

Every finite-key figure in this package so far assumed blocks of one fixed
duration: the link accumulates sifted bits for a second, settles up, starts
again. That is a modelling convenience, and the reports flagged it as one. A
real receiver chooses when to close a block, and the choice matters for a
reason that is specific to the finite-key bound: **a block is scored on its own
average error rate, and a stretch of bad fibre poisons every bit in the block it
lands in**. Ten good seconds and one bad second in the same block can be worth
nothing; the same eleven seconds cut into two blocks are worth ten seconds of
key.

Two strategies and two oracles
------------------------------
:func:`fixed_blocks` cuts on a timer, which is what phases 3 and 4 assumed.
:func:`adaptive_blocks` is a controller: it watches a noisy estimate of the
error rate — the one a receiver actually has, from the bits it publishes for
parameter estimation — and at every decision instant compares two futures,
closing the block when closing is worth more than carrying it. It has no tuning
constant. :func:`optimal_blocks` is the oracle: dynamic programming over the
true error series, choosing the partition that maximises total key with perfect
hindsight.

There is a second pair, and on a choppy link it is the one that matters. A
block does not have to be one stretch of fibre: privacy amplification runs over
a *set* of sifted bits, and the bound sees only how many there are and what
fraction of them erred. So a receiver can leave a noisy second out and settle
up on the rest as one block. ``adaptive_blocks(..., pooling=True)`` is the
controller allowed to do that, and :func:`pooled_optimal_blocks` is its oracle
— exact, and strictly above the interval one.

The oracle is not a proposal. It exists so the controller can be judged against
the best any strategy could do rather than against the fixed-timer baseline it
is bound to beat. It also has to be the *right* oracle: an earlier version of
this study searched only over fixed durations, one per run, capped at ten
seconds, concluded that variable blocking bought 1.28x at most, and recommended
against building any of this. That oracle cannot express the one thing a
variable block does — cut the good window out of a bad run — and the conclusion
drawn from it was wrong. On a badly compensated link a timer of any duration
above a second yields exactly zero bits, and the controller here yields
hundreds.

Why dynamic programming is exact here
-------------------------------------
The key extractable from a block depends only on where it starts and where it
ends — on the number of bits and their mean error rate, and on nothing outside
those bounds. So the total over a partition decomposes, and the best partition
of the first :math:`j` segments is the best over choices of where the last block
began. That is an :math:`O(n^2)` recursion with no approximation in it, which
is what makes the bound a real ceiling rather than a heuristic.

The pooled oracle is the same recursion with the segments sorted by error rate
first: an optimal grouping is then contiguous in *that* order, with the
discarded segments as a suffix. That is an exchange argument rather than a
decomposition, so it is checked against brute force over every grouping of a
small run instead of being asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .link import bb84_finite_key_length

__all__ = ["BlockPlan", "fixed_blocks", "adaptive_blocks", "optimal_blocks",
           "pooled_optimal_blocks"]


@dataclass(frozen=True)
class BlockPlan:
    """One way of cutting a run into key blocks, and what it yielded.

    A plan is a list of half-open ``[start, end)`` sample ranges — *pieces* —
    and a block is one or more of them. Pieces need not cover the run: a
    strategy may **discard** a stretch it judges unusable rather than fold it
    into a block, which is the whole point of adapting. The rate is always
    divided by the length of the run, not by the time spent inside blocks, so
    discarding never flatters a strategy.

    A block made of more than one piece is not a modelling liberty. Privacy
    amplification runs over a *set* of sifted bits; nothing in the finite-key
    bound requires them to have arrived consecutively. A receiver that knows
    which second was noisy can leave that second out and settle up on the rest
    as one block, and the strategies that do this are labelled so.

    Attributes:
        starts, ends: sample indices bounding each piece.
        key_bits: secret bits extracted from each **block**.
        dt_s: sampling interval of the error series the plan was built on.
        run_samples: length of the whole run, the denominator for the rate.
        strategy: which rule produced it, for reporting.
        labels: which block each piece belongs to, as an index into
            ``key_bits``. ``None`` means one piece per block, which is what
            every contiguous strategy produces.
    """

    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    key_bits: NDArray[np.float64]
    dt_s: float
    run_samples: int
    strategy: str = ""
    labels: NDArray[np.int64] | None = None

    @property
    def block_of_piece(self) -> NDArray[np.int64]:
        """Block index per piece, with the contiguous case filled in."""
        if self.labels is not None:
            return np.asarray(self.labels, dtype=np.int64)
        return np.arange(self.starts.size, dtype=np.int64)

    @property
    def count(self) -> int:
        """How many blocks were settled up on."""
        return int(self.key_bits.size)

    @property
    def pieces(self) -> int:
        """How many separate stretches of fibre those blocks were drawn from."""
        return int(self.starts.size)

    @property
    def piece_samples(self) -> NDArray[np.int64]:
        return (self.ends - self.starts).astype(np.int64)

    @property
    def durations_s(self) -> NDArray[np.float64]:
        """Fibre time inside each block, summed over its pieces."""
        if self.count == 0:
            return np.zeros(0)
        totals = np.bincount(self.block_of_piece, weights=self.piece_samples,
                             minlength=self.count)
        return totals * self.dt_s

    @property
    def total_key_bits(self) -> float:
        return float(np.sum(self.key_bits))

    @property
    def duration_s(self) -> float:
        return float(self.run_samples * self.dt_s)

    @property
    def mean_key_rate_hz(self) -> float:
        """Secret bits per second over the whole run, discarded time included."""
        return self.total_key_bits / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def duty(self) -> float:
        """Share of the run that sat inside some block."""
        if self.run_samples <= 0:
            return 0.0
        return float(np.sum(self.piece_samples) / self.run_samples)

    @property
    def empty_fraction(self) -> float:
        """Share of blocks that yielded nothing.

        Worth reporting on its own: a strategy can have a good mean and still be
        throwing away half its blocks, and those are the seconds during which a
        deployed link is producing no key at all.
        """
        return float(np.mean(self.key_bits <= 0.0)) if self.count else 0.0


def _segment_edges(samples: int, dt_s: float, granularity_s: float
                   ) -> NDArray[np.int64]:
    """Candidate block boundaries, on a coarser grid than the samples.

    A block may only start and end on one of these. The grid is what keeps the
    dynamic programme quadratic in a few hundred rather than in tens of
    thousands, and it costs nothing real: no receiver closes a block on a
    microsecond boundary either.
    """
    stride = max(int(round(granularity_s / dt_s)), 1)
    edges = np.arange(0, samples + 1, stride, dtype=np.int64)
    if edges[-1] != samples:
        edges = np.append(edges, samples)
    return edges


def _block_key_bits(prefix: NDArray[np.float64], starts: ArrayLike,
                    ends: ArrayLike, sifted_rate_hz: float, dt_s: float,
                    **finite_key_kwargs) -> NDArray[np.float64]:
    """Key from each ``[start, end)`` block, vectorised over the pairs.

    Args:
        prefix: cumulative sum of the error series, with a leading zero, so a
            block's mean error is one subtraction.
        starts, ends: sample indices, broadcast against each other.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        **finite_key_kwargs: passed to
            :func:`~qnet_autopilot.physics.link.bb84_finite_key_length`.

    Returns:
        Secret bits per block, never negative.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    lengths = (ends - starts).astype(float)
    return _key_from_totals(lengths, prefix[ends] - prefix[starts],
                            sifted_rate_hz, dt_s, **finite_key_kwargs)


def _key_from_totals(samples: ArrayLike, error_sum: ArrayLike,
                     sifted_rate_hz: float, dt_s: float,
                     **finite_key_kwargs) -> NDArray[np.float64]:
    """Key from a block described only by its size and its total error.

    The form every strategy here ultimately needs, and the reason a block does
    not have to be one interval: the bound sees a number of bits and a mean
    error rate, and is indifferent to when those bits arrived.
    """
    lengths = np.asarray(samples, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_qber = np.where(lengths > 0,
                             np.asarray(error_sum, dtype=float)
                             / np.maximum(lengths, 1), 1.0)
    block_bits = sifted_rate_hz * lengths * dt_s
    return bb84_finite_key_length(block_bits, mean_qber, **finite_key_kwargs)


def fixed_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                 duration_s: float, **finite_key_kwargs) -> BlockPlan:
    """Cut on a timer, which is what phases 3 and 4 assumed.

    Args:
        qber: error rate per sample.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        duration_s: block length. A trailing partial block is dropped rather
            than scored short, because a receiver would carry it into the next
            block rather than settle up on it.

    Returns:
        The plan.

    Raises:
        ValueError: on a non-positive duration or interval.
    """
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")

    error = np.asarray(qber, dtype=float)
    per_block = max(int(round(duration_s / dt_s)), 1)
    count = error.size // per_block
    if count == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, error.size,
                         strategy=f"fixo {duration_s:g} s")

    boundaries = np.arange(count + 1, dtype=np.int64) * per_block
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    key_bits = _block_key_bits(prefix, boundaries[:-1], boundaries[1:],
                               sifted_rate_hz, dt_s, **finite_key_kwargs)
    return BlockPlan(boundaries[:-1], boundaries[1:], key_bits, dt_s,
                     error.size, strategy=f"fixo {duration_s:g} s")


def optimal_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                   granularity_s: float = 0.1,
                   max_block_s: float | None = None,
                   **finite_key_kwargs) -> BlockPlan:
    """The partition that maximises total key, with perfect hindsight.

    Args:
        qber: error rate per sample — the *true* one, which is what makes this
            an oracle and not a proposal.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        granularity_s: candidate boundaries are on this grid.
        max_block_s: cap on block length, if the application has one.
        **finite_key_kwargs: passed through to the finite-key bound.

    Returns:
        The plan. Its total is an upper bound on every strategy that cuts on
        this grid, causal or not.

    Note:
        The recursion is exact because a block's key depends only on its own
        endpoints. That is a property of the finite-key formula, not an
        assumption: privacy amplification is applied per block, and no term
        couples one block to another.
    """
    error = np.asarray(qber, dtype=float)
    if error.size == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, 0, strategy="oráculo")

    edges = _segment_edges(error.size, dt_s, granularity_s)
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    segments = edges.size

    maximum_samples = (error.size if max_block_s is None
                       else max(int(round(max_block_s / dt_s)), 1))

    best = np.full(segments, -np.inf)
    best[0] = 0.0
    previous = np.zeros(segments, dtype=np.int64)

    for end in range(1, segments):
        reachable = edges[end] - edges[:end] <= maximum_samples
        candidates = np.flatnonzero(reachable)
        if candidates.size == 0:
            candidates = np.array([end - 1])
        gains = _block_key_bits(prefix, edges[candidates],
                                np.full(candidates.size, edges[end]),
                                sifted_rate_hz, dt_s, **finite_key_kwargs)
        totals = best[candidates] + gains
        winner = int(np.argmax(totals))
        best[end] = totals[winner]
        previous[end] = candidates[winner]

    boundaries = [segments - 1]
    while boundaries[-1] != 0:
        boundaries.append(int(previous[boundaries[-1]]))
    indices = np.array(sorted(boundaries), dtype=np.int64)
    cut = edges[indices]
    key_bits = _block_key_bits(prefix, cut[:-1], cut[1:], sifted_rate_hz, dt_s,
                               **finite_key_kwargs)
    # A block the recursion left empty is a stretch it chose to write off, and
    # reporting it as a block would understate the duty cycle of the ones that
    # actually produced. Dropping them changes no total.
    keep = key_bits > 0.0
    if not np.any(keep):
        keep = np.ones_like(key_bits, dtype=bool)
    return BlockPlan(cut[:-1][keep], cut[1:][keep], key_bits[keep], dt_s,
                     error.size, strategy="oráculo")


def pooled_optimal_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                          granularity_s: float = 0.1,
                          max_block_s: float | None = None,
                          **finite_key_kwargs) -> BlockPlan:
    """The ceiling when a block does not have to be one stretch of fibre.

    :func:`optimal_blocks` partitions the run into intervals, which is what a
    receiver that settles up on a clock can do. But privacy amplification runs
    over a *set* of sifted bits, and the bound sees only how many there are and
    what fraction of them erred. A receiver that knows which second was noisy
    can leave that second out and settle up on the rest as one block. That
    strategy set is strictly larger, so this oracle is strictly above the
    interval one, and it is the honest ceiling for a controller that is allowed
    to skip.

    The search is still exact and still quadratic. Sort the segments by error
    rate; an optimal solution then groups them into runs that are contiguous
    *in that order*, with the discarded ones forming a suffix. Given that, the
    same recursion applies, over sorted position instead of over time.

    Args:
        qber: the true error rate per sample.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        granularity_s: the grid segments are cut on.
        max_block_s: cap on the fibre time inside one block, if the
            application has one.
        **finite_key_kwargs: passed through to the finite-key bound.

    Returns:
        The plan, with ``labels`` marking which pieces share a block.

    Note:
        That an optimal grouping is contiguous in sorted order is an exchange
        argument, not an assumption inherited from somewhere: the tests check
        it against brute force over every partition of a small run, which is
        the only way to find out whether the argument survives the finite-size
        terms.
    """
    error = np.asarray(qber, dtype=float)
    if error.size == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, 0, strategy="oráculo agrupado")

    edges = _segment_edges(error.size, dt_s, granularity_s)
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    starts, ends = edges[:-1], edges[1:]
    samples = (ends - starts).astype(float)
    sums = prefix[ends] - prefix[starts]
    order = np.argsort(np.where(samples > 0, sums / np.maximum(samples, 1), 1.0),
                       kind="stable")

    sorted_samples = np.concatenate([[0.0], np.cumsum(samples[order])])
    sorted_sums = np.concatenate([[0.0], np.cumsum(sums[order])])
    count = order.size
    maximum_samples = (float(error.size) if max_block_s is None
                       else max(max_block_s / dt_s, 1.0))

    best = np.full(count + 1, -np.inf)
    best[0] = 0.0
    previous = np.zeros(count + 1, dtype=np.int64)

    for end in range(1, count + 1):
        # Option one: leave the segment out altogether.
        best[end] = best[end - 1]
        previous[end] = end - 1
        # Option two: it closes a block that started somewhere earlier in the
        # sorted order.
        lengths = sorted_samples[end] - sorted_samples[:end]
        reachable = np.flatnonzero(lengths <= maximum_samples)
        if reachable.size == 0:
            continue
        gains = _key_from_totals(lengths[reachable],
                                 sorted_sums[end] - sorted_sums[reachable],
                                 sifted_rate_hz, dt_s, **finite_key_kwargs)
        totals = best[reachable] + gains
        winner = int(np.argmax(totals))
        if totals[winner] > best[end]:
            best[end] = totals[winner]
            previous[end] = reachable[winner]

    # Walk the choices back. A step of exactly one that was taken as "leave it
    # out" and one taken as a one-segment block are told apart by re-scoring.
    groups: list[NDArray[np.int64]] = []
    position = count
    while position > 0:
        earlier = int(previous[position])
        if earlier == position - 1 and best[position] == best[position - 1]:
            position = earlier
            continue
        groups.append(order[earlier:position])
        position = earlier

    if not groups:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, error.size,
                         strategy="oráculo agrupado")

    piece_starts: list[int] = []
    piece_ends: list[int] = []
    labels: list[int] = []
    key_bits: list[float] = []
    for index, group in enumerate(reversed(groups)):
        for segment in np.sort(group):
            piece_starts.append(int(starts[segment]))
            piece_ends.append(int(ends[segment]))
            labels.append(index)
        key_bits.append(float(_key_from_totals(float(samples[group].sum()),
                                               float(sums[group].sum()),
                                               sifted_rate_hz, dt_s,
                                               **finite_key_kwargs)))

    return _merge_pieces(np.array(piece_starts, dtype=np.int64),
                         np.array(piece_ends, dtype=np.int64),
                         np.array(labels, dtype=np.int64),
                         np.array(key_bits, dtype=float), dt_s, error.size,
                         "oráculo agrupado")


def _merge_pieces(starts: NDArray[np.int64], ends: NDArray[np.int64],
                  labels: NDArray[np.int64], key_bits: NDArray[np.float64],
                  dt_s: float, run_samples: int, strategy: str) -> BlockPlan:
    """Join pieces of the same block that happen to be adjacent in time.

    Cosmetic for the totals and not for the reporting: a block that turned out
    to be one unbroken stretch should be reported as one piece, so that the
    piece count means what it says — how fragmented the block really was.
    """
    keep_start, keep_end, keep_label = [], [], []
    for start, end, label in zip(starts, ends, labels):
        if keep_label and keep_label[-1] == label and keep_end[-1] == start:
            keep_end[-1] = int(end)
            continue
        keep_start.append(int(start))
        keep_end.append(int(end))
        keep_label.append(int(label))
    return BlockPlan(np.array(keep_start, dtype=np.int64),
                     np.array(keep_end, dtype=np.int64), key_bits, dt_s,
                     run_samples, strategy=strategy,
                     labels=np.array(keep_label, dtype=np.int64))



def _forecast_horizon(recent, horizon_samples: int,
                      disclosed_per_sample: float, key_of
                      ) -> tuple[float, float, float, float]:
    """What a horizon of fibre like the recent past is worth, and how to take it.

    Args:
        recent: ``(end, tested, errors)`` per recently measured segment — the
            published bits and how many of them erred.
        horizon_samples: the block budget, in samples.
        disclosed_per_sample: published bits per sample of fibre time.
        key_of: the controller's own estimate of the finite-key bound.

    Returns:
        The share of the horizon worth putting in a block, the error rate that
        share has, the key per sample of horizon that share yields, and — for a
        controller that cannot skip — the key per sample of taking all of it.
        On a link whose segments all look alike the two rates coincide; on a
        link that alternates between good and unusable, the selected one is
        much larger, and that gap is exactly what pooling buys.

    Note:
        The selected rate is only reachable by a strategy allowed to leave
        segments out of an open block. Handing it to a contiguous controller
        makes closing look better than it is, and measurably costs key: it
        banks a block expecting to restart at a quality it cannot actually
        cherry-pick.
    """
    if not recent:
        return 1.0, 1.0, 0.0, 0.0
    tested = np.array([entry[1] for entry in recent], dtype=float)
    errors = np.array([entry[2] for entry in recent], dtype=float)
    usable = tested > 0
    if not np.any(usable):
        return 1.0, 1.0, 0.0, 0.0
    tested, errors = tested[usable], errors[usable]

    order = np.argsort(errors / tested, kind="stable")
    published = np.cumsum(tested[order])
    failed = np.cumsum(errors[order])
    share = published / published[-1]
    samples = share * horizon_samples
    keys = key_of(samples, disclosed_per_sample * samples,
                  disclosed_per_sample * samples * (failed / published))
    winner = int(np.argmax(keys))
    return (float(share[winner]), float(failed[winner] / published[winner]),
            float(keys[winner]) / horizon_samples,
            float(keys[-1]) / horizon_samples)


def adaptive_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                    granularity_s: float = 0.1,
                    estimation_fraction: float = 0.1,
                    abort_qber: float = 0.11,
                    max_block_s: float = 20.0,
                    forecast_window_s: float | None = None,
                    pooling: bool = False,
                    rng: np.random.Generator | int | None = None,
                    **finite_key_kwargs) -> BlockPlan:
    """A causal controller: at every instant, take the continuation worth most.

    The obvious rule — close when the newest segment would lower the block's
    key — is myopic and measurably wrong. It closes on any excursion, and each
    close makes the next block pay the finite-size cost over again: on a
    well-compensated link it gives up about a fifth of the key by cutting into
    two-second blocks where one twenty-second block was available. What the
    controller has to compare is not two blocks but two *futures*.

    So it plans. At every decision instant it knows three things, all of them
    from bits the protocol has already published for parameter estimation:

    * ``q_block`` — the error rate of the block it is holding open;
    * ``q_segment`` — the error rate of the segment just measured;
    * ``q_link`` — the error rate of the link over the recent past, which is
      its forecast of what the remaining fibre will look like.

    From ``q_link`` it computes ``R``, the key rate a stationary link of that
    quality would sustain under its own block budget, and then values the
    options over the horizon the budget defines:

    * **close now** — bank the block as it stands, and run at ``R`` for the
      rest of the horizon;
    * **keep it open** — carry the block, this segment included, to whichever
      later closing point is worth most, and run at ``R`` after that;
    * **skip the segment** — only with ``pooling``: hold the block open but
      leave this segment out of it, and carry on.

    All of the behaviour falls out of that comparison, with no tuning constant:

    * **A quiet link is left alone.** ``R`` is high, so throwing away a
      part-grown block to restart at the same quality gains nothing, and the
      controller extends. It converges on one long block, which is what the
      oracle does too, and a single noisy segment cannot talk it out of that
      because the option value is taken over the *best* later closing point,
      not over closing at once.
    * **A bad link is cut and skipped.** When ``q_link`` is above the
      threshold, ``R`` is zero: the future is worth nothing, and the only value
      on the table is the block already in hand. Any segment that would lower
      it closes it, and a segment estimated above ``abort_qber`` opens no new
      block, so the bad stretch is discarded rather than averaged in. On a link
      that spends most of its time above the threshold, that is the difference
      between some key and none — no fixed timer, of any duration, produces a
      single bit there.

    Args:
        qber: the true error rate per sample. The controller never sees it —
            it sees the binomial draws taken from it below.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        granularity_s: how often the controller may act, and the grid the
            planned closing points are searched on.
        estimation_fraction: share of the sifted bits published as they are
            produced, which is what the controller estimates from. This is the
            *monitoring* budget, not the test fraction the bound optimises
            internally: a receiver discloses a running subset to watch the link
            and settles the accounting at the end of the block. The noise is
            the difficulty — a segment that looks bad may only have been
            unlucky, and closing on it costs real key.
        abort_qber: a segment estimated above this cannot contribute to any
            block, so none is opened on it. The BB84 threshold is the natural
            value.
        max_block_s: the block budget — the longest a block may stay open, and
            therefore also the horizon the controller plans over. A deployment
            with a key-delivery deadline sets it from that deadline.
        forecast_window_s: how far back the link is treated as stationary when
            forecasting the rest of the horizon. ``None`` uses the whole run so
            far, which is the right answer only if the link is stationary; on a
            link that changes regime, a window is both more realistic and
            better.
        pooling: allow a block to skip a bad segment and stay open, rather than
            having to close in order to exclude it. Nothing in the finite-key
            bound requires a block's bits to be consecutive, and the pieces are
            reported separately so the fragmentation is visible.
        rng: seeds the estimation noise.
        **finite_key_kwargs: passed through to the finite-key bound, both for
            the controller's own forecast and for the scoring.

    Returns:
        The plan. Blocks need not tile the run; discarded stretches are absent
        from it and still count in the denominator of the rate.

    Note:
        The controller decides on estimates and is scored on the truth. A rule
        judged on the same noisy numbers it acts on would hide the cost of
        reacting to noise, which is most of what separates a controller from an
        oracle.
    """
    error = np.asarray(qber, dtype=float)
    strategy = "adaptativo agrupado" if pooling else "adaptativo"
    if error.size == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, 0, strategy=strategy)
    if max_block_s <= 0 or granularity_s <= 0 or dt_s <= 0:
        raise ValueError("max_block_s, granularity_s and dt_s must be positive")

    generator = np.random.default_rng(rng)
    edges = _segment_edges(error.size, dt_s, granularity_s)
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    maximum_samples = max(int(round(max_block_s / dt_s)), 1)
    window_samples = (None if forecast_window_s is None
                      else max(int(round(forecast_window_s / dt_s)), 1))
    #: Bits published per sample of fibre time — the controller's whole view.
    disclosed_per_sample = sifted_rate_hz * dt_s * estimation_fraction

    def disclose(start: int, end: int) -> tuple[float, float]:
        """Bits published over ``[start, end)`` and how many of them erred."""
        samples = end - start
        if samples <= 0:
            return 0.0, 0.0
        truth = float(np.clip((prefix[end] - prefix[start]) / samples, 0.0, 1.0))
        tested = int(disclosed_per_sample * samples)
        if tested < 1:
            # Too few published bits to estimate from; the controller is blind
            # and can only take the segment at face value.
            return 1.0, truth
        return float(tested), float(generator.binomial(tested, truth))

    def key_of(samples: ArrayLike, tested: ArrayLike, errors: ArrayLike
               ) -> NDArray[np.float64]:
        """The bound the controller believes it would settle up with."""
        length = np.asarray(samples, dtype=float)
        published = np.asarray(tested, dtype=float)
        observed = np.divide(np.asarray(errors, dtype=float),
                             np.where(published > 0, published, 1.0))
        keys = bb84_finite_key_length(sifted_rate_hz * length * dt_s, observed,
                                      **finite_key_kwargs)
        return np.where((length > 0) & (published > 0), keys, 0.0)

    plan_starts: list[int] = []
    plan_ends: list[int] = []
    plan_labels: list[int] = []
    block_index = 0

    open_at: int | None = None
    pieces: list[list[int]] = []
    block_samples = 0
    block_tested = block_errors = 0.0
    # The forecast is over published bits, so the window is kept as a running
    # list of what each segment contributed rather than as a mean.
    history: list[tuple[int, float, float]] = []

    def reset() -> None:
        nonlocal open_at, pieces, block_samples, block_tested, block_errors
        open_at, pieces = None, []
        block_samples = 0
        block_tested = block_errors = 0.0

    def close() -> None:
        """Settle up, and only keep what the controller believes is worth it.

        A block whose own estimate says it will yield nothing is never
        processed by a receiver — the bits are dropped rather than run through
        privacy amplification. Keeping it here would be scoring the controller
        on a decision it would not make. The estimate is the controller's, so a
        block it wrongly writes off is a real loss and shows up as one.
        """
        nonlocal block_index
        if pieces and float(key_of(block_samples, block_tested,
                                   block_errors)) > 0.0:
            for start, end in pieces:
                plan_starts.append(start)
                plan_ends.append(end)
                plan_labels.append(block_index)
            block_index += 1
        reset()

    def admit(start: int, end: int, tested: float, errors: float) -> None:
        nonlocal block_samples, block_tested, block_errors
        if pieces and pieces[-1][1] == start:
            pieces[-1][1] = end
        else:
            pieces.append([start, end])
        block_samples += end - start
        block_tested += tested
        block_errors += errors

    for index in range(1, edges.size):
        segment_start, segment_end = int(edges[index - 1]), int(edges[index])
        tested, errors = disclose(segment_start, segment_end)
        history.append((segment_end, tested, errors))
        hopeless = tested > 0 and errors / tested > abort_qber

        if open_at is None:
            if hopeless:
                continue
            open_at = segment_start

        if block_samples > 0:
            # What the rest of the horizon is worth if the link goes on
            # behaving as it recently has. Not "as good as its average": a
            # choppy link is bimodal, and a receiver that can leave the bad
            # windows out is worth what its *good* windows are worth. So the
            # forecast is a pair — how much of the fibre is worth taking, and
            # how clean that part is — found by the same sorted-prefix search
            # the pooled oracle runs, on the recent segments instead of on the
            # truth. On a stationary link the search returns "take all of it"
            # and the pair collapses to the average.
            if window_samples is None:
                recent = history
            else:
                floor = segment_end - window_samples
                recent = [entry for entry in history if entry[0] > floor]
            seen = sum(entry[1] for entry in recent)
            failed = sum(entry[2] for entry in recent)
            forecast = failed / seen if seen > 0 else 1.0
            duty, selected, pooled_rate, plain_rate = _forecast_horizon(
                recent, maximum_samples, disclosed_per_sample, key_of)
            per_sample = pooled_rate if pooling else plain_rate

            horizon_end = open_at + maximum_samples
            banked = float(key_of(block_samples, block_tested, block_errors))
            closing = banked + per_sample * (horizon_end - segment_start)

            # Keeping it open: carry the block through this segment to the best
            # closing point the grid offers, filling what follows the segment
            # with fibre at the forecast quality.
            later = edges[(edges >= segment_end) & (edges <= horizon_end)]
            if later.size == 0:
                later = np.array([segment_end], dtype=np.int64)
            span = (later - segment_end).astype(float)
            # A contiguous block that runs on absorbs whatever comes, good and
            # bad, so it fills at the plain average. One that may skip fills
            # only with the part it would take, at the quality that part has.
            filled = duty * span if pooling else span
            fill_error = selected if pooling else forecast

            def carry(extra_samples: float, extra_tested: float,
                      extra_errors: float) -> float:
                keys = key_of(block_samples + extra_samples + filled,
                              block_tested + extra_tested
                              + disclosed_per_sample * filled,
                              block_errors + extra_errors
                              + disclosed_per_sample * filled * fill_error)
                return float(np.max(keys + per_sample * (horizon_end - later)))

            keeping = carry(segment_end - segment_start, tested, errors)
            skipping = carry(0.0, 0.0, 0.0) if pooling else -np.inf

            # ``>=`` rather than ``>`` so that a tie closes. Ties happen in one
            # place only — every option worth nothing, which is a block that
            # has gone bad on a link whose future is also bad — and there the
            # right move is to drop it and start clean rather than keep
            # accumulating into a block that can no longer pay.
            if closing >= max(keeping, skipping):
                close()
                if hopeless:
                    continue
                open_at = segment_start
            elif skipping > keeping:
                if segment_end - open_at >= maximum_samples:
                    close()
                continue

        admit(segment_start, segment_end, tested, errors)
        if segment_end - open_at >= maximum_samples:
            close()

    close()

    if not plan_starts:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, error.size, strategy=strategy)

    starts = np.array(plan_starts, dtype=np.int64)
    ends = np.array(plan_ends, dtype=np.int64)
    labels = np.array(plan_labels, dtype=np.int64)
    samples = np.bincount(labels, weights=(ends - starts).astype(float))
    sums = np.bincount(labels, weights=prefix[ends] - prefix[starts])
    key_bits = _key_from_totals(samples, sums, sifted_rate_hz, dt_s,
                                **finite_key_kwargs)
    return _merge_pieces(starts, ends, labels, key_bits, dt_s, error.size,
                         strategy)
