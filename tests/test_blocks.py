"""Where the key blocks get cut, and whether the controller earns its keep.

The finite-key bound charges a block's errors against every bit in it, so the
boundary is not bookkeeping — it decides whether a run produces key at all.
Three strategies share one interface here, and most of what can go wrong is a
strategy that looks good because it is being scored generously:

* an oracle that is not actually an upper bound is worthless as a ceiling, so
  it is checked against exhaustive enumeration on a small case and against
  every other strategy on large ones;
* a controller that peeks at the truth would beat the oracle occasionally, and
  that is the sharpest available test that it is not peeking;
* a rate divided by the time spent *inside* blocks rather than by the run would
  reward throwing fibre away, so discarding is checked to cost what it costs.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qnet_autopilot.physics.blocks import (BlockPlan, _key_from_totals,
                                           _segment_edges, adaptive_blocks,
                                           fixed_blocks, optimal_blocks,
                                           pooled_optimal_blocks)

#: A link making a hundred thousand sifted bits a second, sampled every
#: millisecond — the phase-3 operating point.
RATE_HZ = 1.0e5
DT_S = 1.0e-3


def _series(pattern, dt_s: float = DT_S) -> np.ndarray:
    """A QBER series from ``(duration_s, qber)`` pairs."""
    parts = [np.full(int(round(duration / dt_s)), value)
             for duration, value in pattern]
    return np.concatenate(parts)


class TestFixedBlocks:

    def test_it_cuts_the_number_of_blocks_the_timer_implies(self):
        plan = fixed_blocks(_series([(10.0, 0.02)]), RATE_HZ, DT_S, 2.0)
        assert plan.count == 5
        assert np.allclose(plan.durations_s, 2.0)
        assert plan.duty == pytest.approx(1.0)

    def test_it_drops_the_trailing_partial_block(self):
        """A receiver carries the remainder into the next block, not into this
        one — scoring it short would credit the timer with a block it never
        settled up on."""
        plan = fixed_blocks(_series([(10.5, 0.02)]), RATE_HZ, DT_S, 2.0)
        assert plan.count == 5
        assert plan.duty == pytest.approx(10.0 / 10.5, rel=1e-9)
        # The dropped half second still counts against the rate.
        assert plan.duration_s == pytest.approx(10.5)

    def test_a_run_shorter_than_one_block_yields_nothing(self):
        plan = fixed_blocks(_series([(0.5, 0.01)]), RATE_HZ, DT_S, 2.0)
        assert plan.count == 0
        assert plan.total_key_bits == 0.0
        assert plan.mean_key_rate_hz == 0.0

    def test_it_refuses_a_non_positive_duration(self):
        for duration, dt in ((0.0, DT_S), (-1.0, DT_S), (1.0, 0.0)):
            with pytest.raises(ValueError):
                fixed_blocks(_series([(1.0, 0.01)]), RATE_HZ, dt, duration)

    def test_longer_blocks_pay_on_a_stationary_link(self):
        """The finite-size cost falls with block length, so on a link that never
        moves the longest timer wins. This is the baseline the controller has to
        reproduce without being told the link is quiet."""
        qber = _series([(20.0, 0.01)])
        rates = [fixed_blocks(qber, RATE_HZ, DT_S, duration).mean_key_rate_hz
                 for duration in (0.1, 1.0, 5.0, 20.0)]
        assert rates == sorted(rates)


class TestOracle:

    def test_it_matches_exhaustive_enumeration(self):
        """The dynamic programme against brute force over every partition.

        Six segments is 32 partitions, which is cheap to enumerate and enough
        to catch an off-by-one in the recursion or a boundary the DP cannot
        reach.
        """
        qber = _series([(0.2, 0.01), (0.2, 0.15), (0.2, 0.01),
                        (0.2, 0.2), (0.2, 0.005), (0.2, 0.01)])
        plan = optimal_blocks(qber, RATE_HZ, DT_S, granularity_s=0.2)

        edges = np.arange(0, qber.size + 1, int(round(0.2 / DT_S)))
        prefix = np.concatenate([[0.0], np.cumsum(qber)])
        from qnet_autopilot.physics.blocks import _block_key_bits

        best = -np.inf
        interior = list(edges[1:-1])
        for size in range(len(interior) + 1):
            for chosen in itertools.combinations(interior, size):
                cut = np.array([edges[0], *chosen, edges[-1]], dtype=np.int64)
                total = float(np.sum(_block_key_bits(prefix, cut[:-1], cut[1:],
                                                     RATE_HZ, DT_S)))
                best = max(best, total)
        assert plan.total_key_bits == pytest.approx(best, rel=1e-9)

    def test_it_is_a_ceiling_over_the_timers(self):
        """Any timer whose blocks land on the oracle's grid is a partition the
        oracle could have chosen, so it can never do better."""
        qber = _series([(3.0, 0.005), (0.4, 0.25), (6.0, 0.01),
                        (0.6, 0.3), (5.0, 0.02), (5.0, 0.06)])
        oracle = optimal_blocks(qber, RATE_HZ, DT_S, granularity_s=0.1)
        for duration in (0.1, 0.5, 1.0, 5.0, 10.0, 20.0):
            timer = fixed_blocks(qber, RATE_HZ, DT_S, duration)
            assert timer.total_key_bits <= oracle.total_key_bits + 1e-6

    def test_it_leaves_the_hopeless_stretches_out(self):
        """A run that is good for two seconds and unusable for eight should be
        two seconds of key and eight seconds of silence, not ten seconds of
        nothing."""
        qber = _series([(2.0, 0.01), (8.0, 0.30)])
        oracle = optimal_blocks(qber, RATE_HZ, DT_S, granularity_s=0.1)
        assert oracle.total_key_bits > 0.0
        assert oracle.duty == pytest.approx(0.2, abs=0.02)
        assert fixed_blocks(qber, RATE_HZ, DT_S, 10.0).total_key_bits == 0.0

    def test_a_block_cap_can_only_cost(self):
        qber = _series([(10.0, 0.01)])
        uncapped = optimal_blocks(qber, RATE_HZ, DT_S)
        capped = optimal_blocks(qber, RATE_HZ, DT_S, max_block_s=1.0)
        assert capped.total_key_bits <= uncapped.total_key_bits
        assert np.all(capped.durations_s <= 1.0 + 1e-9)

    def test_an_empty_run_is_an_empty_plan(self):
        plan = optimal_blocks(np.zeros(0), RATE_HZ, DT_S)
        assert plan.count == 0
        assert plan.mean_key_rate_hz == 0.0
        assert plan.duty == 0.0


class TestPooledOracle:
    """The ceiling when a block need not be one stretch of fibre.

    The whole construction rests on one claim — that an optimal grouping is
    contiguous in *sorted* order, so the same quadratic recursion applies. That
    is an exchange argument, and exchange arguments are exactly the kind of
    thing the finite-size terms can break, so it is checked against brute force
    rather than asserted.
    """

    @staticmethod
    def _brute_force(samples, sums) -> float:
        """Every way of grouping the segments, discards included."""
        best = 0.0

        def walk(index, groups):
            nonlocal best
            if index == len(samples):
                best = max(best, sum(
                    float(_key_from_totals(sum(samples[j] for j in group),
                                           sum(sums[j] for j in group),
                                           RATE_HZ, DT_S))
                    for group in groups))
                return
            walk(index + 1, groups)                       # leave it out
            for position in range(len(groups)):
                walk(index + 1, groups[:position]
                     + [groups[position] + [index]] + groups[position + 1:])
            walk(index + 1, groups + [[index]])           # a block of its own

        walk(0, [])
        return best

    def test_it_matches_brute_force_over_every_grouping(self):
        rng = np.random.default_rng(0)
        for _ in range(6):
            levels = rng.choice([0.004, 0.01, 0.02, 0.05, 0.09, 0.14, 0.25],
                                size=6)
            qber = _series([(0.2, float(level)) for level in levels])
            plan = pooled_optimal_blocks(qber, RATE_HZ, DT_S, granularity_s=0.2)

            edges = _segment_edges(qber.size, DT_S, 0.2)
            prefix = np.concatenate([[0.0], np.cumsum(qber)])
            samples = [float(edges[i + 1] - edges[i]) for i in range(edges.size - 1)]
            sums = [float(prefix[edges[i + 1]] - prefix[edges[i]])
                    for i in range(edges.size - 1)]
            assert plan.total_key_bits == pytest.approx(
                self._brute_force(samples, sums), rel=1e-9)

    def test_it_is_never_below_the_interval_oracle(self):
        """Every interval partition is also a grouping, so this cannot lose."""
        rng = np.random.default_rng(7)
        for _ in range(6):
            pattern = [(float(rng.uniform(0.3, 2.0)),
                        float(rng.choice([0.005, 0.02, 0.05, 0.12, 0.3])))
                       for _ in range(10)]
            qber = _series(pattern)
            assert (pooled_optimal_blocks(qber, RATE_HZ, DT_S).total_key_bits
                    >= optimal_blocks(qber, RATE_HZ, DT_S).total_key_bits - 1e-6)

    def test_it_pools_two_good_windows_across_a_bad_one(self):
        """The case the interval oracle cannot express: one block, two pieces."""
        qber = _series([(0.4, 0.01), (0.2, 0.30), (0.4, 0.01)])
        plan = pooled_optimal_blocks(qber, RATE_HZ, DT_S)
        assert plan.count == 1
        assert plan.pieces == 2
        assert plan.durations_s[0] == pytest.approx(0.8, abs=0.05)
        assert plan.duty == pytest.approx(0.8, abs=0.05)

    def test_adjacent_pieces_of_one_block_are_reported_as_one(self):
        """A block that turned out unbroken should not look fragmented."""
        plan = pooled_optimal_blocks(_series([(4.0, 0.01)]), RATE_HZ, DT_S)
        assert plan.count == 1
        assert plan.pieces == 1

    def test_a_block_cap_can_only_cost(self):
        qber = _series([(1.0, 0.01), (0.3, 0.25), (2.0, 0.015)])
        uncapped = pooled_optimal_blocks(qber, RATE_HZ, DT_S)
        capped = pooled_optimal_blocks(qber, RATE_HZ, DT_S, max_block_s=0.5)
        assert capped.total_key_bits <= uncapped.total_key_bits + 1e-6
        assert np.all(capped.durations_s <= 0.5 + 1e-9)

    def test_an_empty_run_is_an_empty_plan(self):
        plan = pooled_optimal_blocks(np.zeros(0), RATE_HZ, DT_S)
        assert plan.count == 0 and plan.pieces == 0
        assert plan.mean_key_rate_hz == 0.0


class TestAdaptiveController:

    def test_it_never_beats_the_oracle(self):
        """The one test that would catch a controller reading the truth.

        It decides on binomial draws and is scored on the series those draws
        came from; a plan built from the truth would sit at the ceiling instead
        of under it.
        """
        rng = np.random.default_rng(4)
        for _ in range(6):
            pattern = [(float(rng.uniform(0.5, 3.0)),
                        float(rng.choice([0.005, 0.02, 0.05, 0.12, 0.3])))
                       for _ in range(8)]
            qber = _series(pattern)
            controller = adaptive_blocks(qber, RATE_HZ, DT_S, rng=1,
                                         max_block_s=float(qber.size * DT_S))
            oracle = optimal_blocks(qber, RATE_HZ, DT_S)
            assert controller.total_key_bits <= oracle.total_key_bits + 1e-6

    def test_it_produces_key_where_no_timer_can(self):
        """The finding the fixed-duration assumption was hiding.

        A link that is briefly good and mostly unusable yields exactly zero
        under a timer of any length that averages a good window together with a
        bad one. The controller cuts the window out and banks it.
        """
        qber = _series([(0.4, 0.01), (2.0, 0.30)] * 5)
        for duration in (1.0, 2.0, 4.0, 12.0):
            assert fixed_blocks(qber, RATE_HZ, DT_S, duration).total_key_bits == 0.0
        controller = adaptive_blocks(qber, RATE_HZ, DT_S, rng=0,
                                     max_block_s=float(qber.size * DT_S))
        assert controller.total_key_bits > 0.0
        # It bought that key by refusing most of the run, and the rate is still
        # divided by the whole run.
        assert controller.duty < 0.4
        assert controller.duration_s == pytest.approx(12.0)

    def test_it_leaves_a_quiet_link_in_one_long_block(self):
        """The failure mode of every threshold rule tried before this one: the
        estimate is noisy, so a rule that closes when a segment looks worse than
        the block closes constantly, and each close pays the finite-size cost
        again."""
        qber = _series([(20.0, 0.01)])
        controller = adaptive_blocks(qber, RATE_HZ, DT_S, rng=3, max_block_s=20.0)
        timer = fixed_blocks(qber, RATE_HZ, DT_S, 20.0)
        assert controller.count <= 2
        assert controller.duty > 0.95
        assert controller.total_key_bits > 0.9 * timer.total_key_bits

    def test_noise_alone_does_not_close_a_block(self):
        """Same quiet link, ten different draws of the estimation noise."""
        qber = _series([(20.0, 0.01)])
        counts = [adaptive_blocks(qber, RATE_HZ, DT_S, rng=seed,
                                  max_block_s=20.0).count for seed in range(10)]
        assert max(counts) <= 3

    def test_it_honours_the_block_budget(self):
        qber = _series([(20.0, 0.01)])
        controller = adaptive_blocks(qber, RATE_HZ, DT_S, rng=0, max_block_s=2.0)
        assert np.all(controller.durations_s <= 2.0 + 1e-9)
        assert controller.count >= 8

    def test_the_same_seed_gives_the_same_plan(self):
        qber = _series([(2.0, 0.01), (1.0, 0.2), (3.0, 0.03)])
        first = adaptive_blocks(qber, RATE_HZ, DT_S, rng=11)
        second = adaptive_blocks(qber, RATE_HZ, DT_S, rng=11)
        assert np.array_equal(first.starts, second.starts)
        assert np.array_equal(first.ends, second.ends)

    def test_its_blocks_are_ordered_and_disjoint(self):
        qber = _series([(1.0, 0.01), (0.5, 0.4), (2.0, 0.02), (0.3, 0.25),
                        (1.5, 0.05)])
        plan = adaptive_blocks(qber, RATE_HZ, DT_S, rng=2)
        assert np.all(plan.ends > plan.starts)
        assert np.all(plan.starts[1:] >= plan.ends[:-1])
        assert plan.starts[0] >= 0 and plan.ends[-1] <= qber.size

    def test_an_unusable_run_yields_no_blocks_at_all(self):
        plan = adaptive_blocks(_series([(5.0, 0.35)]), RATE_HZ, DT_S, rng=0)
        assert plan.count == 0
        assert plan.total_key_bits == 0.0
        assert plan.duty == 0.0

    def test_an_empty_run_is_an_empty_plan(self):
        plan = adaptive_blocks(np.zeros(0), RATE_HZ, DT_S, rng=0)
        assert plan.count == 0
        assert plan.mean_key_rate_hz == 0.0

    def test_it_refuses_a_non_positive_budget(self):
        for kwargs in ({"max_block_s": 0.0}, {"granularity_s": 0.0}):
            with pytest.raises(ValueError):
                adaptive_blocks(_series([(1.0, 0.01)]), RATE_HZ, DT_S, **kwargs)


class TestPoolingController:
    """The controller allowed to skip a segment without closing the block."""

    def test_it_never_beats_its_own_oracle(self):
        rng = np.random.default_rng(5)
        for _ in range(6):
            pattern = [(float(rng.uniform(0.5, 3.0)),
                        float(rng.choice([0.005, 0.02, 0.05, 0.12, 0.3])))
                       for _ in range(8)]
            qber = _series(pattern)
            controller = adaptive_blocks(qber, RATE_HZ, DT_S, rng=1, pooling=True,
                                         max_block_s=float(qber.size * DT_S))
            oracle = pooled_optimal_blocks(qber, RATE_HZ, DT_S)
            assert controller.total_key_bits <= oracle.total_key_bits + 1e-6

    def test_it_keeps_one_block_across_a_bad_window(self):
        """What the contiguous controller has to close for, and this does not.

        Two good stretches separated by an unusable one: the contiguous
        controller must bank the first and start over, paying the finite-size
        cost twice. This one drops the bad window and settles up once.
        """
        qber = _series([(3.0, 0.01), (0.4, 0.30), (3.0, 0.01)])
        pooled = adaptive_blocks(qber, RATE_HZ, DT_S, rng=0, pooling=True,
                                 max_block_s=7.0)
        contiguous = adaptive_blocks(qber, RATE_HZ, DT_S, rng=0, max_block_s=7.0)
        assert pooled.count < contiguous.count
        assert pooled.pieces >= pooled.count
        assert pooled.total_key_bits > contiguous.total_key_bits
        assert pooled.duty < 1.0

    def test_it_leaves_a_quiet_link_in_one_block(self):
        plan = adaptive_blocks(_series([(20.0, 0.01)]), RATE_HZ, DT_S, rng=3,
                               pooling=True, max_block_s=20.0)
        assert plan.count <= 2
        assert plan.duty > 0.95

    def test_pooling_off_is_the_contiguous_controller(self):
        """The default must not have moved: the two modes are compared, and a
        refinement that also changed the baseline would not be measurable."""
        qber = _series([(2.0, 0.01), (0.5, 0.25), (3.0, 0.02)])
        plan = adaptive_blocks(qber, RATE_HZ, DT_S, rng=4)
        assert plan.pieces == plan.count
        assert np.array_equal(np.sort(plan.block_of_piece),
                              np.arange(plan.count))

    def test_its_pieces_are_ordered_and_disjoint(self):
        qber = _series([(1.0, 0.01), (0.3, 0.4), (2.0, 0.02), (0.2, 0.25),
                        (1.5, 0.05)])
        plan = adaptive_blocks(qber, RATE_HZ, DT_S, rng=2, pooling=True)
        assert np.all(plan.ends > plan.starts)
        assert np.all(plan.starts[1:] >= plan.ends[:-1])
        assert plan.block_of_piece.max() < plan.count

    def test_a_short_forecast_window_is_accepted_and_changes_nothing_stationary(self):
        """On a link that never moves, how far back it looks cannot matter."""
        qber = _series([(20.0, 0.01)])
        whole = adaptive_blocks(qber, RATE_HZ, DT_S, rng=6, pooling=True,
                                max_block_s=20.0)
        windowed = adaptive_blocks(qber, RATE_HZ, DT_S, rng=6, pooling=True,
                                   max_block_s=20.0, forecast_window_s=2.0)
        assert windowed.total_key_bits == pytest.approx(whole.total_key_bits,
                                                        rel=0.05)


class TestPlanAccounting:

    def test_the_rate_is_divided_by_the_run_not_by_the_blocks(self):
        """Otherwise a strategy could reach any rate it liked by keeping one
        good millisecond and calling the rest of the run somebody else's
        problem."""
        plan = BlockPlan(np.array([0]), np.array([100]), np.array([1000.0]),
                         dt_s=DT_S, run_samples=10_000, strategy="teste")
        assert plan.duration_s == pytest.approx(10.0)
        assert plan.mean_key_rate_hz == pytest.approx(100.0)
        assert plan.duty == pytest.approx(0.01)

    def test_the_empty_fraction_counts_blocks_that_produced_nothing(self):
        plan = BlockPlan(np.array([0, 100]), np.array([100, 200]),
                         np.array([500.0, 0.0]), dt_s=DT_S, run_samples=200)
        assert plan.empty_fraction == pytest.approx(0.5)
        assert plan.count == 2
        assert plan.total_key_bits == pytest.approx(500.0)

    def test_a_plan_with_no_blocks_has_no_rate(self):
        plan = BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s=DT_S, run_samples=1000)
        assert plan.mean_key_rate_hz == 0.0
        assert plan.empty_fraction == 0.0
        assert plan.duty == 0.0
