"""Compensators: from "do nothing" to a learned predictor, behind one interface.

All of them answer the same question — where will the fibre's rotation be one
control latency from now — and all of them see the same thing: noisy rotation
estimates from the monitoring channel, plus the environmental sensors.

The representation is what makes this tractable.  Rotations are predicted as
**increments in the tangent space**: everything is expressed as
:math:`\\log(R_{\\text{then}} R_{\\text{now}}^{\\mathsf T})`, a three-vector, and
the prediction is :math:`\\exp(\\hat y)\\,R_{\\text{now}}`.  Two reasons that is
the right frame and not a convenience:

* **Rotation matrices do not average.**  The mean of two rotation matrices is
  not a rotation, so any model that regresses on matrix entries spends its
  capacity relearning the SO(3) constraint from data. Increments compose
  additively to first order, which is the regime a control loop lives in.
* **It makes the problem shift-invariant.**  What the fibre does next depends on
  how it is moving, not on where on the sphere it happens to be. Predicting in
  the current frame means a model trained on one part of the sphere works
  everywhere on it — without that, a learned predictor needs to see the whole
  sphere in training and still generalises badly.

The classical baselines are written to be good.  :class:`AlphaBetaFilter` in
particular is the standard tracker a control engineer would reach for, and on
the slow thermal component it is very hard to beat; the learned models have to
earn their margin somewhere else.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..physics.fiber import rotation_matrix, rotation_vector

__all__ = [
    "FeatureSpec", "Standardiser",
    "NoCompensation", "Persistence", "LinearExtrapolation", "AlphaBetaFilter",
    "LearnedCompensator", "build_features",
]

IDENTITY = np.eye(3)


def _exp_map(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation from a rotation vector — single matrix, written out for speed.

    The general :func:`~qnet_autopilot.physics.fiber.rotation_matrix` broadcasts
    over arbitrary leading axes, and that machinery costs more than the
    arithmetic when it runs tens of thousands of times inside a control loop on
    one 3-vector at a time. This is the same Rodrigues formula with the
    broadcasting removed; ``tests/test_compensation.py`` checks the two agree.
    """
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return IDENTITY.copy()
    axis = vector / angle
    cos, sin = np.cos(angle), np.sin(angle)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cos * IDENTITY + sin * cross + (1.0 - cos) * np.outer(axis, axis)


def _log_map(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """SO(3) logarithm for a single matrix, with the same fast-path reasoning.

    Falls back to the general implementation near ``pi``, where the simple
    formula loses its precision and the careful branch is worth its cost.
    """
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    cosine = (trace - 1.0) / 2.0
    if cosine >= 1.0 - 1e-12:
        return 0.5 * np.array([matrix[2, 1] - matrix[1, 2],
                               matrix[0, 2] - matrix[2, 0],
                               matrix[1, 0] - matrix[0, 1]])
    if cosine <= -1.0 + 1e-9:
        return np.asarray(rotation_vector(matrix), dtype=float)
    angle = np.arccos(cosine)
    scale = angle / (2.0 * np.sin(angle))
    return scale * np.array([matrix[2, 1] - matrix[1, 2],
                             matrix[0, 2] - matrix[2, 0],
                             matrix[1, 0] - matrix[0, 1]])


# ---------------------------------------------------------------------------
# feature construction, shared by training and inference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """What a learned compensator looks at.

    Attributes:
        window: how many past rotation estimates to use.  ``window=2`` gives
            one velocity estimate; more lets a model see the curvature, and the
            oscillation of the vibration band.
        use_covariates: include temperature, its rate, and the accelerometer.
        horizon_scale: the increments are divided by this before the model sees
            them, so the target and the inputs sit at a comparable scale.
            Set from the control period at fit time.
        lags: which past steps to look at, if not every one of the last
            ``window``.  A model that has to identify a tone before it can
            predict its phase needs a window spanning several periods of the
            slowest tone in its band, and taking every step of such a window
            costs features quadratically for information the neighbouring steps
            already carry.  A dilated set — 1, 2, 3, 4, 6, 8, 11, 16, … — spans
            eight times as long for a third of the features.  ``window`` still
            says how far back the spec reaches, and must cover the largest lag.
        tone_lags: if set, append a scalar estimate of the dominant tone
            frequency, read off the recursion a pure oscillation satisfies:
            ``x[k+1] + x[k-1] = 2 cos(ω Δt) x[k]``.  The feature is
            ``cos(ω Δt)`` itself, pooled over the three rotation components and
            fitted over this many recent steps.  It is causal — the same
            estimates the model already sees, nothing else — and it exists
            because the reach study found that one model across a wide band
            fails while one model per frequency works, which is what a missing
            conditioning variable looks like.
    """

    window: int = 8
    use_covariates: bool = True
    horizon_scale: float = 1.0
    lags: tuple[int, ...] | None = None
    tone_lags: int = 0

    @property
    def offsets(self) -> tuple[int, ...]:
        """The lags actually read, dilated set or every step."""
        if self.lags is not None:
            return self.lags
        return tuple(range(1, self.window))

    @property
    def size(self) -> int:
        return (3 * len(self.offsets) + (3 if self.use_covariates else 0)
                + (1 if self.tone_lags else 0))

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2 to see any motion")
        if self.lags is not None:
            if not self.lags or min(self.lags) < 1:
                raise ValueError("lags must be positive and non-empty")
            if max(self.lags) > self.window - 1:
                raise ValueError("window must reach the largest lag")
        if self.tone_lags and self.tone_lags > self.window - 1:
            raise ValueError("window must reach back over the tone estimate")


def build_features(history: Sequence[NDArray[np.float64]],
                   covariates: NDArray[np.float64],
                   spec: FeatureSpec) -> NDArray[np.float64]:
    """Turn a rotation history into a feature vector, in the current frame.

    Args:
        history: rotation estimates, oldest first, at least ``spec.window`` of
            them.  Only the last ``window`` are used.
        covariates: ``[temperature, dT/dt, vibration]``.
        spec: what to include.

    Returns:
        A vector of length ``spec.size``.

    Note:
        Element *i* is where the fibre was *i* control steps ago, relative to
        now.  The sign convention means a steady drift produces features that
        grow linearly with lag, so even a linear model can read off a velocity.
    """
    now = history[-1]

    def relative(lag: int) -> NDArray[np.float64]:
        past = history[-1 - lag] if len(history) > lag else history[0]
        return _log_map(past @ now.T) / spec.horizon_scale

    parts = [relative(lag) for lag in spec.offsets]
    if spec.tone_lags:
        parts.append(np.array([_tone_estimate(
            np.stack([relative(lag) for lag in range(spec.tone_lags, -1, -1)]))]))
    if spec.use_covariates:
        parts.append(np.asarray(covariates, dtype=float))
    return np.concatenate(parts)


def _tone_estimate(track: NDArray[np.float64]) -> float:
    """``cos(2 pi f dt)`` of the dominant oscillation in a short track.

    A pure tone sampled uniformly satisfies ``x[k+1] + x[k-1] = 2c x[k]`` with
    ``c = cos(omega dt)``, exactly and independently of amplitude and phase.
    Least squares on that recursion is therefore a frequency estimator that
    costs a handful of multiplications and needs no transform, no buffer beyond
    the window the model already has, and no knowledge of anything but the
    estimates.

    Args:
        track: ``(steps, 3)`` rotation estimates relative to now, oldest first.

    Returns:
        The estimate, clipped to ``[-1, 1]``. Zero when the track is too short
        or too still to say anything, which is the right answer rather than a
        number read out of numerical noise: a model given a confident-looking
        frequency for a link that is not oscillating would learn to trust it.

    Note:
        Returning the cosine rather than the frequency is deliberate. It is
        bounded, it is smooth through the Nyquist fold, and it is the quantity
        the recursion is linear in, so the estimator has no branch in it.

        The recursion is fitted to the track's *first difference*, not to the
        track. Both are pure tones at the same frequency when the input is, so
        the estimator stays exact — but a real track also carries thermal
        drift, whose displacement over a hundred-millisecond window dwarfs the
        oscillation and drags the least-squares fit towards ``cos = 1``.
        Measured on the calibrated aerial fibre, the undifferenced form reads
        +0.20 at 180 Hz where the truth is −0.64, and collapses 45 Hz and
        180 Hz into the same corner of its range; differencing recovers a
        monotone spread across the band, which is what a conditioning variable
        has to have.
    """
    if track.shape[0] < 4:
        return 0.0
    track = np.diff(track, axis=0)
    middle = track[1:-1]
    neighbours = track[2:] + track[:-2]
    energy = float(np.sum(middle * middle))
    if energy <= 0.0:
        return 0.0
    return float(np.clip(np.sum(neighbours * middle) / (2.0 * energy), -1.0, 1.0))


@dataclass
class Standardiser:
    """Zero-mean, unit-variance scaling, fitted on the training split.

    The features span four orders of magnitude — a rotation increment is
    ~10\\ :sup:`-4` rad while a temperature reading is ~20 — so without this the
    first layer of any model is dominated by whichever feature happens to carry
    the largest units.
    """

    mean: NDArray[np.float64] | None = None
    scale: NDArray[np.float64] | None = None

    def fit(self, features: NDArray[np.float64]) -> "Standardiser":
        self.mean = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), 1e-12)
        return self

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean is None:
            return features
        return (features - self.mean) / self.scale


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

class NoCompensation:
    """Leave the fibre alone.  The floor, and the reason compensation exists."""

    name = "sem compensação"

    def reset(self) -> None:
        pass

    def predict(self, time_s, measured, covariates, horizon_s):
        return IDENTITY.copy()


class Persistence:
    """Correct to the last measurement, and hope the fibre stayed put.

    Optimal at zero control latency and the natural thing to implement.  Its
    error is exactly the drift accumulated during the loop delay, which is what
    every other compensator here is trying to cancel.
    """

    name = "persistência"

    def reset(self) -> None:
        pass

    def predict(self, time_s, measured, covariates, horizon_s):
        return np.asarray(measured, dtype=float)


@dataclass
class LinearExtrapolation:
    """Constant angular velocity from the last two estimates.

    The simplest thing that answers the latency at all.  Its weakness is
    measurement noise: differencing two noisy estimates doubles the variance,
    so at low photon counts it chases noise and does worse than persistence —
    which is precisely why the alpha-beta filter exists.
    """

    name: str = "extrapolação linear"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: NDArray[np.float64] | None = None
        self._previous_time: float | None = None

    def predict(self, time_s, measured, covariates, horizon_s):
        measured = np.asarray(measured, dtype=float)
        if self._previous is None or self._previous_time is None:
            self._previous, self._previous_time = measured, time_s
            return measured

        interval = time_s - self._previous_time
        if interval <= 0:
            return measured
        velocity = _log_map(measured @ self._previous.T) / interval
        self._previous, self._previous_time = measured, time_s
        return _exp_map(velocity * horizon_s) @ measured


@dataclass
class AlphaBetaFilter:
    """Smoothed position and velocity — the standard tracker, and a strong bar.

    Two exponential moving averages: one on the rotation itself (which rejects
    measurement noise) and one on the angular velocity (which answers the
    latency).  ``alpha`` and ``beta`` trade tracking lag against noise
    rejection, and the defaults are tuned for the operating point in the report.

    This is the baseline that matters.  It is what a control engineer would
    deploy, it costs nothing to run, and on a purely diffusive drift it is close
    to optimal — a Kalman filter on a random walk reduces to something very like
    it.  A learned predictor has to beat it on the structure it cannot see: the
    narrowband vibration, and the coupling to temperature.

    Attributes:
        alpha: weight on the newest measurement for the position estimate.
        beta: weight on the newest velocity estimate.
    """

    name: str = "filtro alfa-beta"
    alpha: float = 0.35
    beta: float = 0.10

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: NDArray[np.float64] | None = None
        self._velocity = np.zeros(3)
        self._previous_time: float | None = None

    def predict(self, time_s, measured, covariates, horizon_s):
        measured = np.asarray(measured, dtype=float)
        if self._state is None or self._previous_time is None:
            self._state, self._previous_time = measured, time_s
            return measured

        interval = time_s - self._previous_time
        if interval <= 0:
            return self._state

        # Predict where the filter thought we would be, then correct towards
        # the measurement — the ordinary alpha-beta update, in the tangent space.
        predicted = _exp_map(self._velocity * interval) @ self._state
        innovation = _log_map(measured @ predicted.T)

        self._state = _exp_map(self.alpha * innovation) @ predicted
        self._velocity = self._velocity + (self.beta / interval) * innovation
        self._previous_time = time_s

        return _exp_map(self._velocity * horizon_s) @ self._state


# ---------------------------------------------------------------------------
# learned
# ---------------------------------------------------------------------------

@dataclass
class LearnedCompensator:
    """Wraps any fitted regressor that maps features to a rotation increment.

    The model predicts :math:`\\hat y`, the rotation vector from now to the
    horizon, and the compensator returns :math:`\\exp(\\hat y) R_{\\text{now}}`.
    Anything with a ``predict(X) -> (n, 3)`` method works — a scikit-learn
    estimator, or the numpy MLP in :mod:`qnet_autopilot.agents.numpy_nn`.

    Until the history fills up, it falls back to persistence rather than padding
    with zeros: a model fed a fabricated history produces a confident wrong
    answer, and the first few control steps of every run would be corrupted.

    Attributes:
        model: the fitted regressor.
        spec: the feature layout it was fitted with.
        standardiser: the scaling it was fitted with.
        name: display name.
    """

    model: object
    spec: FeatureSpec
    standardiser: Standardiser
    name: str = "aprendido"

    def reset(self) -> None:
        self._history: deque = deque(maxlen=self.spec.window)
        self._fallback = Persistence()

    def __post_init__(self) -> None:
        self.reset()

    def predict(self, time_s, measured, covariates, horizon_s):
        measured = np.asarray(measured, dtype=float)
        self._history.append(measured)
        if len(self._history) < self.spec.window:
            return self._fallback.predict(time_s, measured, covariates, horizon_s)

        features = build_features(list(self._history), covariates, self.spec)
        scaled = self.standardiser.transform(features[None, :])
        increment = np.asarray(self.model.predict(scaled), dtype=float).reshape(3)
        return _exp_map(increment * self.spec.horizon_scale) @ measured
