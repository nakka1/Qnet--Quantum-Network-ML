"""Turn a fitted compensator into a fixed-point one, and say what it costs.

The build is a calibration followed by a rewrite. Calibration runs the
floating-point model over *training* trajectories and records the range every
tensor actually occupies — the scaled features reach 86, the output layer's
pre-activations reach 1052, and one global format for both would waste range at
one end and resolution at the other. Each tensor then gets the format with the
most fraction bits that still holds its range at the chosen percentile.

Calibrating on the test seeds would be the same mistake as fitting on them: the
format would be chosen knowing the excursions it has to survive. The seeds are
the training ones, and the sweep is scored on the held-out ones.

The rewrite keeps the interface. :class:`QuantisedCompensator` has the
``reset``/``predict`` pair that :func:`~qnet_autopilot.compensation.closed_loop.run_closed_loop`
expects, so the quantised model is scored by exactly the same closed loop, on
exactly the same trajectories, as the float one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..compensation.predictors import (FeatureSpec, Persistence, Standardiser,
                                       build_features)
from .fixed_point import (CordicConfig, FixedLayer, FixedNetwork,
                          FixedStandardiser, QFormat, choose_format,
                          fixed_exp_map, fixed_log_map)

__all__ = ["QuantisationConfig", "QuantisedCompensator", "build_quantised",
           "collect_calibration"]


@dataclass(frozen=True)
class QuantisationConfig:
    """What the port is allowed to spend.

    Attributes:
        width: datapath width for features, activations and weights.
        cordic: the transcendental blocks' configuration.
        percentile: share of each tensor's magnitude that must fit without
            saturating, at calibration time.
        quantise_transcendental: when false, the logarithms and the exponential
            stay in double precision and only the linear path is quantised.
            Not a deployment option — an attribution one, so the sweep can say
            which stage an error came from.
        name: label for reports.
    """

    width: int = 16
    cordic: CordicConfig = field(default_factory=CordicConfig)
    percentile: float = 100.0
    quantise_transcendental: bool = True
    name: str = ""

    def labelled(self) -> str:
        if self.name:
            return self.name
        transcendental = (f"CORDIC {self.cordic.iterations}"
                          if self.quantise_transcendental else "float")
        return f"{self.width} bits, {transcendental}"


class _Recorder:
    """A compensator that answers exactly as the real one and keeps the trace.

    Calibration has to see the feature vectors the model will actually be given,
    and those are not the ones a pass over the trajectory produces. In closed
    loop the history is made of *monitoring estimates* — a finite binomial
    sample of photons, not the true rotation — and the extra noise makes the
    logarithms larger. Calibrating against the truth undersizes the format: it
    chose a range of 5.2 where the loop then produced excursions past 8, and
    every one of the twenty thousand saturations that followed was in the
    standardiser's output.

    So the recorder wraps the fitted compensator, delegates every call to it so
    the loop behaves identically, and banks what went through.
    """

    def __init__(self, compensator):
        self.inner = compensator
        self.name = compensator.name
        self.features: list[NDArray[np.float64]] = []

    def reset(self) -> None:
        self.inner.reset()

    def predict(self, time_s, measured, covariates, horizon_s):
        history = self.inner._history
        if len(history) >= self.inner.spec.window - 1:
            candidate = list(history)[1:] + [np.asarray(measured, dtype=float)] \
                if len(history) == self.inner.spec.window else \
                list(history) + [np.asarray(measured, dtype=float)]
            if len(candidate) == self.inner.spec.window:
                self.features.append(
                    build_features(candidate, covariates, self.inner.spec))
        return self.inner.predict(time_s, measured, covariates, horizon_s)


def collect_calibration(compensator, trajectories, control, rng: int = 7000,
                        stride: int = 1) -> dict:
    """Ranges every tensor occupies, on traces from the closed loop itself.

    Args:
        compensator: the fitted floating-point compensator.
        trajectories: *training* trajectories. Calibrating on the test seeds
            would size the format knowing the excursions it has to survive,
            which is the same mistake as fitting on them.
        control: the control configuration.
        rng: seed offset for the monitoring noise, matching the evaluation.
        stride: keep one recorded window in this many.

    Returns:
        A dictionary of arrays: raw features, scaled features, the
        pre-activation of each layer, and the layer parameters themselves.
    """
    from ..compensation.closed_loop import run_closed_loop

    raw = []
    for index, trajectory in enumerate(trajectories):
        recorder = _Recorder(compensator)
        run_closed_loop(trajectory, recorder, control, rng=rng + index)
        raw.extend(recorder.features[::stride])

    raw = np.asarray(raw, dtype=float)
    scaled = compensator.standardiser.transform(raw)
    layers = _layers_of(compensator)

    activations, pre_activations = scaled, []
    for weight, bias, relu in layers:
        pre = activations @ weight + bias
        pre_activations.append(pre)
        activations = np.maximum(pre, 0.0) if relu else pre

    return {"raw": raw, "scaled": scaled, "pre_activations": pre_activations,
            "layers": layers}


def _layers_of(compensator) -> list[tuple[NDArray, NDArray, bool]]:
    """``(weight, bias, relu)`` per layer, for either model this package fits.

    The ridge is a one-layer network with no rectifier, which is worth saying
    in code rather than special-casing everywhere downstream: it is 144
    multiply-accumulates against the MLP's 46 656, and it ties the MLP at the
    phase-3 operating point.
    """
    model = compensator.model
    network = getattr(model, "network", None)
    if network is not None:
        last = len(network.weights) - 1
        return [(np.asarray(weight, dtype=float), np.asarray(bias, dtype=float),
                 index != last)
                for index, (weight, bias) in enumerate(zip(network.weights,
                                                           network.biases))]
    coefficients = np.asarray(model.coef_, dtype=float)
    intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=float))
    return [(coefficients.T, intercept, False)]


def build_quantised(compensator, calibration: dict,
                    config: QuantisationConfig) -> "QuantisedCompensator":
    """Quantise a fitted compensator against a calibration pass."""
    width, percentile = config.width, config.percentile
    standardiser: Standardiser = compensator.standardiser

    mean = np.asarray(standardiser.mean, dtype=float)
    centred = calibration["raw"] - mean
    # One fraction per element, sized from that element's own excursion after
    # the offset comes off. The rotation features and the covariates differ by
    # three orders of magnitude and a shared format serves neither.
    per_element = np.array([choose_format(column, width, percentile).fraction
                            for column in centred.T], dtype=np.int64)
    reciprocal = 1.0 / np.asarray(standardiser.scale, dtype=float)
    reciprocal_format = choose_format(reciprocal, width, 100.0)
    scaled_format = choose_format(calibration["scaled"], width, percentile)

    fixed_standardiser = FixedStandardiser(
        mean=mean, fraction=per_element,
        reciprocal=reciprocal_format.quantise(reciprocal), width=width,
        reciprocal_format=reciprocal_format, output_format=scaled_format)

    layers, current = [], scaled_format
    for (weight, bias, relu), pre in zip(calibration["layers"],
                                         calibration["pre_activations"]):
        weight_format = choose_format(weight, width, 100.0)
        bias_format = choose_format(bias, width, 100.0)
        output_format = choose_format(pre, width, percentile)
        layers.append(FixedLayer(
            weight=weight_format.quantise(weight),
            bias=bias_format.quantise(bias),
            weight_format=weight_format, bias_format=bias_format,
            output_format=output_format, relu=relu,
            input_width=current.width))
        current = output_format

    return QuantisedCompensator(
        spec=compensator.spec,
        standardiser=fixed_standardiser,
        network=FixedNetwork(layers=layers, input_format=scaled_format),
        config=config,
        name=f"{compensator.name} · {config.labelled()}")


@dataclass
class QuantisedCompensator:
    """The compensator as it would run on the target, scored by the same loop.

    Attributes:
        spec: the feature layout, unchanged from the fitted model.
        standardiser: the fixed-point scaling.
        network: the fixed-point layers.
        config: what the port was allowed to spend.
        name: display name.
    """

    spec: FeatureSpec
    standardiser: FixedStandardiser
    network: FixedNetwork
    config: QuantisationConfig
    name: str = "quantizado"

    #: Rotation matrices arrive from monitoring in [-1, 1], so one integer bit
    #: and a sign are enough and everything else is resolution.
    matrix_format: QFormat = field(init=False)
    #: The rotation vectors the logarithm returns reach pi.
    vector_format: QFormat = field(init=False)

    def __post_init__(self) -> None:
        width = self.config.width
        self.matrix_format = QFormat(width, width - 2)
        self.vector_format = QFormat(width, width - 3)
        self.reset()

    def reset(self) -> None:
        self._history: deque = deque(maxlen=self.spec.window)
        self._fallback = Persistence()

    # -- the datapath -------------------------------------------------------

    def _relative(self, past: NDArray, now: NDArray) -> NDArray[np.float64]:
        """One lag's contribution: ``log(past @ now.T) / horizon_scale``."""
        product = np.asarray(past, dtype=float) @ np.asarray(now, dtype=float).T
        if not self.config.quantise_transcendental:
            from ..compensation.predictors import _log_map

            return _log_map(product) / self.spec.horizon_scale

        quantised = self.matrix_format.quantise(product)
        vector = fixed_log_map(quantised, self.matrix_format,
                               self.config.cordic, self.vector_format)
        return self.vector_format.dequantise(vector) / self.spec.horizon_scale

    def _features(self, covariates) -> NDArray[np.float64]:
        history = list(self._history)
        now = history[-1]
        parts = [self._relative(history[-1 - lag] if len(history) > lag
                                else history[0], now)
                 for lag in self.spec.offsets]
        if self.spec.tone_lags:
            raise NotImplementedError(
                "the tone feature is not on the deployment path; the reach "
                "study measured that it does not help, so porting it would be "
                "porting a measured regression")
        if self.spec.use_covariates:
            parts.append(np.asarray(covariates, dtype=float))
        return np.concatenate(parts)

    def predict(self, time_s, measured, covariates, horizon_s):
        measured = np.asarray(measured, dtype=float)
        self._history.append(measured)
        if len(self._history) < self.spec.window:
            return self._fallback.predict(time_s, measured, covariates, horizon_s)

        features = self._features(covariates)
        scaled = self.standardiser.forward(
            self.standardiser.quantise_input(features))
        raw_output = self.network.forward(scaled)
        increment = (self.network.output_format.dequantise(raw_output)
                     * self.spec.horizon_scale)

        if not self.config.quantise_transcendental:
            from ..compensation.predictors import _exp_map

            return _exp_map(increment) @ measured

        rotation = fixed_exp_map(self.vector_format.quantise(increment),
                                 self.vector_format, self.config.cordic,
                                 self.matrix_format)
        return self.matrix_format.dequantise(rotation) @ measured

    # -- what it would cost -------------------------------------------------

    def report(self) -> dict:
        """Resource-shaped numbers, for sizing rather than for claiming."""
        logarithms = len(self.spec.offsets)
        return {
            "name": self.name,
            "width": self.config.width,
            "cordic_iterations": (self.config.cordic.iterations
                                  if self.config.quantise_transcendental else 0),
            "multiply_accumulates": self.network.multiply_accumulates,
            "parameter_bits": self.network.parameter_bits,
            "parameter_kib": self.network.parameter_bits / 8192.0,
            "logarithms_per_step": logarithms,
            "cordic_passes_per_step": logarithms + 1,
            "saturations": (self.network.saturations
                            + self.standardiser.saturations),
        }
