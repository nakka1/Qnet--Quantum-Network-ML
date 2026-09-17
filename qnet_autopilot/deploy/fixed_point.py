"""The compensator's datapath in fixed point, with the semantics RTL will have.

Why this module exists in this form
-----------------------------------
The question that sends a model to an FPGA is usually assumed to be throughput.
For this compensator it is not: 46 656 multiply-accumulates at a 2 ms control
period is 23 MMAC/s, which a microcontroller does. Timing the real datapath
shows where the work actually is:

======================================  ==========  =======
block                                   time        share
======================================  ==========  =======
``build_features`` — 15 SO(3) logarithms  124.8 us   72%
network forward — 46 656 MAC               25.4 us   15%
``exp_map`` + 3x3 product                  19.1 us   11%
standardiser                                2.8 us    2%
======================================  ==========  =======

So the expensive part is transcendental, not arithmetic, and a port that
quantises only the weights answers the wrong question. This module quantises
the **whole** path — logarithms, network, exponential — so the study in
``scripts/evaluate_quantisation.py`` can attribute the error to each stage.

A tempting shortcut that does not work
--------------------------------------
Near the identity, :math:`\\log R \\approx \\frac{1}{2}(R - R^{T})^{\\vee}` and
the transcendentals vanish. Measured on the calibrated operating point, the
rotations the feature window spans are **not** near the identity: the angles
have median 0.52 rad and reach 2.22 rad, because the acoustic band moves the
state a long way in the 30 ms the window covers. The small-angle form is wrong
by up to 1.37 rad there. It is not used.

What is used instead
--------------------
The logarithm is reformulated to avoid ``arccos``:

.. math::

    v = \\tfrac{1}{2}(R - R^{T})^{\\vee}, \\quad
    s = \\lVert v \\rVert = \\sin\\theta, \\quad
    c = \\tfrac{1}{2}(\\mathrm{tr}\\,R - 1) = \\cos\\theta, \\quad
    \\log R = v\\,\\frac{\\operatorname{atan2}(s, c)}{s}.

This agrees with the floating-point implementation to $1.5\\times10^{-15}$ over
the observed range, and every operation in it is one a CORDIC does natively:
``atan2`` and a magnitude come out of the same vectoring pass, and the
exponential's sine and cosine come out of a rotation pass. That is the reason
for the reformulation — not numerical accuracy, which was already fine.

Fidelity to the hardware
------------------------
There is no HDL simulator in the container this was developed in, so the
faithfulness of the Verilog in :mod:`qnet_autopilot.deploy.verilog` cannot be
established by co-simulation. It is established by construction instead: this
module is written to have exactly the semantics the emitted RTL has —
products at full width, accumulation in a wide register with no intermediate
rounding, one round-half-up and one saturation at the end of each layer, and
CORDIC with a fixed iteration count and the same angle table. The testbench
that ships with the RTL checks it against vectors produced here, so this
module is the golden reference and any disagreement is a bug in the RTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "QFormat", "choose_format", "FixedLayer", "FixedNetwork",
    "FixedStandardiser", "CordicConfig", "cordic_atan2", "cordic_sincos",
    "fixed_log_map", "fixed_exp_map", "QuantisedCompensator",
]


# ---------------------------------------------------------------------------
# the number format
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QFormat:
    """A signed two's-complement fixed-point format.

    Attributes:
        width: total bits, sign included.
        fraction: bits after the binary point. Integer bits are
            ``width - fraction - 1``.

    Note:
        Rounding is round-half-up, which is what an adder with a carry-in at
        the rounding bit does and therefore what costs nothing in hardware.
        Round-to-nearest-even would cost a comparator chain and is not worth it
        for a control loop.
    """

    width: int
    fraction: int

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError("width must leave room for a sign bit")
        if not -self.width < self.fraction <= self.width - 1:
            raise ValueError("fraction must leave at least the sign bit")

    @property
    def scale(self) -> float:
        return float(2.0 ** self.fraction)

    @property
    def largest(self) -> int:
        return (1 << (self.width - 1)) - 1

    @property
    def smallest(self) -> int:
        return -(1 << (self.width - 1))

    @property
    def resolution(self) -> float:
        return 1.0 / self.scale

    def saturate(self, values: ArrayLike) -> NDArray[np.int64]:
        return np.clip(np.asarray(values, dtype=np.int64),
                       self.smallest, self.largest)

    def quantise(self, values: ArrayLike) -> NDArray[np.int64]:
        """Float to integer, rounded half-up and saturated."""
        scaled = np.asarray(values, dtype=float) * self.scale
        return self.saturate(np.floor(scaled + 0.5))

    def dequantise(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64) / self.scale

    def __str__(self) -> str:
        return f"Q{self.width - self.fraction - 1}.{self.fraction}"


def choose_format(values: ArrayLike, width: int,
                  percentile: float = 100.0) -> QFormat:
    """Pick the fraction length that fits ``values`` in ``width`` bits.

    Args:
        values: the tensor the format has to hold.
        width: total bits available.
        percentile: share of the magnitude distribution that must fit without
            saturating. 100 fits everything; a lower value buys resolution at
            the cost of clipping the tail, which is the trade a deployment
            makes deliberately.

    Returns:
        The format with the most fraction bits that still holds the chosen
        percentile of the data.

    Note:
        Per tensor, not per network. The measured ranges differ by three orders
        of magnitude between the scaled features and the output layer's
        pre-activations, and one global format for all of them wastes either
        range or resolution everywhere.
    """
    magnitude = np.max(np.abs(np.asarray(values, dtype=float))) if percentile >= 100.0 \
        else float(np.percentile(np.abs(np.asarray(values, dtype=float)), percentile))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        return QFormat(width, width - 1)
    # Integer bits needed for the magnitude, at least zero, plus the sign bit.
    integer_bits = max(int(np.ceil(np.log2(magnitude + 1e-30))), 0)
    return QFormat(width, min(width - 1 - integer_bits, width - 1))


# ---------------------------------------------------------------------------
# the linear part
# ---------------------------------------------------------------------------

@dataclass
class FixedLayer:
    """One affine layer, accumulated the way a DSP chain accumulates.

    The product of an input in ``Q a.b`` and a weight in ``Q c.d`` lands in
    ``Q (a+c+1).(b+d)``; the accumulator keeps every bit of it and of the sum,
    with no rounding until the layer is finished. Only then is the result
    shifted to the output format, rounded once and saturated once. Rounding per
    product instead, which is what a naive float-to-fixed conversion does,
    changes the answer and is not what the hardware will do.

    Attributes:
        weight, bias: quantised parameters.
        weight_format, bias_format, output_format: their formats.
        relu: apply the rectifier before the output format is applied.
        saturations: how many output elements hit a rail, counted across every
            call. A port that saturates is a port whose format is wrong, and
            silence about it is how that ships.
    """

    weight: NDArray[np.int64]
    bias: NDArray[np.int64]
    weight_format: QFormat
    bias_format: QFormat
    output_format: QFormat
    relu: bool = False
    saturations: int = 0
    calls: int = 0

    @property
    def accumulator_bits(self) -> int:
        """Width the RTL accumulator must have for the sum never to wrap."""
        inputs = int(self.weight.shape[0])
        product = self.weight_format.width + self.input_width
        return int(product + np.ceil(np.log2(max(inputs, 1) + 1)))

    input_width: int = 16

    def forward(self, values: NDArray[np.int64],
                input_format: QFormat) -> NDArray[np.int64]:
        """Integer matrix-vector product, then one round and one saturation.

        The accumulation is done in ``int64`` whenever the layer's declared
        accumulator width fits in it, and in Python integers otherwise. Both
        paths produce the same numbers — the choice is only about speed, and it
        matters because the sweep runs this inside a closed loop ten thousand
        times per trajectory. The ``int64`` path is guarded by the same width
        calculation the RTL uses to size its accumulator register, so a
        configuration that would overflow silently in ``int64`` takes the exact
        path instead of wrapping.
        """
        exact = self.accumulator_bits > 62
        dtype = object if exact else np.int64
        accumulator = np.asarray(values, dtype=dtype) @ self.weight.astype(dtype)
        product_fraction = input_format.fraction + self.weight_format.fraction
        bias_shift = product_fraction - self.bias_format.fraction
        if bias_shift >= 0:
            accumulator = accumulator + self.bias.astype(object) * (1 << bias_shift)
        else:
            accumulator = accumulator + self.bias.astype(object) >> (-bias_shift)

        if self.relu:
            accumulator = np.where(accumulator > 0, accumulator, 0)

        shift = product_fraction - self.output_format.fraction
        if shift > 0:
            rounded = (accumulator + (1 << (shift - 1))) >> shift
        else:
            rounded = accumulator * (1 << -shift)

        rounded = np.asarray(rounded.tolist() if exact else rounded,
                             dtype=np.int64)
        clipped = self.output_format.saturate(rounded)
        self.saturations += int(np.sum(clipped != rounded))
        self.calls += 1
        return clipped


@dataclass
class FixedNetwork:
    """A stack of :class:`FixedLayer`, with the input format it expects."""

    layers: list[FixedLayer]
    input_format: QFormat

    @property
    def saturations(self) -> int:
        return sum(layer.saturations for layer in self.layers)

    @property
    def multiply_accumulates(self) -> int:
        return int(sum(layer.weight.size for layer in self.layers))

    @property
    def parameter_bits(self) -> int:
        return int(sum(layer.weight.size * layer.weight_format.width
                       + layer.bias.size * layer.bias_format.width
                       for layer in self.layers))

    def forward(self, values: NDArray[np.int64]) -> NDArray[np.int64]:
        current_format = self.input_format
        result = np.asarray(values, dtype=np.int64)
        for layer in self.layers:
            result = layer.forward(result, current_format)
            current_format = layer.output_format
        return result

    @property
    def output_format(self) -> QFormat:
        return self.layers[-1].output_format


@dataclass
class FixedStandardiser:
    """``(x - mean) * reciprocal``, with the offset removed before quantisation.

    Two things here are not the obvious implementation, and both were forced by
    measurement.

    **The offset comes off first.** Quantising the raw feature and subtracting a
    quantised mean afterwards cancels catastrophically: the temperature
    covariate has a mean near 20 and a standard deviation near 0.003, so a
    format wide enough to hold 20 leaves a resolution coarser than the signal,
    and dividing that by the scale turns a feature that should span +/-4 into
    one that reaches twenty thousand. In closed loop that saturated the output
    on every single control step. Subtracting the constant offset in the front
    end, before the word is formed, is what every ADC chain does anyway, and it
    makes the difference exact.

    **The format is per element.** After offset removal the 45 rotation features
    still span a different range from the three covariates, and one shared
    format wastes range at one end and resolution at the other. A per-element
    fraction is a constant shift per channel, which is wiring rather than
    logic.

    Attributes:
        mean: the constant offset, in float, removed ahead of quantisation.
        fraction: per-element fraction bits for the centred value.
        reciprocal: quantised ``1 / scale``.
        width: datapath width.
        reciprocal_format, output_format: the remaining formats.
        saturations: elements that hit a rail, counted across every call.
    """

    mean: NDArray[np.float64]
    fraction: NDArray[np.int64]
    reciprocal: NDArray[np.int64]
    width: int
    reciprocal_format: QFormat
    output_format: QFormat
    saturations: int = 0

    @property
    def largest(self) -> int:
        return (1 << (self.width - 1)) - 1

    def quantise_input(self, features: ArrayLike) -> NDArray[np.int64]:
        """Offset removal and per-element quantisation, as the front end does."""
        centred = np.asarray(features, dtype=float) - self.mean
        scaled = np.floor(centred * (2.0 ** self.fraction) + 0.5)
        return np.clip(scaled, -self.largest - 1, self.largest).astype(np.int64)

    def forward(self, centred: NDArray[np.int64]) -> NDArray[np.int64]:
        product = np.asarray(centred, dtype=np.int64) * self.reciprocal
        shift = (self.fraction + self.reciprocal_format.fraction
                 - self.output_format.fraction)
        rounded = np.where(
            shift > 0,
            (product + (1 << np.maximum(shift - 1, 0))) >> np.maximum(shift, 0),
            product << np.maximum(-shift, 0))
        clipped = self.output_format.saturate(rounded)
        self.saturations += int(np.sum(clipped != rounded))
        return clipped


# ---------------------------------------------------------------------------
# the transcendental part
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CordicConfig:
    """How many CORDIC stages, and at what width.

    Attributes:
        iterations: rotation stages. Each one buys about one bit of angle
            resolution, so this is the knob the sweep turns.
        fraction: fraction bits the CORDIC datapath carries.
        width: total datapath width.
    """

    iterations: int = 16
    fraction: int = 20
    width: int = 32

    @property
    def format(self) -> QFormat:
        return QFormat(self.width, self.fraction)

    @property
    def angles(self) -> NDArray[np.int64]:
        """``atan(2**-i)`` in the datapath format — the RTL's constant table."""
        table = np.arctan(2.0 ** -np.arange(self.iterations))
        return np.floor(table * (1 << self.fraction) + 0.5).astype(np.int64)

    @property
    def gain(self) -> float:
        """The scale a CORDIC rotation introduces, which rotation mode undoes."""
        return float(np.prod(np.sqrt(1.0 + 4.0 ** -np.arange(self.iterations))))


def cordic_atan2(y: int, x: int, config: CordicConfig) -> tuple[int, int]:
    """Vectoring CORDIC: rotate ``(x, y)`` onto the axis, banking the angle.

    Args:
        y, x: the vector, as integers in the CORDIC format.
        config: stage count and widths.

    Returns:
        ``(angle, magnitude)`` with the angle in the CORDIC format and the
        magnitude still carrying the CORDIC gain, exactly as the RTL leaves it.
        The caller divides the gain out, or folds it into a later constant,
        which is what costs nothing.

    Note:
        CORDIC converges only within about 1.74 rad of the axis, so the first
        and second quadrants are folded in before the loop and the angle is
        corrected afterwards. That fold is two multiplexers in hardware and is
        the reason the routine handles the full circle.
    """
    angles = config.angles
    half_pi = int(np.floor(np.pi / 2 * (1 << config.fraction) + 0.5))
    correction = 0
    if x < 0:
        # Fold by 90 degrees rather than 180: it keeps the sign handling to one
        # branch and the rotation exact.
        x, y, correction = y, -x, half_pi
        if x < 0:
            x, y, correction = y, -x, 2 * half_pi

    angle = 0
    for index in range(config.iterations):
        if y > 0:
            x, y = x + (y >> index), y - (x >> index)
            angle += int(angles[index])
        else:
            x, y = x - (y >> index), y + (x >> index)
            angle -= int(angles[index])
    total = angle + correction
    if total > 2 * half_pi:
        total -= 4 * half_pi
    return total, x


def cordic_sincos(angle: int, config: CordicConfig) -> tuple[int, int]:
    """Rotation CORDIC: ``(cos, sin)`` of an angle, gain removed.

    Args:
        angle: in the CORDIC format, any value in ``[-pi, pi]``.
        config: stage count and widths.

    Returns:
        ``(cos, sin)`` in the CORDIC format.
    """
    angles = config.angles
    half_pi = int(np.floor(np.pi / 2 * (1 << config.fraction) + 0.5))
    flip = False
    if angle > half_pi:
        angle -= 2 * half_pi
        flip = True
    elif angle < -half_pi:
        angle += 2 * half_pi
        flip = True

    one = int(np.floor((1 << config.fraction) / config.gain + 0.5))
    x, y, residual = one, 0, angle
    for index in range(config.iterations):
        if residual >= 0:
            x, y = x - (y >> index), y + (x >> index)
            residual -= int(angles[index])
        else:
            x, y = x + (y >> index), y - (x >> index)
            residual += int(angles[index])
    return (-x, -y) if flip else (x, y)


def _norm_fixed(components) -> int:
    """Euclidean norm of fixed-point components, in their own format.

    The sum of squares of values in ``Q i.f`` lands in ``Q 2i.2f``, and the
    integer square root of that lands back in ``Q i.f`` with no shift at all.
    Writing it the obvious way instead — rescale the sum to ``Q i.f`` first,
    then take the root — throws away half the bits before the root sees them,
    and for the small vectors the exponential is fed it throws away almost all
    of them: it left a 9e-4 error floor that did not move when the datapath was
    widened from 12 to 24 bits, which is the signature of precision lost
    upstream of the format being swept.

    Integer square root is a shift-and-subtract loop in hardware; Python's
    exact :func:`math.isqrt` gives the value that loop converges to, so this
    stays bit-exact without simulating the loop.
    """
    from math import isqrt

    total = sum(int(component) ** 2 for component in components)
    return isqrt(max(total, 0))


def fixed_log_map(matrix: NDArray[np.int64], matrix_format: QFormat,
                  config: CordicConfig, output_format: QFormat
                  ) -> NDArray[np.int64]:
    """SO(3) logarithm through ``atan2``, in fixed point.

    Args:
        matrix: 3x3 rotation, quantised in ``matrix_format``.
        matrix_format: its format; entries live in ``[-1, 1]``.
        config: the CORDIC.
        output_format: format of the returned rotation vector.

    Returns:
        The rotation vector, quantised.
    """
    fraction = config.fraction
    shift = fraction - matrix_format.fraction

    def lift(value: int) -> int:
        return int(value) << shift if shift >= 0 else int(value) >> -shift

    axis = np.array([lift(matrix[2, 1] - matrix[1, 2]),
                     lift(matrix[0, 2] - matrix[2, 0]),
                     lift(matrix[1, 0] - matrix[0, 1])], dtype=object) // 2
    trace = lift(int(matrix[0, 0] + matrix[1, 1] + matrix[2, 2]))
    cosine = (trace - (1 << fraction)) // 2

    sine = _norm_fixed(axis)
    if sine == 0:
        return np.zeros(3, dtype=np.int64)

    angle, _ = cordic_atan2(sine, int(cosine), config)
    gain = (angle << fraction) // sine
    scaled = [(int(component) * gain) >> fraction for component in axis]

    back = output_format.fraction - fraction
    if back >= 0:
        values = [value << back for value in scaled]
    else:
        values = [(value + (1 << (-back - 1))) >> -back for value in scaled]
    return output_format.saturate(values)


def fixed_exp_map(vector: NDArray[np.int64], vector_format: QFormat,
                  config: CordicConfig, output_format: QFormat
                  ) -> NDArray[np.int64]:
    """Rodrigues, in fixed point: a rotation vector to a 3x3 matrix."""
    fraction = config.fraction
    shift = fraction - vector_format.fraction
    lifted = [int(value) << shift if shift >= 0 else int(value) >> -shift
              for value in vector]

    angle = _norm_fixed(lifted)
    one = 1 << fraction
    if angle == 0:
        return output_format.quantise(np.eye(3))

    axis = [(value << fraction) // angle for value in lifted]
    cosine, sine = cordic_sincos(angle, config)
    complement = one - cosine
    x, y, z = axis

    def product(first: int, second: int) -> int:
        return (first * second) >> fraction

    outer = [[product(a, b) for b in axis] for a in axis]
    cross = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    result = [[0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            identity = one if row == column else 0
            result[row][column] = (product(cosine, identity)
                                   + product(sine, cross[row][column])
                                   + product(complement, outer[row][column]))

    back = output_format.fraction - fraction
    if back >= 0:
        values = [[value << back for value in row] for row in result]
    else:
        values = [[(value + (1 << (-back - 1))) >> -back for value in row]
                  for row in result]
    return output_format.saturate(values)
