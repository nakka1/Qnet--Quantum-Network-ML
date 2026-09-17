"""The fixed-point port, and the RTL it emits.

There is no HDL simulator in this container, so the emitted Verilog cannot be
co-simulated here. That is not a reason to ship it unchecked. Two things stand
in for a simulator:

* the fixed-point reference is tested against the floating-point datapath it
  replaces, block by block, so the numbers the bench compares against are
  themselves checked;
* the sequential engine's **state machine** is re-implemented here cycle by
  cycle, following the emitted ``always`` block statement for statement, and
  required to produce exactly what the reference layer produces. That is what
  caught the bug it is written to catch: the first version formed its result
  from the accumulator on the cycle that consumed the last input, dropping the
  last term of every dot product. The arithmetic looked right, the shapes were
  right, and one part in forty-eight went missing.

The cycle model is not a simulator and this file does not claim it is. It
exercises the addressing, the accumulator's reset timing and the order results
come out in — which is where a generated engine actually goes wrong — and says
nothing about synthesis, timing closure or the Verilog's syntax.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.compensation.predictors import _exp_map, _log_map
from qnet_autopilot.deploy import (CordicConfig, FixedLayer, QFormat,
                                   choose_format, cordic_atan2, cordic_sincos,
                                   fixed_exp_map, fixed_log_map)
from qnet_autopilot.physics.fiber import rotation_matrix

CORDIC = CordicConfig(iterations=16, fraction=20, width=32)


class TestQFormat:

    def test_it_round_trips_within_its_own_resolution(self):
        fmt = QFormat(16, 12)
        values = np.linspace(-7.0, 7.0, 401)
        recovered = fmt.dequantise(fmt.quantise(values))
        assert np.max(np.abs(recovered - values)) <= fmt.resolution / 2 + 1e-12

    def test_it_saturates_rather_than_wrapping(self):
        """Wrapping turns a large positive into a large negative, which in a
        control loop is a command in the opposite direction."""
        fmt = QFormat(8, 4)
        assert fmt.quantise([100.0])[0] == fmt.largest
        assert fmt.quantise([-100.0])[0] == fmt.smallest

    def test_it_rounds_half_up_because_that_is_what_the_adder_does(self):
        fmt = QFormat(16, 0)
        assert list(fmt.quantise([0.5, 1.5, -0.5, -1.5])) == [1, 2, 0, -1]

    def test_it_refuses_a_format_with_no_room_for_a_sign(self):
        with pytest.raises(ValueError):
            QFormat(1, 0)
        with pytest.raises(ValueError):
            QFormat(8, 8)

    def test_choosing_a_format_keeps_the_range_it_was_given(self):
        values = np.array([-3.4, 0.2, 7.9])
        fmt = choose_format(values, 16)
        assert fmt.dequantise(fmt.quantise(values)) == pytest.approx(values,
                                                                     abs=1e-3)
        assert np.all(np.abs(fmt.quantise(values)) < fmt.largest)


class TestCordic:

    def test_vectoring_recovers_the_angle_over_the_whole_circle(self):
        rng = np.random.default_rng(0)
        scale = 1 << CORDIC.fraction
        worst = 0.0
        for _ in range(500):
            angle = float(rng.uniform(-np.pi + 1e-3, np.pi - 1e-3))
            magnitude = float(rng.uniform(0.05, 1.0))
            got, _ = cordic_atan2(int(magnitude * np.sin(angle) * scale),
                                  int(magnitude * np.cos(angle) * scale), CORDIC)
            worst = max(worst, abs(got / scale - angle))
        assert worst < 1e-3

    def test_rotation_recovers_sine_and_cosine(self):
        rng = np.random.default_rng(1)
        scale = 1 << CORDIC.fraction
        worst = 0.0
        for _ in range(500):
            angle = float(rng.uniform(-np.pi + 1e-3, np.pi - 1e-3))
            cosine, sine = cordic_sincos(int(angle * scale), CORDIC)
            worst = max(worst, abs(cosine / scale - np.cos(angle)),
                        abs(sine / scale - np.sin(angle)))
        assert worst < 1e-3

    def test_more_stages_buy_accuracy(self):
        """Each stage is about a bit, and that is the trade the sweep turns."""
        scale_of = lambda config: 1 << config.fraction
        angle = 1.1
        errors = []
        for iterations in (6, 10, 16):
            config = CordicConfig(iterations, 20, 32)
            got, _ = cordic_atan2(int(np.sin(angle) * scale_of(config)),
                                  int(np.cos(angle) * scale_of(config)), config)
            errors.append(abs(got / scale_of(config) - angle))
        assert errors[0] > errors[1] > errors[2]


class TestTranscendentalBlocks:

    def test_the_logarithm_matches_the_float_one(self):
        """Over the range the feature window actually spans, which reaches
        2.2 rad — not the small angles a first guess assumes."""
        rng = np.random.default_rng(2)
        matrix_format, vector_format = QFormat(20, 18), QFormat(20, 17)
        worst = 0.0
        for _ in range(200):
            matrix = np.asarray(rotation_matrix(rng.normal(size=3),
                                                rng.uniform(0.0, 2.2)),
                                dtype=float)
            got = vector_format.dequantise(
                fixed_log_map(matrix_format.quantise(matrix), matrix_format,
                              CORDIC, vector_format))
            worst = max(worst, float(np.linalg.norm(got - _log_map(matrix))))
        assert worst < 1e-3

    def test_the_exponential_matches_the_float_one(self):
        rng = np.random.default_rng(3)
        matrix_format, vector_format = QFormat(20, 18), QFormat(20, 17)
        worst = 0.0
        for _ in range(200):
            vector = rng.normal(size=3) * 1e-3
            got = matrix_format.dequantise(
                fixed_exp_map(vector_format.quantise(vector), vector_format,
                              CORDIC, matrix_format))
            worst = max(worst, float(np.abs(got - _exp_map(vector)).max()))
        assert worst < 1e-3

    def test_the_norm_keeps_the_bits_the_small_vectors_need(self):
        """The regression this guards: rescaling the sum of squares down to the
        input format before taking the root threw away half the bits, and left
        an error floor that did not move when the datapath was widened."""
        vector_format = QFormat(24, 21)
        matrix_format = QFormat(24, 22)
        errors = []
        for width in (16, 20, 24):
            fmt_v, fmt_m = QFormat(width, width - 3), QFormat(width, width - 2)
            vector = np.array([3e-4, -1e-4, 2e-4])
            got = fmt_m.dequantise(fixed_exp_map(fmt_v.quantise(vector), fmt_v,
                                                 CORDIC, fmt_m))
            errors.append(float(np.abs(got - _exp_map(vector)).max()))
        assert errors[0] > errors[-1] * 2, "widening the datapath must help"


class TestFixedLayer:

    @staticmethod
    def _layer(inputs: int, outputs: int, relu: bool, seed: int) -> FixedLayer:
        rng = np.random.default_rng(seed)
        weight_format = QFormat(16, 14)
        return FixedLayer(
            weight=weight_format.quantise(rng.normal(size=(inputs, outputs)) * 0.2),
            bias=weight_format.quantise(rng.normal(size=outputs) * 0.1),
            weight_format=weight_format, bias_format=weight_format,
            output_format=QFormat(16, 9), relu=relu, input_width=16)

    def test_the_exact_and_the_int64_path_agree(self):
        """The fast path is guarded by the accumulator width the RTL uses; if
        the two ever disagreed, that guard would be wrong."""
        layer = self._layer(48, 32, True, 4)
        values = QFormat(16, 12).quantise(
            np.random.default_rng(5).normal(size=48))
        fast = layer.forward(values, QFormat(16, 12))
        layer.__dict__["input_width"] = 40      # force the object-integer path
        slow = layer.forward(values, QFormat(16, 12))
        assert np.array_equal(fast, slow)

    def test_it_counts_the_rails_it_hits(self):
        layer = self._layer(48, 8, False, 6)
        layer.output_format = QFormat(16, 15)   # deliberately far too narrow
        layer.forward(QFormat(16, 12).quantise(
            np.random.default_rng(7).normal(size=48) * 5), QFormat(16, 12))
        assert layer.saturations > 0


class TestTheEmittedStateMachine:
    """A cycle model of ``qnet_mac``, following the emitted always block.

    Not a simulator. It reproduces the addressing, the accumulator reset and
    the order results appear in, because those are what a generated engine gets
    wrong, and requires the sequence to equal the reference layer's output
    exactly.
    """

    @staticmethod
    def _run_engine(layer: FixedLayer, values, input_format: QFormat):
        weight = layer.weight            # (inputs, outputs)
        inputs, outputs = weight.shape
        product_fraction = input_format.fraction + layer.weight_format.fraction
        shift = product_fraction - layer.output_format.fraction
        bias_shift = product_fraction - layer.bias_format.fraction

        # Column-major, exactly as emit_verilog writes the hex file.
        stream = weight.T.reshape(-1)

        accumulator, unit, x_addr, w_addr = 0, 0, 0, 0
        results = []
        while unit < outputs:
            total = accumulator + int(values[x_addr]) * int(stream[w_addr])
            last = x_addr == inputs - 1
            if last:
                biased = total + (int(layer.bias[unit]) << bias_shift
                                  if bias_shift >= 0
                                  else int(layer.bias[unit]) >> -bias_shift)
                rectified = max(biased, 0) if layer.relu else biased
                rounded = ((rectified + (1 << (shift - 1))) >> shift
                           if shift > 0 else rectified << -shift)
                results.append(int(np.clip(rounded, layer.output_format.smallest,
                                           layer.output_format.largest)))
                accumulator, x_addr, unit = 0, 0, unit + 1
            else:
                accumulator, x_addr = total, x_addr + 1
            w_addr += 1
        return np.array(results, dtype=np.int64), w_addr

    @pytest.mark.parametrize("inputs,outputs,relu",
                             [(48, 192, True), (192, 3, False), (48, 3, False)])
    def test_the_engine_reproduces_the_reference_layer(self, inputs, outputs,
                                                       relu):
        layer = TestFixedLayer._layer(inputs, outputs, relu, seed=inputs)
        input_format = QFormat(16, 12)
        values = input_format.quantise(
            np.random.default_rng(inputs).normal(size=inputs))

        expected = layer.forward(values, input_format)
        produced, cycles = self._run_engine(layer, values, input_format)

        assert np.array_equal(produced, expected)
        assert cycles == inputs * outputs, "one weight per cycle, no more"

    def test_dropping_the_last_product_is_caught(self):
        """The bug the engine was written wrong with the first time.

        If the result is formed from the accumulator instead of the
        accumulator plus the current product, the last term of every dot
        product goes missing. This asserts the cycle model would notice.
        """
        layer = TestFixedLayer._layer(48, 16, False, 11)
        input_format = QFormat(16, 12)
        values = input_format.quantise(
            np.random.default_rng(12).normal(size=48))

        # Same walk, but reading the accumulator on the last cycle.
        weight = layer.weight
        inputs, outputs = weight.shape
        stream = weight.T.reshape(-1)
        product_fraction = input_format.fraction + layer.weight_format.fraction
        shift = product_fraction - layer.output_format.fraction
        accumulator, unit, x_addr, w_addr, wrong = 0, 0, 0, 0, []
        while unit < outputs:
            if x_addr == inputs - 1:
                biased = accumulator + (int(layer.bias[unit])
                                        << (product_fraction
                                            - layer.bias_format.fraction))
                wrong.append(int((biased + (1 << (shift - 1))) >> shift))
                accumulator, x_addr, unit = 0, 0, unit + 1
            else:
                accumulator += int(values[x_addr]) * int(stream[w_addr])
                x_addr += 1
            w_addr += 1

        expected = layer.forward(values, input_format)
        assert not np.array_equal(np.array(wrong), expected)
