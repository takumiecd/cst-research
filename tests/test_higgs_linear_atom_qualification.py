"""CPU-only meaningful native-state/critical-oracle and immutable-driver policy gates."""

import copy
import os
import tempfile
import unittest
from pathlib import Path

from cst_research.experiments import higgs_linear_atom_driver as driver
from cst_research.experiments import higgs_linear_atom_qualification as gate
from cst_research.experiments import higgs_linear_atom_scaling as study
from cst_research.experiments import higgs_linear_atom_state as state


class DriverBoundary(unittest.TestCase):
    def test_reject_unlisted_payload_before_reading_any_pinned_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "future-confirmation.npz").write_bytes(b"never load")
            with self.assertRaisesRegex(ValueError, "exact8"):
                gate.verify_inputs(root)

    def test_external_cache_binding_does_not_depend_on_consumer_commit(self):
        pins = gate.input_pins()
        binding = driver.cache_binding(pins, {"gpu": "exact-owner-runtime"})
        self.assertNotIn("consumer_commit", binding)
        with tempfile.TemporaryDirectory() as directory:
            from cst_research.experiments.higgs_depth_normalization_driver import (
                bind_cache,
            )

            root = Path(directory)
            bind_cache(root, binding)
            bind_cache(root, copy.deepcopy(binding))
            with self.assertRaisesRegex(ValueError, "binding differs"):
                bind_cache(root, {**binding, "runtime_pin": {"gpu": "changed"}})

    def test_no_output_into_consumer_or_cache_or_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "consumer", root / "cache"
            source.mkdir()
            for output in (source / "result", cache / "result", root):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    driver.destinations(output, cache, source)
            self.assertEqual(
                driver.destinations(root / "result", cache, source),
                ((root / "result").resolve(), cache.resolve()),
            )


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"),
    "explicit verified public975 CPU fixture required",
)
class ActualNativeState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the actual archive-byte verification harness, not numerical production code.
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def test_exact_H8_seven_owner_load_and_all_other_resume_bytes(self):
        t = self.t
        case = next(c for c in study.pilot_cases() if c["id"] == "N1024-H8-cst-c1")
        model = study.make_case(case)
        optimizer, _ = state.make_optimizer(model, cpu_reference=True)
        for point in model.parameters():
            point.grad = t.full_like(point, 0.001)
        optimizer.step()
        state.check_optimizer(model, optimizer, 1)
        raw = state.snapshot(model, optimizer, 0, 1, case, {"CPU_fixture_only": True})
        resumed, loaded, postload, report = state.load_state(
            case, raw, "cpu", cpu_reference=True
        )
        expected = {f"{i}.atom_state._extra_state": 1 for i in range(2, 16, 2)}
        self.assertEqual(report["loaded"], expected)
        self.assertEqual(len(list(resumed.parameters())), 11)
        state.check_optimizer(resumed, loaded, 1)
        # Native generation is a boundary rule, not a blanket ignored field.
        tampered = state.metadata_copy(postload)
        tampered["model"][next(iter(expected))]["layout_version"] += 1
        with self.assertRaises(ValueError):
            state.exact.assert_equal(tampered, postload, "later-evaluation-mutation")
        for candidate in (model, resumed):
            for point in candidate.parameters():
                point.grad = t.full_like(point, -0.002)
        optimizer.step()
        loaded.step()
        state.check_optimizer(resumed, loaded, 2)
        a = state.snapshot(model, optimizer, 0, 2, case, {})
        b = state.snapshot(resumed, loaded, 0, 2, case, {})
        state.exact.assert_equal(
            state.core_state(b),
            state.with_generations(a, expected),
            "resumed-all-other-bytes",
        )
        # Real model-side and optimizer-side store corruption must be rejected.
        point = next(resumed.parameters())
        old = loaded.state[point].pop("exp_avg")
        with self.assertRaises(ValueError):
            state.check_optimizer(resumed, loaded, 2)
        loaded.state[point]["exp_avg"] = old.reshape(-1)
        with self.assertRaises(ValueError):
            state.check_optimizer(resumed, loaded, 2)
        loaded.state[point]["exp_avg"] = old
        loaded.state[point]["step"].fill_(3)
        with self.assertRaises(ValueError):
            state.check_optimizer(resumed, loaded, 2)
        # Only actual cfg owners/schema may receive the native load transition.
        wrong = copy.deepcopy(raw)
        wrong["model"][next(iter(expected))]["format_version"] = 2
        with self.assertRaises((ValueError, RuntimeError)):
            state.load_state(case, wrong, "cpu", cpu_reference=True)

    def test_relative_gates_reject_zero_signflip_and_nonfinite_even_at_tiny_scale(self):
        t = self.t
        fields = ["Y", "dX", "2.atom_state.atoms.p"]
        reference = (
            t.ones(4, 2).double(),
            t.full((4, 28), 1e-30).double(),
            t.full((2, 4), 1e-30).double(),
        )
        actual = tuple(v.float() for v in reference)
        self.assertTrue(gate.gates(actual, reference, fields, 1)["pass"])
        for replace in (
            t.zeros_like(actual[1]),
            -actual[1],
            t.full_like(actual[1], float("nan")),
        ):
            bad = (actual[0], replace, actual[2])
            self.assertFalse(gate.gates(bad, reference, fields, 1)["pass"])
        material = tuple(t.ones_like(v) for v in reference)
        self.assertTrue(gate.gates(material, material, fields, 1e24)["pass"])
        bad = list(material)
        bad[2] = bad[2].clone()
        bad[2][:, 3] = 0
        self.assertFalse(gate.gates(tuple(bad), material, fields, 1e24)["pass"])
        self.assertFalse(gate.gates(actual, reference, fields, 1e24)["pass"])


if __name__ == "__main__":
    unittest.main()
