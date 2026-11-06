"""StageB policy and real initial-state/metadata admission checks; never a CPU epoch."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from cst_research.experiments import higgs_linear_atom_learning as learning
from cst_research.experiments import higgs_linear_atom_learning_evidence as evidence
from cst_research.experiments import higgs_linear_atom_scaling as model_contract
from cst_research.experiments import higgs_linear_atom_state as state


class LearningPolicy(unittest.TestCase):
    def test_bounded_16case_NONE400k_does_not_substitute_qualification_or_smaller_success(
        self,
    ):
        protocol = learning.protocol()
        cases = protocol["cases"]
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(c["method"] == "dense" for c in cases), 4)
        self.assertEqual(
            {
                (c["width"], c["hidden_layers"], c["c"])
                for c in cases
                if c["method"] == "cst"
            },
            {(n, h, c) for n in (1024, 2048) for h in (4, 8) for c in (1, 4, 16)},
        )
        self.assertEqual(
            protocol["main_updates"], len(cases) * protocol["updates_per_case"]
        )
        self.assertEqual(protocol["main_updates"], 400000)
        self.assertEqual(protocol["epochs_per_case"], 1)
        self.assertFalse(protocol["heldout_evaluated"])
        self.assertEqual(protocol["new_qualification_or_replay_updates"], 0)
        self.assertFalse(protocol["conditional_normalization_or_gain_adoption"])

    def test_missing_actual_passed_admission_stops_before_any_tensor_or_taskdata_read(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "proof bytes"):
                evidence.verify_stage_a(
                    root,
                    root / "not-yet-admitted.json",
                    "1" * 64,
                    root / "missing-release.json",
                    "2" * 64,
                    root,
                    {},
                )
            # A preparation/partial report is not an actual16 certificate.
            pending = root / "pending.json"
            pending.write_text(
                json.dumps({"status": "PREPARED", "all16_passed": False})
            )
            digest = hashlib.sha256(pending.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "status"):
                evidence.verify_stage_a(
                    root, pending, digest, pending, digest, root, {}
                )

    def test_unknown_future_payload_rejected_before_decoding_or_hashing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "future-confirmation.npz").write_bytes(b"must-not-load")
            with self.assertRaisesRegex(ValueError, "exact10"):
                learning.verify_inputs(root, "1" * 64, "2" * 64)

    def test_readonly_full_inventory_rejects_extra_file_and_byte_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.pt"
            raw.write_bytes(b"unchanged")
            expected = {
                "raw.pt": {
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "bytes": raw.stat().st_size,
                }
            }
            evidence.strict_files(root, expected)
            extra = root / "unexpected.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory"):
                evidence.strict_files(root, expected)
            extra.unlink()  # Only a test-owned temporary file.
            raw.write_bytes(b"tamperxxx")
            with self.assertRaisesRegex(ValueError, "bytes"):
                evidence.strict_files(root, expected)


@unittest.skipUnless(
    os.environ.get("CST_DEPTH_SOURCE"), "explicit verified975 CPU fixture required"
)
class ActualInitialState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_higgs_linear_atom_scaling import PublicFactoryChecks

        PublicFactoryChecks.setUpClass.__func__(cls)

    def test_actual_public_H8_initial_checkpoint_allowed_but_never_qualified_step8(
        self,
    ):
        case = next(
            c for c in model_contract.pilot_cases() if c["id"] == "N1024-H8-cst-c1"
        )
        model = model_contract.make_case(case)
        optimizer, _ = state.make_optimizer(model, cpu_reference=True)
        identity = {"CPU-only-fixture": True}
        raw = state.snapshot(model, optimizer, 0, 0, case, identity)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "initial0.pt"
            self.t.save(raw, path)
            proof = {
                "producer_identity": identity,
                "cases": {
                    case["id"]: {
                        "initial_state_file": "initial0.pt",
                        "initial_state_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
            restored = evidence.load_initial(root, proof, case)
            state.exact.assert_equal(
                state.core_state(restored),
                state.core_state(raw),
                "actualinitialfixture",
            )
            for key, value in (("updates", 8), ("epoch", 1)):
                bad = copy.deepcopy(raw)
                bad[key] = value
                self.t.save(bad, path)
                proof["cases"][case["id"]]["initial_state_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                with self.assertRaisesRegex(ValueError, "freshinitial0"):
                    evidence.load_initial(root, proof, case)
            # Actual optimizer history with forged zero counters is still not initial0.
            for point in model.parameters():
                point.grad = self.t.full_like(point, 0.001)
            optimizer.step()
            dirty = state.snapshot(model, optimizer, 0, 0, case, identity)
            self.t.save(dirty, path)
            proof["cases"][case["id"]]["initial_state_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "freshinitial0"):
                evidence.load_initial(root, proof, case)


if __name__ == "__main__":
    unittest.main()
