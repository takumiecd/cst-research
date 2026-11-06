"""Draft model/algebra checks only; no GPU, task data, recipe or quality validation."""

import hashlib
import json
import os
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from cst_research.experiments import higgs_linear_atom_scaling as draft


class AtomBudgetPolicy(unittest.TestCase):
    def test_exact_cN_no_binary_float_or_silent_floor(self):
        self.assertEqual(draft.exact_atoms(1024, Fraction(13, 128)), 104)
        self.assertEqual(draft.exact_atoms(1024, "131/128"), 1048)
        for width in (1024, 2048, 4096, 8192):
            for c in (1, 4, 16):
                self.assertEqual(draft.exact_atoms(width, c), width * c)
        for c in (0, -1, 0.1, True, "0/1", "nan"):
            with self.subTest(c=c), self.assertRaises(ValueError):
                draft.exact_atoms(1024, c)
        with self.assertRaisesRegex(ValueError, "integral"):
            draft.exact_atoms(1024, "1/3")

    def test_linear_trainable_parameter_algebra_at_both_depths_and_reserved_widths(
        self,
    ):
        for width in (1024, 2048, 4096, 8192):
            for hidden in (4, 8):
                dense = draft.configuration(
                    method="dense",
                    width=width,
                    hidden_layers=hidden,
                    c=None,
                    normalization="none",
                    ordinary_initialization=draft.ORDINARY_INIT,
                )
                self.assertEqual(
                    dense["trainable_scalars"], (hidden - 1) * width**2 + 31 * width + 2
                )
                for c in (1, 4, 16):
                    cfg = draft.configuration(
                        method="cst",
                        width=width,
                        hidden_layers=hidden,
                        c=c,
                        normalization="ln",
                        ordinary_initialization=draft.ORDINARY_INIT,
                    )
                    self.assertEqual(
                        cfg["trainable_scalars"],
                        (4 * c * (hidden - 1) + 31) * width + 2,
                    )
                    self.assertEqual(cfg["normalization_trainable_scalars"], 0)
                    self.assertEqual(cfg["fixed_gain_buffer_scalars"], hidden - 1)
                    self.assertEqual(
                        cfg["internal_module_indices"], list(range(2, 2 * hidden, 2))
                    )
                    self.assertEqual(cfg["architecture"], [28] + [width] * hidden + [2])

    def test_recipe_unsealed_and_later_widths_not_in_short_pilot(self):
        p = draft.protocol()
        self.assertFalse(p["recipe_sealed"])
        self.assertEqual(p["normalization_choice"], "none")
        self.assertEqual(p["ordinary_initialization_choice"], draft.ORDINARY_INIT)
        self.assertFalse(p["training_entrypoint_implemented"])
        self.assertFalse(p["protocol_inspection_GPU_operations"])
        self.assertTrue(p["stage_A"]["requires_native_GPU"])
        self.assertFalse(p["confirmation_or_test_payload_staged_or_scored"])
        self.assertFalse(
            p["planned_candidate_pilot_NOT_authorized_for_execution"][
                "width4096_or8192_included"
            ]
        )
        self.assertEqual(
            p["prototype_initializer"]["H8_adequacy"].split(";")[0], "UNMEASURED"
        )

    def test_explicit_init_norm_depth_and_public_shape_guards(self):
        base = {
            "method": "cst",
            "width": 1024,
            "hidden_layers": 4,
            "c": 1,
            "normalization": "none",
            "ordinary_initialization": draft.ORDINARY_INIT,
        }
        for field, value in [
            ("width", True),
            ("hidden_layers", 5),
            ("ordinary_initialization", "auto"),
            ("normalization", "auto"),
            ("width", 32),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                draft.configuration(**{**base, field: value})
        with self.assertRaises(ValueError):
            draft.configuration(**{**base, "method": "dense"})


SOURCE = os.environ.get("CST_DEPTH_SOURCE")


@unittest.skipUnless(SOURCE, "explicit SHA-verified public975 CPU fixture required")
class PublicFactoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(SOURCE).resolve()
        manifest = json.loads(Path(os.environ["CST_DEPTH_MANIFEST"]).read_text())
        if manifest["source_commit"] != draft.inherited.SOURCE:
            raise ValueError("wrong public975 source")
        for name, value in manifest["files"].items():
            path = source / name
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != value
            ):
                raise ValueError("external source fixture changed")
        sys.path[:0] = [str(source / "src"), str(source)]
        import torch
        import torchcst

        if not Path(torchcst.__file__).resolve().is_relative_to(source):
            raise ValueError("wrong torchcst import boundary")
        torch.set_num_threads(1)
        cls.t = torch

    def factory(self, method="cst", width=128, hidden=4, c=1, norm="none", gains=None):
        return draft.make_model(
            method=method,
            width=width,
            hidden_layers=hidden,
            c=c if method == "cst" else None,
            normalization=norm,
            ordinary_initialization=draft.ORDINARY_INIT,
            seed=17,
            fixed_gains=gains,
        )

    def test_actual_topology_parameter_ownership_no_affine_and_matched_IO(self):
        t = self.t
        from torchcst.nn import CSTModule

        for hidden in (4, 8):
            dense = self.factory("dense", hidden=hidden)
            for method in ("dense", "cst"):
                model = self.factory(method, hidden=hidden, norm="ln")
                cfg = model._linear_atom_configuration
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()), cfg["trainable_scalars"]
                )
                self.assertEqual(len(list(model.parameters())), hidden + 3)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertFalse(model.fixed_output_gains.requires_grad)
                self.assertEqual(model.fixed_output_gains.numel(), hidden - 1)
                self.assertEqual(
                    sum(isinstance(layer, t.nn.ReLU) for layer in model), hidden
                )
                self.assertEqual(
                    sum(isinstance(layer, (t.nn.Linear, CSTModule)) for layer in model),
                    hidden + 1,
                )
                for index in cfg["internal_module_indices"]:
                    layer = model[index]
                    if isinstance(layer, t.nn.Linear):
                        self.assertIsNone(layer.bias)
                    else:
                        self.assertEqual(tuple(layer.atoms.p.shape), (128, 4))
                for index in (0, cfg["output_module_index"]):
                    self.assertTrue(t.equal(model[index].weight, dense[index].weight))
                    self.assertTrue(t.equal(model[index].bias, dense[index].bias))
                optimizer, _ = draft.inherited.make_optimizer(model, cpu_reference=True)
                owned = [p for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(
                    {id(p) for p in owned}, {id(p) for p in model.parameters()}
                )
                self.assertEqual(len(owned), len({id(p) for p in owned}))

    def test_H4_N1024_A104_A1048_old_parameter_and_CPU_forward_parity(self):
        t = self.t
        from torchcst.nn import CSTModule

        for c, divisor in ((Fraction(13, 128), 10000), (Fraction(131, 128), 1000)):
            old = draft.inherited.make_model("all", divisor, 17, 1024)
            new = self.factory("cst", width=1024, c=c)
            self.assertEqual(
                set(dict(old.named_parameters())), set(dict(new.named_parameters()))
            )
            for name, parameter in old.named_parameters():
                self.assertTrue(
                    t.equal(parameter, dict(new.named_parameters())[name]), name
                )
            for model in (old, new):
                for site in model:
                    if isinstance(site, CSTModule):
                        site.forward = lambda value, site=site: t.nn.functional.linear(
                            value, site.dense_weight()
                        )
            x = t.randn(4, 28, generator=t.Generator().manual_seed(1117))
            with t.no_grad():
                self.assertTrue(t.equal(old(x), new(x)))
        old_dense = draft.inherited.make_model("dense", None, 17, 1024)
        new_dense = self.factory("dense", width=1024)
        for name, parameter in old_dense.named_parameters():
            self.assertTrue(
                t.equal(parameter, dict(new_dense.named_parameters())[name])
            )
        with t.no_grad():
            self.assertTrue(t.equal(old_dense(x), new_dense(x)))

    def test_tiny_independent_norm_Y_dX_all_parameter_gradient_oracle(self):
        t = self.t
        # Tiny Dense exercises8-depth and independent norm formulas; no CST width guard bypass.
        for hidden in (4, 8):
            for norm in ("none", "rms", "ln"):
                model = self.factory(
                    "dense", width=8, hidden=hidden, norm=norm
                ).double()
                x = t.randn(
                    4, 28, dtype=t.float64, generator=t.Generator().manual_seed(1701)
                ).requires_grad_()
                actual, reference = model(x), draft.reference_forward(model, x)
                self.assertTrue(t.allclose(actual, reference, atol=1e-13, rtol=1e-10))
                cotangent = t.tensor(
                    [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
                )
                a = t.autograd.grad(
                    actual, (x, *model.parameters()), cotangent, retain_graph=True
                )
                b = t.autograd.grad(reference, (x, *model.parameters()), cotangent)
                for left, right in zip(a, b, strict=True):
                    self.assertTrue(t.isfinite(left).all())
                    self.assertTrue(
                        t.allclose(left, right, atol=1e-11, rtol=1e-8), (hidden, norm)
                    )

    def test_small_public_CST_norm_wrapper_full_gradient_algebra_and_RNG_restore(self):
        t = self.t
        from torchcst.nn import CSTModule

        before = t.random.get_rng_state().clone()
        model = self.factory("cst", width=128, c="1/2", norm="ln").double()
        self.assertTrue(t.equal(before, t.random.get_rng_state()))
        for site in model:
            if isinstance(site, CSTModule):
                site.forward = lambda value, site=site: t.nn.functional.linear(
                    value, site.dense_weight()
                )
        x = t.randn(
            4, 28, dtype=t.float64, generator=t.Generator().manual_seed(22)
        ).requires_grad_()
        actual, reference = model(x), draft.reference_forward(model, x)
        self.assertTrue(t.allclose(actual, reference, atol=1e-12, rtol=1e-9))
        dy = t.tensor(
            [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
        )
        a = t.autograd.grad(actual, (x, *model.parameters()), dy, retain_graph=True)
        b = t.autograd.grad(reference, (x, *model.parameters()), dy)
        for left, right in zip(a, b, strict=True):
            self.assertTrue(t.allclose(left, right, atol=1e-10, rtol=1e-7))

    def test_manual_chunked_full_reverse_matches_all_autograd_fields_without_norm(self):

        from cst_research.experiments.higgs_linear_atom_reference import full_reference

        t = self.t
        for hidden in (4, 8):
            for method in ("dense", "cst"):
                model = self.factory(
                    method, width=128 if method == "cst" else 8, hidden=hidden, c="1/2"
                ).double()
                x = t.randn(
                    4, 28, dtype=t.float64, generator=t.Generator().manual_seed(1817)
                ).requires_grad_()
                cotangent = t.tensor(
                    [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5], [-0.3, 0.1]], dtype=t.float64
                )
                manual_y, manual_x, manual_p = full_reference(
                    model, x, cotangent, chunk_atoms=17
                )
                expected_y = draft.reference_forward(model, x)
                fields = (x, *model.parameters())
                expected_g = t.autograd.grad(expected_y, fields, cotangent)
                actual = (manual_y, manual_x, *manual_p.values())
                expected = (expected_y, *expected_g)
                self.assertEqual(
                    list(manual_p), [n for n, _ in model.named_parameters()]
                )
                for a, b in zip(actual, expected, strict=True):
                    self.assertTrue(t.isfinite(a).all())
                    scale = float(b.detach().abs().max())
                    if scale == 0:
                        self.assertTrue(t.equal(a, b))
                    else:
                        relative = float(
                            t.linalg.vector_norm((a - b.detach()) / scale)
                            / t.linalg.vector_norm(b.detach() / scale)
                        )
                        self.assertLess(relative, 1e-8, (hidden, method, relative))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
        ln = self.factory("dense", width=8, norm="ln").double()
        with self.assertRaisesRegex(ValueError, "NONE"):
            full_reference(ln, t.zeros(4, 28, dtype=t.float64))

    def test_manual_reference_rejects_nonfinite_dtype_shape_norm_and_gain(self):
        from cst_research.experiments.higgs_linear_atom_reference import full_reference

        t = self.t
        model = self.factory("dense", width=8).double()
        valid = t.zeros(4, 28, dtype=t.float64)
        for invalid in (valid.float(), valid[:3], t.full_like(valid, float("nan"))):
            with self.assertRaises(ValueError):
                full_reference(model, invalid)
        for dy in (
            t.zeros(4, 3, dtype=t.float64),
            t.full((4, 2), float("inf"), dtype=t.float64),
        ):
            with self.assertRaises(ValueError):
                full_reference(model, valid, dy)
        model.fixed_output_gains[0] = 2
        with self.assertRaises(ValueError):
            full_reference(model, valid)
        model.fixed_output_gains[0] = 1
        with t.no_grad():
            next(model.parameters()).reshape(-1)[0] = float("nan")
        with self.assertRaises(ValueError):
            full_reference(model, valid)

    def test_invalid_fixed_gain_shape_or_nonpositive_signal_not_hidden(self):
        for gains in ([1.0, 1.0], [1.0, 0.0, 1.0], [1.0, float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                self.factory(gains=gains)
