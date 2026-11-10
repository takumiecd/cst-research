"""Fresh M8 c16 and two active-count DST LR families; old4A DST is auxiliary."""

import argparse
import copy
import io
import json
from pathlib import Path

import torch

from . import ffn_dst
from .ffn_model import parameter_digest
from .ffn_radius_fraction_qualification import assert_model_resume, assert_nested
from .ffn_run import sha, token_batch, verify_data
from .ffn_selective import active_ffn_scalars, projections
from .lm_connected import step
from .lm_geometry_trial_run import audit, full_oracle, train, write
from .lm_m8_active_count import ARMS, active_slots, model_for, optimizer_for
from .lm_m8_capacity import model_for as m8_model
from .lm_m8_capacity_run import ordinary, prior
from .lm_placement_probe import probe


def count_audit(root, output):
    report = {"state": "running", "arms": {}}
    for arm, (kind, coefficient, _) in ARMS.items():
        records = []
        for clock in (256, 976):
            saved = torch.load(
                root / arm / f"checkpoint-{clock}.pt",
                map_location="cpu",
                weights_only=False,
            )
            m = model_for(arm)
            m.load_state_dict(saved["model"])
            o = optimizer_for(m)
            o.load_state_dict(saved["optimizer"])
            slots = active_slots(m)
            assert set(slots.values()) == {coefficient * 1024}
            records.append(
                {
                    "step": clock,
                    "slots_per_map": slots,
                    "dst_invariants": ffn_dst.invariants(m, o) if kind == "dst" else {},
                }
            )
            del m, o, saved
            torch.cuda.empty_cache()
        report["arms"][arm] = records
        write(output / "active-count-audit.json", report)
    report["state"] = "passed"
    write(output / "active-count-audit.json", report)


def select(root, output):
    records = []
    for arm, (kind, coefficient, lr) in ARMS.items():
        if kind != "dst":
            continue
        summary = json.loads((root / arm / "summary.json").read_text())
        assert summary["state"] == "completed" and summary["completed_steps"] == 976
        curve = [
            json.loads(s)
            for s in (root / arm / "curves.jsonl").read_text().splitlines()
        ]
        terminal = next(s for s in curve if s["step"] == 976)["development"]
        records.append(
            {
                "arm": arm,
                "coefficient": coefficient,
                "nnz_per_map": coefficient * 1024,
                "sparse_lr": lr,
                "development": terminal,
            }
        )
    selected = {
        str(c): min(
            (s for s in records if s["coefficient"] == c),
            key=lambda s: (s["development"]["ce"], s["sparse_lr"]),
        )["arm"]
        for c in (8, 16)
    }
    write(
        output / "selection.json",
        {
            "state": "passed",
            "criterion": "minimum final DEV CE, tie smaller LR",
            "selection_on_development": True,
            "confirmation_evaluated": False,
            "candidates": records,
            "selected": selected,
        },
    )


def qualify(output, data, previous):
    old = prior(previous)
    base = m8_model("m8-c8", device="cpu")
    assert (
        parameter_digest(base) == old["arms"]["legacy-m8"]["initial_parameters_sha256"]
    )
    shared = ordinary(base)
    del base
    report = {
        "state": "running",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "matching_criterion": "CST Atom count equals DST mask-one count per map",
        "prior_manifest_sha256": sha(previous / "manifest.json"),
        "m8_c8_initial_parameters_match": True,
        "ordinary_parameters_match": True,
        "arms": {},
    }
    initial = {}
    for arm, (kind, coefficient, _) in ARMS.items():
        m = model_for(arm)
        assert ordinary(m) == shared
        digest = parameter_digest(m)
        if kind == "cst":
            assert digest == old["arms"]["m16-c16-e8"]["initial_parameters_sha256"]
        else:
            assert initial.get(coefficient, digest) == digest
            initial[coefficient] = digest
        row = {
            "initial_parameters_sha256": digest,
            "active_slots": active_slots(m),
            "active_ffn_scalars": active_ffn_scalars(m),
        }
        assert set(row["active_slots"].values()) == {coefficient * 1024}
        report["arms"][arm] = row
        if kind == "cst":
            row["oracles"] = {
                s: full_oracle(projections(m)[s].map) for s in ("1.up", "1.down")
            }
            write(output / "qualification.json", report)
            assert all(v["passed"] for v in row["oracles"].values())
        o = optimizer_for(m)
        assert all(g["eps"] == 1e-8 for g in o.param_groups)
        batches = [token_batch(data["train"], [0, 128], 128, "cuda")]
        step(m, o, batches, 1)
        payload = io.BytesIO()
        torch.save({"model": m.state_dict(), "optimizer": o.state_dict()}, payload)
        payload.seek(0)
        saved = torch.load(payload, weights_only=False)
        other = model_for(arm)
        other.load_state_dict(saved["model"])
        oo = optimizer_for(other)
        oo.load_state_dict(copy.deepcopy(saved["optimizer"]))
        probe(m, *token_batch(data["train"], [0], 128, "cuda"), retain=True)
        step(m, o, batches, 2)
        step(other, oo, batches, 2)
        assert_model_resume(m, other)
        assert_nested(o.state_dict(), oo.state_dict())
        row["serialized_resume_and_probe_next_update_exact"] = True
        write(output / "qualification.json", report)
        print("QUALIFIED", arm, flush=True)
        del m, o, other, oo, saved
        torch.cuda.empty_cache()
    report["state"] = "passed"
    write(output / "qualification.json", report)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("qualify", "train", "audit", "select"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--previous-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090"
    assert torch.cuda.get_device_capability() == (8, 6)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(a.data / "manifest.json") == a.data_manifest_sha256
    _, data = verify_data(a.data)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.mode == "qualify":
        qualify(a.output, data, a.previous_run)
    elif a.mode == "audit":
        audit(
            a.run,
            a.output,
            data,
            arm_names=ARMS,
            model_builder=model_for,
            optimizer_builder=optimizer_for,
        )
        count_audit(a.run, a.output)
    elif a.mode == "select":
        select(a.run, a.output)
    else:
        q = json.loads(a.qualification.read_text())
        assert q["state"] == "passed" and q["torch"] == torch.__version__
        prior(a.previous_run)
        kind, coefficient, sparse_lr = ARMS[a.arm]
        train(
            a.arm,
            a.output,
            data,
            a.data_manifest_sha256,
            q,
            model_builder=model_for,
            optimizer_builder=optimizer_for,
            declaration=(
                kind,
                coefficient,
                1e-8,
                "legacy-m8" if kind == "cst" else f"DST nnz=A, sparse_lr={sparse_lr}",
            ),
        )


if __name__ == "__main__":
    main()
