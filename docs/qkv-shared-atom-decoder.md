# Shared Atom decoder for QKV

English | [日本語](qkv-shared-atom-decoder.ja.md)

This pilot uses the frozen width-1024, c64, seed-17 TinyStories QKV rational
envelope run as its control. The four raw values at each Atom serve as the
initial latent vector. One shared residual 4 → 16 → 4 SiLU MLP decodes the
vector into amplitude, radius fraction, output center and input center. The
same rational envelope decodes the fraction to physical radius. The zero
output layer starts the decoder at the identity on the initial Atom rows.
The MLP and latent vectors train jointly through the usual next-token CE.

The MLP has 148 trainable scalars. It consumes the full task gradient across
all 12 QKV projections, with its own gradient norm clip of 0.1 and the base
learning-rate schedule. Other parameters use the reference clipping threshold
of 1.0 and the Atom learning-rate multiplier of 32. The run uses the reference
initialization, token order, training and development splits, 4882-step
budget and longer learning-rate horizon. It measures a residual
parameterization with a fixed hand-designed radius envelope and one seed.

Use the frozen Python sources and archived protocol of the reference. Put
the frozen source trees on `PYTHONPATH` and use the same CUDA build
environment as the reference. Pass the two inherited source-change manifests
in chronological order. The output directory must be new.

```bash
python "$CHECKOUT/src/cst_research/experiments/qkv_shared_decoder_run.py" \
  --reference "$QKV_REFERENCE" --protocol "$WIDTH_PROTOCOL" \
  --source-changes "$RADIUS_CAP_MANIFEST" --source-changes "$SQUARE_QKV_MANIFEST" \
  --output "$OUTPUT"
```

Before training, the runner compares the initial task loss and latent
gradients with the direct QKV model. It records development CE, per-band
physical radius medians, gradient norms, time, memory and checkpoint hashes.
The final checkpoint is reloaded and evaluated.
