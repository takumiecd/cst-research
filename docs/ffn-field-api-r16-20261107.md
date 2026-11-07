# Public field learning-rate API validation (R16)

English (canonical) | [日本語](ffn-field-api-r16-20261107.ja.md)

The public absolute `field_lrs` API passed its native and actual-LM gates and
merged through [torchcst PR151](https://github.com/takumiecd/torchcst/pull/151),
merge commit `c3b68dcf6275f96d29a21ff2aa6e989d42de2cff`. The tested operator was
`a6c5f12a31442a03afb79cf47ebdb88db13afbb7`; final PR head21ac2029 only adds
validation notes and preserves runtime/test bytes. Its final CPU CI42324636658
passed. The separate `[ffn]` extra pins the merged library; historical `[torchcst]`
retains HIGGS975 and must use a separate environment.

Use one ordinary group at0.0003 and an Atom group at0.0048 when reproducing the
field-rate distinction. In public API terms:

```python
optimizer = CSTAdamW([
    {"params": ordinary_parameters, "lr": 3e-4},
    {"params": atom_parameters, "lr": 4.8e-3,
     "field_lrs": {"amplitude": 2.4e-3, "input_center": 4.8e-3,
                   "output_center": 4.8e-3}},
], model=model, weight_decay=0.0)
```

These are configurable initial absolute rates, not a recommended universal
recipe. Unspecified fields and Polar outer time use group LR. Standard
schedulers supply a common factor through the initial-group anchor;
`get_field_lrs()` reports current effective overrides. Fields come from kernel
parameterization/geometry declarations, not raw columns. There remains one
Parameter, moment table and counter per Atom site. q/activity stays coupled to
amplitude motion. See the
[canonical optimizer contract](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/cst-optimizer.md#field-learning-rates).

Actual RTX4090 CC8.9/24GB-class, Torch2.11.0+cu126, Python3.12.9, NumPy2.0.2:
25 remote CPU tests and8 permanent native-CUDA tests passed with0 skips. The
CUDA cases cover native/wrapper, full field peaks, CPU FP64 Parameter/moment/
counter comparisons, schedules including0 and a serialized next update.

Eight fresh disposable four-block N256/FFN4N/c8 models then checked native/
wrapper × amplitude8/16/32/64 at centers/outer16. Every configuration used all
eight CST FFN projections, ordinary attention/embed/output, batch2/context128/
accumulation4, FP32/TF32off, AdamW(.9,.95), epsilon1e-8, decay0 and global clip1.
Twenty ordinary warmup updates plus5 full-peak stress task updates on the next
ordered batches matched the research adapter exactly in all loss and Parameter
observations(maximum difference0). Gradients/moments passed recorded FP32
tolerances; counters matched. A public optimizer constructed without overrides
restored checkpoint field settings and reproduced the26th full-peak task update
with parameter difference0. All64 trained projection Y/dX/all-Atom VJPs passed
CPU FP64 chunked-reference checks.

This is correctness evidence. The stress sequence is not a normal20M learning
trajectory or quality/speed/memory experiment. No original learner was resumed
or checkpoint overwritten, and no retry was needed. Historical R12–R15 quality
results keep their old operator/adapter provenance. Eager field rates are
qualified in this scope; the stock Graph benchmark rejects them during cold
qualification and falls back to eager. No Graph or wider-hardware claim follows.

All8 retrieved archive members, including the manifest, matched SHA256:

- Archive: `64ae9bac51b1565d35cae76b875f45096a9145de663a2cb0f65ffd98367dfc06`.
- Manifest: `c37a46db68037dbdc5b3b36a1d484df8047251e222f4b86719e2a399137471f2`.
- Full-LM result: `49125fd92525bde1fba4ee1220fbe92a5fdcb9534852b626ccc5d1453b880800`.
- Tested operator archive: `aeac783c7053f97374337f1fb0a918ae5d69e90337b05d4a0df3ed9cc8c6af7a`.
- Consumer: `351617dc81f529ce10e964e5a762f336b7c24b39`.
- Data manifest: `5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.

The registered external evidence locator retains controller/requests, frozen
sources, JUnit/logs, `qualify-public-field-api-r16-02.py` and the full result.
Run the script with `--data TOKEN_DIR --output NEW_DIR` in the recorded numeric
runtime using the pinned operator/consumer. Permanent small cases live in
`tests/test_optimizer_field_lrs*.py` in torchcst. The
[upstream validation note](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/research-history/optimizer-field-lrs-20261107.md)
records their reproduction commands and local CPU/build scope. Local2121passes/
2474skips did not stand in for these GPU gates. The owned validation worker
finished; the user-owned Hub remains running.
