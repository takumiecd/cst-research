# HIGGS A100 MIG環境の限定比較

[English (canonical)](higgs-gpu-comparison.md) | 日本語

予約済みの **NVIDIA A100 80GB PCIe MIG 3g.40gb** 環境と、以前のL4予算pilotを比較する
独立した運用実験です。実行中のHIGGS alpha確認実験のfamilyを変更・拡張しません。
このA100区画は42 SM、42,144,366,592 bytes、compute capability8.0で、A100全体ではありません。
DST、密度grid、学習率調整、held-out評価はこのprotocolに含めません。

## ソースとworkloadの境界

adapterは `cst_research.experiments.higgs_gpu_comparison` です。
torchcstのcommit `97580582ed28c4e494bf5c2656bbe94a85c544c3` を `git archive` した
外部ソースと、そのprivate HIGGS Graph benchmark adapterを変更せず使います。
torchcst runtimeを本repoへコピーせず、公開の学習APIとして扱いません。
実行前に本repoのadapter commitと両source manifestを固定します。

両群とも元の `benchmarks.training.higgs_confirmation` runnerを使います。

- N2048、28→2048→2048→2 ReLU、全層学習、中間層biasなし。
- 中間層denseと、中間層CSTのdivisor1000、alpha=0.001を比較します。
  atomは floor(2048²/1000)=4194個、各4parameterです。
- seed17、LR0.001、FP32、TF32 off、CPU math threads1、真batch4。
  AdamW/Polar更新則と初期化streamは元のままです。
- 1 full epoch、train100,000例、25,000実optimizer更新。
  developmentはvalidation [0,50000) のみ。
  confirmation [50000,100000) と公式testは評価しません。
- prepared archive SHA256:
  `62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac`。

以前の参照はL4 qualification job `l4job-0ac0ee0f5deb442eb09b8d63a18d1f4a` の
`budget-pilot-2048-dense` と `budget-pilot-2048-cst` です。
無関係なHPO曲線でなく、実際のsource/data/order/runtime記録を比較します。

## Runtimeとresource運用

status、前景実行、upload/downloadには **kyutech-a100** スキルの同梱clientだけを使います。
Jupyter contents上の `srv11/cst-lab/higgs-a100-20261105-*` 以下へ分離します。
実行kernelとcontents APIで相対rootが異なる場合があるため、展開・実行前に対応を確認します。
tokenの閲覧・コピー、server profile変更、global package変更、他job操作、GPUの代替を行いません。

初期server stackはPython3.12.9、Torch2.6.0+cu126、CUDA/NVCC12.6、driver570.86.15、
Intel Xeon Gold6348です。公式Torch2.11.0+cu126 CPython3.12 wheelを利用できます。
依存はjob専用の分離virtual environmentだけへ導入します。
L4に合わせるためにcu130を入れたりdriverを変えたりしません。
実際のTorch/Triton/Python/CUDA/NVCC/driver/CPU/device特性を記録します。
両方Torch2.11でもCUDA/compiler/driver/Python/CPUは異なり得ます。
これは **環境を含む比較** で、GPU hardwareだけの因果比較ではありません。
失敗を保全し、計時のためにstock関数・finite・metadata・lifecycle・captureのguardを迂回しません。

## 計時前のadmission

全source fileのbyteとprepared data hashを確認し、予約されたMIGの厳密なidentityを要求します。
両pilotの開始前に次を実行します。

1. pinned HIGGS cohort、stock qualification、topology guardのテストを変更せず実行します。
   finite/bootstrapとcapture契約の確認を含みます。
2. N2048/A4194のY、dX、全atom parameter勾配を独立した公開CPU FP64 local Fourier参照と
   比較します。元の誤差基準を使い、許容誤差を緩めません。
3. 保持する31 atom subsetを独立48点quadratureと比較し、Y/dX/全subset parameter勾配を確認します。
4. dense/CSTそれぞれeager/Graphの2epoch×16 minibatchを実行します。
   dev32はこのqualification modeだけで許可します。
   両epoch境界の実model/optimizer/gradient/clock Tensorをbyte比較し、
   32実更新と5個のAdam clock一致を要求します。captureと評価→train境界を含みます。

計時前にraw oracle/checkpoint Tensor、log、比較とpassed qualification記録を保存します。
一つでも失敗すれば実験を止め、自動fallbackを合格としません。
qualification費用と2つのpilot計時は区別します。

## 実行と証拠

予約serverは起動済みである必要があります。接続・転送はadapterでなくresource skillが担当します。
分離したsource/input/result directoryへの配置後の例です。

```bash
PYTHONPATH=RESEARCH_SOURCE/src JOB_ENV/bin/python \
  -m cst_research.experiments.higgs_gpu_comparison \
  --torchcst-source TORCHCST_SOURCE \
  --torchcst-manifest TORCHCST_SOURCE_MANIFEST.json \
  --adapter-manifest ADAPTER_SOURCE_MANIFEST.json \
  --adapter-commit FULL_RESEARCH_COMMIT \
  --data INPUTS/screen.npz --output FRESH_RESULTS
```

manifestは `source_commit`、`archive_sha256`、安全なrelative pathからSHA256への `files` を持ちます。
明示Git commitからarchiveし、archiveと展開fileのhashを独立確認し、証拠へ両方保持します。
adapterは余分なsource fileを拒否し、実行後に両treeを再検証します。
CUDA compileはjob専用extension cache、MAX_JOBS2、architecture8.0を使います。
全processを前景・期限付きで実行し、serviceやdetach jobを作りません。

`experiment.json` に進捗・runtime・数値gate・両pilot結果、`manifest.json` に全result fileのhashを保存します。
raw initial/best-development/final checkpoint、prediction、source/result archive、
package/setup logとmanifestをignored外部証拠に保全します。
完了を報告する前にdownloadし、hashと実Tensorをローカル確認します。
compiled extensionのbyteは実物を保持した場合だけ独立確認済みとします。

同期した完全training time、準備込み時間、capture込みallocated/reserved peakを比較します。
元runnerのtraining timeはvalidation/checkpoint保存/hash診断/data/model配置を除外します。
境界を維持し、同環境の2群を併記し、以前のL4 jobのprovenanceも残します。
1epoch・1seedの限定運用比較であり、最終精度や一般的hardware scalingの結果ではありません。
