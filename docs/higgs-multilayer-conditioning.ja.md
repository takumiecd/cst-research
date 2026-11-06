# HIGGS多層conditioning：新しいA100の6条件診断

[English (canonical)](higgs-multilayer-conditioning.md) | 日本語

**PLANNED / PREPARED、未実行。** 問いは、複数CST写像がstallするとき何を直すべきかであり、
LayerNormの必要性を仮定しない。従来の正規化条件はLNとcalibrate済み固定gainを組み合わせていたため、
bare-LN単独の効果を特定しない。この別研究でも旧数値consumer・registryを維持し、
公開torchcstや実行中の研究を変更しない。

## 新しいpaired6条件

全条件で`28 → 1024 × 8 → 2`、biasなし正方写像7個、ReLU8個、学習可能Dense IO、seed17、
変更しない従来初期化を使う。各CST写像は厳密にA=4N=4096 Atom。
Atom/幅/深さ/LR/epsilon sweepは含まない。

| Case ID | 正方写像 | 正規化 | 固定gain |
|---|---|---|---|
| dense-none | Dense | NONE |1|
| dense-bare-ln | Dense | affineなしLN |1|
| cst-none | CST | NONE |1|
| cst-fixed-gain | CST | NONE | calibration値 |
| cst-bare-ln | CST | affineなしLN |1|
| cst-ln-gain | CST | affineなしLN | calibration値 |

全6モデルをA100で新しく学習する。旧RTX4090 state/curveをpaired A100対照へ付け替えない。
各method内で学習前のraw ordinary/Atom tensorとIO seed streamを一致させる。
LNに学習affineはなく、exampleごと1024 featureのpopulation variance、epsilon1e−5を使う。

CSTのgain2条件は、前段のcandidate gain/normも含めて、7個の正方出力を順次
**同一の初期Dense-stock pre-ReLU RMS** に合わせる。
固定された先頭256 TRAIN feature行だけを使い、label/DEV選択は行わない。
CPU FP64の公開chunk referenceで計算し、倍率をFP32へ丸め、[1e−6,1e6]の非学習bufferとして固定する。
target/signalがゼロ・非有限、ratioが範囲外ならclip・fallback・epsilon sweepなしに失敗させる。
raw P/ordinary tensorのbytesは不変でなければならない。amplitude初期化への介入ではない。

\[
\operatorname{LN}(z)=\frac{z-\bar z}{\sqrt{v+10^{-5}}},\qquad
\operatorname{Var}(\operatorname{LN}(z))=\frac{v}{v+10^{-5}}.
\]

小入力は小さいままになり得る。epsilon floorでのLN gainは有限の約316.23倍である。
NONE/正規化なしのbias-free/ReLU/無制限head構造では、固定正gainはglobalな理想実数関数クラスを
保つが、初期関数やAdam座標/epsilonの作用を変える。単純なproduct-gain/head吸収をLN＋gainへ
拡張しない。途中のLN epsilonが正の同次性を壊す。LN自体も入力依存の関数mapを変える。
どちらも必要・十分・同等とは仮定しない。

## 固定budgetと実際の曲線

B4、FP32、TF32 off、eagerのstock q1 CSTAdamW、LR0.0003、betas(0.9,0.999)、
Adam epsilon1e−8、ordinary/amplitude weight decay0.01、raw Atom weight decay0。
公開975のphysical radius-cell-mass M8/merge2/spacing1/birth4設定は変更しない。
各条件full100kの1epochで実task更新25000回、合計150000回。
実際の`randperm(100000)`、seed117の順序を保存し、6条件で共有する。

updates0、1000、5000、10000、15000、20000、25000において、**同じstateでfull TRAIN100kと
DEV50k** を評価する。logits、labels、raw row IDs、再計算したgapを保存する。
事前指定した診断曲線でありHPOではない。選択は **固定final25000** のみ。
最小DEV、early stop、自動延長はしない。その全stateを厳密なnative7owner generation遷移でloadし、
TRAIN/DEVのarray bytes・metricsがcached finalと一致することを要求する。
以後のmutationは実post-load baselineと比較し、metadata fieldを削除しない。

## task学習前の新しい数値・観測admission

実際の各H8条件で、新しい手動CPU FP64 Dense/ReLU/LN/gain逆伝播と、変更しない975の
256 AtomずつのCST VJPを使い、full Y/dX/全11parameter勾配を確認する。
全実Atom/幅を扱い、CST full-network autograd graphや`[A,N,N]` tensorは保持しない。
小さい公開dense-export/full autogradとfunctional LNを使う独立テストで、epsilon floorも含め
新しい逆伝播を確認する。旧NONE専用referenceのguardを回避・弱化しない。

通常cotangent1、増幅1e24はGPU凍結前にCPUでrepresentability/materialを事前確認する。
通常elementwise許容差は3e−5+3e−4|ref|。両scaleでgradient/packed各列の安定FP64 relative L2と
maxerror/refmax≤3e−4を要求し、厳密なゼロreferenceには厳密なゼロactualを要求する。
増幅dXとpacked各列に、|ref|>3e−5+3e−4|ref|となる要素と非ゼロactualが必要。
増幅componentwise失敗は診断として残し、通常FP32のrounded-zero tailも免除なしに記録する。
増幅勾配をoptimizerへ適用しない。全actual/expected VJP array、入力、cotangentを保持する。

実31 Atomの独立48-point quadrature、native scalar-Adam、通常合成CEの連続8更新と
4＋native-load＋4、観測first＋bare7更新と変更しない`ordinary_step`8更新を照合する。
全model/gradient/m/v/clock/native owner storeを一致させる。
load時に`max(before+1,serialized)`へ進めるのは実owner2/4/6/8/10/12/14の7個のみ。
これは各条件20回、合計120回の合成qualification更新で、**task学習ではない**。
全task modelはqualified initial0と厳密に同じfresh0から始める。

実task update1では本物のB4 features/labels/order、raw backward勾配、accepted P/m/v/clocks、
post-step勾配、層ごとのforward/upstream勾配、安定なtorus/FP32変位を保存する。
25000回の一部であり追加更新ではない。このgrad-enabled traced forwardだけでTensor.retain_gradを使い、
optimizer monkeypatchやmodule/full-backward hookは使わない。
raw/post-step一致とcanonical FlatTorus identityの意味を報告し、prepared-buffer捕捉は主張しない。
denominatorは別途scalar-qualifiedな **accepted-state/source-FP32-opmath再構成** であり、
native intermediate bufferの捕捉ではない。accepted変位にはretraction/amplitude decayを含む。
実非ゼロFP32勾配上のepsilon割合と全要素上の割合を分け、因果証明にしない。

固定256 TRAIN featureのforward診断を初期/終点で時計外に行う。
終点では同じ最初のB4を独立したfull-loadコピーでbackwardし、optimizer更新を行わない。
コピーの明示した勾配以外は変えず、元model/optimizer/保存勾配も厳密に保つ。
残りの更新は変更しない`ordinary_step`を使い、1000更新ごとの記述的rolling lossを記録する。

## source・runtime・保持・実行

外部sourceは`97580582ed28c4e494bf5c2656bbe94a85c544c3`、manifestは
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`。
data artifactは`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`、
manifestは`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`。
既存feature exposure・過去のadaptive DEV使用を引き続き明示し、新しい正式confirmationとはしない。
future/original confirmationやofficial testはstageしない。

A100 runtime pinは`35016be129960614d829affbea959797ae8d035b5ea41197b6803ae963709f55`：
Python3.12.9/Torch2.11.0+cu126/CUDA12.6/Triton3.6.0/NumPy2.0.2、driver570.86.15、
厳密にNVIDIA A10080GB PCIe MIG3g.40gb、CC8.0、42SM、GPU42144366592 bytes。
親80GB/4090への置換はしない。Hub操作はownerだけが行う。
foreground consumerは何もinstallせずsessionも確保しない。

```bash
python -B -m cst_research.experiments.higgs_multilayer_conditioning_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/source-runtime-verified-cache \
  --output /path/to/fresh-diagnostic-output --timeout 7200
```

厳密な6個のregular入力はconsumer-source-manifest.json、original-source-manifest.json、
original-source.tar.gz、train-development.npz、data-manifest.json、runtime-pin.json。
レビュー済みclean source snapshotと全入力bytesを実行前に確認する。
cacheは公開source/runtimeのcompilation pathを固定し、変更bytesを拒否する。
timeoutはowned child process groupを回収し、失敗証拠を残してretryしない。

時間測定は **DIAGNOSTIC** であり旧Bとの性能比較ではない。
calibration/admission/first observation/evaluation/snapshot/I/O/probeを分ける。
training-block walltimeには指定1000更新ごとのloggingを含み、成分timerは非加算。
checkpoint/eval前にtraining peakをreset/取得し、resident arrays/model/optimizer/native buffersと
残存allocator reservationの範囲を記載する。

qualified0/4/8/resumed8/observer8、実first-step raw、full final stateを保持する。
中間点は予測と **完全typed core** のSHAを保持し、全row ID/m/v/clock/metadata fieldを含める。
producer digestであり独立raw-midstate replayではない。
load proofは7owner mapとcomplete core SHAを残し、巨大なtyped metadata treeを繰り返し埋め込まない。
保守的上限はraw6GiB、JSON64MiBで、launch前に実inventory/RSSを測る。
full rawはA100で不変に保持してよい。compact proof/predictionを回収し、実空き容量が足りる場合だけ
local compressed archiveを置くか、CUDA無効・rootがpinした別peer CPU saved-evidence auditorを
server-side実行してreport/selected predictionsを回収する。
localへの全展開重複コピーや既存証拠の削除は必須としない。
rootが実storage/runtimeをレビューしてからlaunchする。

旧scientific moduleと11 registryのbytesはすべて不変で、旧120file中119fileはそのまま。
唯一のmetadata例外はregistry-count testの11→12で、元bytesを再構成可能とし、validator gateは弱めない。
正式NI、Atom-order/capacity判定、LN必要性、公開default採用、自動cross-device追試は主張しない。
証拠locatorは`$CST_RESEARCH_EVIDENCE_ROOT/higgs-multilayer-conditioning-20261106/`。
