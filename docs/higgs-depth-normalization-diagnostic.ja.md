# HIGGS：8更新でスケールと中心化を分ける診断

[English](higgs-depth-normalization-diagnostic.md) | 日本語

study `higgs-depth-normalization-diagnostic-20261106` は**準備済み・未実行**です
（registry statusはPLANNED）。観測後の機構診断であり、完了済みf6／69 pilotや別途
進行中のHIGGS確認実験に学習曲線、密度／LR探索、品質選択、自動追試を追加しません。
スケールと符号別エネルギーの疑問を調べるためであり、A104停滞の原因やLayerNormの
効果が証明されたわけではありません。

## 固定範囲

28→1024→1024→1024→1024→2、biasなし正方中間層3つとReLU4つです。
通常の入力／出力層も学習し、全学習テンソルを変更していないf6 factoryで初期化して
対応するstockと一致させます。公開operatorとstockの複数site対応CSTAdamWはexternal975
を使い、q1、LR=.0003、通常WD=.01、振幅decay=.01、raw atom WD=0を維持します。
全caseをFP32、TF32 off、B4、eagerで比較します。公開Graph guardや既定値は変更せず、
その適格性も主張しません。

seed17、最初のepochの `torch.randperm(100000, Generator(seed=117))` の先頭32
indicesから、**各caseで真の学習更新を正確に8回だけ**行います。合成入力のqualificationは
別の新しいモデルで実施し、task状態へ引き継ぎません。epoch、HPO、development評価、
heldout参照、case選択はありません。9 caseと実行順序を固定します。

| Case | Squares | Intervention before each hidden ReLU |
|---|---|---|
| dense-stock | Dense | None |
| dense-rms | Dense | No-affine RMS + calibrated fixed positive gain |
| dense-ln | Dense | No-affine LayerNorm + calibrated fixed positive gain |
| cst104-stock | Three CST, A104 each | None |
| cst104-gain | Three CST, A104 each | Calibrated fixed positive gain, Dense pre-ReLU target |
| cst104-postgain | Three CST, A104 each | Calibrated fixed positive gain, Dense post-ReLU target |
| cst104-rms | Three CST, A104 each | No-affine RMS + calibrated fixed positive gain |
| cst104-ln | Three CST, A104 each | No-affine LayerNorm + calibrated fixed positive gain |
| cst1048-stock | Three CST, A1048 each | None; positive learning anchor from earlier budget only |

正規化は各例の1024特徴量軸に独立に適用し、各正方層の後・ReLU前に置きます。
**学習するaffineやbiasはありません**。

\[
R_\epsilon(z)=z/\sqrt{\operatorname{mean}_{features}(z^2)+\epsilon},\quad
L_\epsilon(z)=\frac{z-\operatorname{mean}_{features}(z)}
{\sqrt{\operatorname{mean}_{features}((z-\operatorname{mean}_{features}z)^2)+\epsilon}},
\quad\epsilon=10^{-5}.
\]

実行側RMSはεを明示したPyTorch `functional.rms_norm`、LNは
`functional.layer_norm` を使います。独立CPU-FP64参照は手計算の平均／二乗式と
公開 `dense_weight()` を用い、同じnorm operatorを呼び直す参照にはしません。

## 実行前の準備変更

最初のGPU実行より前に、出力寄与スケールに関するユーザーの仮説を受け、9番目の
`cst104-postgain` を追加しました。既存の `cst104-gain` はReLU前目標のままです。
新caseだけは、先行するgainを適用した各正方層のReLU後RMSをDense-stockの
site3/5/7の初期RMSに合わせます。正の倍率はReLU前に掛け、理想実数での可換性を
使います。ε・gain範囲・拒否条件は同じです。未実行の8ケースprototypeとSHA manifestを
別に保存しており、結果を見て変えたものではありません。真のtask更新は9×8回だけです。

## タスクラベルを使わない校正

学習データの元の順序の先頭256特徴行を使い、以前のgain pilotと同じDense-stock初期
3正方層出力RMSを目標にします。各norm caseで、先行する正規化と校正gainを順に適用し、
対応するReLU前目標（postgainのみReLU後目標）に合わせる正のFP32 post-norm倍率を固定します。Dense norm対照にも同じ
手順を適用します。A104のgain-onlyはnormを使わず、stock／anchorは倍率1です。
normのεは振りません。信号ゼロ・非有限や、必要gainが継承した固定範囲[1e−6,1e6]を
外れた場合は拒否し、clip、別目標、自動fallbackは使いません。学習テンソルを変更したり、
ラベル／developmentを校正に使ったりしません。

## 真のタスク更新より前のadmission

どのtask probeを始める前にも、全9 caseで次を要求します。

1. 独立CPU-FP64の全network Y/dX/全parameter勾配を確認します。f6／69の通常の
   要素別許容差3e−5+3e−4|reference|と、相対norm／reference最大値での厳密基準
   3e−4を維持し、CSTの各packed列も確認します。増幅cotangentでのmaterial／norm
   gateも継承します。失敗した増幅componentwise stressを消したり成功扱いしたりしません。
2. nativeの合成8 stepと4+保存復元+4で、全tensor状態／勾配／moment／clockの
   完全一致、有限性、所有するparameter7個を確認します。
3. **観測なしnative**合成8 stepと観測付き8 stepを、全case・各stepでmodel／勾配／
   moment／clockのバイトまで一致させます。CSTAdamWの型とstepは元のままで、instanceの
   proposal箇所を一時的に観測し `finally` で復元します。Denseは通常のAdamWです。
4. 勾配1e−10/1e−8/1e−6と符号変化を含むnative fused AdamW scalar fixtureで、
   実proposalと明示的な分母再構成を既存の通常／相対norm基準で照合します。

最後の診断は実param-groupのβ／ε／LR／WDとnative m/v／clockを読みます。
報告式はmhat=m/(1−β₁^t)、sqrt(vhat)=sqrt(v/(1−β₂^t))、
denominator=sqrt(vhat)+εで、FP32 opmath係数の意味を使います。source975は
PyTorch functional AdamWの後にCST retractionを行います。ここで扱うのはその
**raw Adam proposal**の分母であり、別のradius policyの物理更新の主張ではありません。
明示的に再構成しscalar検証した値であり、native kernelの内部bufferや演算子の
ビット単位の値を直接保存したものとは扱いません。実設定の代わりにβを推測しません。

## 保存する観測と解釈の限界

初期状態0と状態1/2/4/8を保存します。状態Nは `update-N.pt['accepted']` が
所有し、完全checkpointを二重保存しません。保存するraw更新には実task B4の入力／
ラベル／logit／CE、層ごとのforward／backward tensor、更新前parameter／raw task
勾配、実際に準備されたCST勾配、raw proposal parameter／moment、受理／retraction後
状態も含めます。`updates.json` には各stepのCE、行ID、RMS／ゼロ率、出力channel別RMS分位点と現在のB4で非ゼロ／正になるchannel率、ReLUエネルギー
保持率、packed polar_s/polar_t/mu_output/mu_input統計、実m/vとbias correction、
sqrt(vhat)/εの分位点と割合<ε、proposal／受理／retractionの変位と相対更新RMSを
記録します。positionにはtorusの最小符号付き変位を使います。polarの2列は振幅／活動度を
共同で符号化するので、独立学習する物理振幅／radiusの2パラメータとは呼びません。

これらはtask batchの診断であり、以前の256行probeや全データのmetricではありません。
校正は別に256特徴行を使います。時間はtensor copy／統計を含むためthroughputの証拠では
ありません。allocated／reserved peakも診断を含むallocator値で、process全体ではありません。
raw容量は約1.5–3GBを見込み、Dense parameter／momentが大部分を占めます。
checkpointの複製を避け、生状態を1組保持します。

初期の符号別エネルギー率、実task勾配、観測したAdam更新から狭い関係を調べられますが、
8 stepでは最終精度、本質的容量限界、最小α、最適N／深さスケール則はわかりません。
学習可能性の下限はレシピ依存で、Dense品質を保つ下限とは異なります。このprobeから
長期実行や正規化の採用へ自動で進むことはありません。

## ソース、データ、資源、recoveryの境界

実行前にconsumer commit／全file manifestと全入力を凍結します。external975は
`97580582ed28c4e494bf5c2656bbe94a85c544c3`、external manifestは
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`です。
f6のtrain/development artifact
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`とmanifest
`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`を再利用します。
学習行は[1000000,1100000)、正規化もそこだけでfitします。過去の集約特徴量統計への
露出は引き続き明記します。変更していないreaderはtrain/developmentの4配列を機械的に
admitしますが、development tensor／評価は構築しません。future-confirmation、元の
HIGGS confirmation、公式testのpayloadは含めません。

direct driverは `research-inputs` の正確に6ファイルだけを受け付けます。
consumer-source-manifest.json、original-source-manifest.json、original-source.tar.gz、
train-development.npz、data-manifest.json、runtime-pin.jsonです。全source／data／
runtime identityを検証し、隔離された不変source pathでcompileし、既存のreadonlyな
job環境でtimeout付きforeground childを実行します。provision／install／queue／
background process／retryは実装しません。レビュー後、rootの資源所有者だけが指定した
RTX3090、RTX4090またはA100 MIG3g.40gbの正確なprofileで実行できます。期待するruntime-file
SHAを明示します。このdriverがColab資源を確保することはありません。


### GPU実行前の資源変更：RTX4090

ユーザーが明示的に起動・承認したRTX4090を対象に追加しました。rootが確認した
Hub profileは `gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`、GPUは
`NVIDIA GeForce RTX 4090`、CC8.9、128SM、25262096384 bytesです。profile名に
PyTorch2.6が含まれていても、そのbase環境をそのまま使いません。別の隔離job環境に
Torch2.11.0+cu126／CUDA12.6／Triton3.6.0を必須とし、rootが隔離環境の実runtimeを確認・レビュー済みで、pinのSHAは
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`
です（Python3.12.9／NumPy2.0.2／driver570.133.20）。driverは正確なhardware tupleと実runtime pinを検証し、Hubで
要求したprofile ID自体はrootのlauncherが別途確認します。GPU特性だけでprofileを
証明したとは扱いません。科学的9 case・各8更新、校正、数式、許容差、全gateは
変更しません。前のreview inventoryと変更前ソースを保存しており、GPUは未実行です。

```sh
python -B src/cst_research/experiments/higgs_depth_normalization_driver.py --consumer-commit FULL_FROZEN_SHA --runtime-sha256 ACTUAL_RUNTIME_FILE_SHA --cache-root ISOLATED_JOB_CACHE --output FRESH_ATTEMPT_OUTPUT --timeout 3600
```

上記は記号的pinを含むレシピで、実行receiptではありません。`progress.json` と
foreground logにqualification／case／更新進捗を残します。成功時も失敗時も、外側の
結果archiveからの回収用に `recovery-index.json` のSHA一覧を保持します。その実attemptを
回収して検査し、部分実行を別の意味へ付け替えたり、qualification／task状態を使って
8更新を超えて続行したりしません。source変更は新しい凍結identityとレビュー済みattemptを
必要とします。登録やCPU検査はGPU admission／完了ではありません。
