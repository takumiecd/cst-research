# HIGGS depth: 固定出力gainによる介入

[English (canonical)](higgs-depth-gain.md) | 日本語

**PLANNED。** 独立したconsumerとして、範囲を限定した探索実験を準備する。
CPU確認やregistry検証はGPU適格性や学習結果を意味しない。元のdepthソース
`f6d08ad76001a69070b0965cefe7614f192ddf3f`、13条件、正式な単一中間層CST/DST研究は変更しない。

## 問いと適応的な条件選択

stock depth探索はDense、単一位置置換9条件、全中間層置換3条件を1seed・3epochで比較した。
development曲線では全層α=0.0001がAUC 0.5のまま、α=0.001と0.01は学習した。
そこで低密度全層モデルに固定したforwardスケール調整が役立つかを調べる。
最初のソース準備はα=0.001で未実行だったが、観測済みstock曲線に基づき
**gain実験の実行前にα=0.0001へ変更**した。適応的な探索条件であり、独立した確認実験ではない。
先のソース準備はignored証拠として保持する。

同じ許可済みGPU/runtimeで、一つのprocess内に次の3条件だけを逐次実行する。

| 条件 | 3つの正方中間層 | 固定gain |
|---|---|---|
| `dense` | Dense | 1, 1, 1 |
| `stock-all` | CST、α=0.0001 | 1, 1, 1 |
| `gain-all` | 同じCST学習対象初期値 | 特徴量だけで較正したgain |

構成は28→1024→1024→1024→1024→2、ReLU 4個、正方層biasなし、入出力Dense、全parameter学習。
各CST層はA=floor(1024²/10000)=104 atoms。層ごとの元の独立seedを使用する。
stock/gainの**全named trainable tensorをbyte一致**させ、通常の入出力tensorもDenseと一致させる。
gainは1つのvector buffer内の3要素の非学習FP32値で、3条件とも各正方層で単位gainを含む乗算を実行する。

全条件seed17、3 full epoch、batch4、FP32、TF32 off、eager、stock CSTAdamW
（DenseはAdamW）、LR0.0003、position multiplier1、通常weight decay0.01、
atom raw decay0、論理amplitude decay0.01。gain grid、LR sweep、適応的追加epoch、
自動追試、追加seedはない。外部public `torchcst`
`97580582ed28c4e494bf5c2656bbe94a85c544c3` は変更しない。

## 一度だけの較正

較正に使うのは、正規化済み新trainingの**先頭256特徴量行**、raw index
[1000000,1000256)だけ。gain計算にlabel/development/confirmation/testは使わない。
未学習copyとpublic CPU FP64 `dense_weight()` 合成から値を求める。
Dense target RMSはDense自身の先行Dense/ReLU経路で、candidate RMSは
すでに較正した先行CST/ReLU経路で求める。

正方層jを順番に処理する。

\[
t_j=\operatorname{RMS}_{X_{256}}(z_j^{\mathrm{dense}}),\qquad
r_j=\operatorname{RMS}_{X_{256}}(C_j(h_{j-1}^{\mathrm{gain}})),\qquad
g_j=\operatorname{FP32}(t_j/r_j),\qquad
h_j^{\mathrm{gain}}=\operatorname{ReLU}(g_j C_j(h_{j-1}^{\mathrm{gain}})).
\]

全target/observed RMSが有限かつ正で、丸め前ratioと保存FP32 gainの両方が
[1e-6,1e6]の範囲内でなければならない。target 0、observed 0、非有限、範囲外は
**inconclusiveとして学習しない**。clip、代替target、自動調整は行わない。
raw特徴量行列、hash/行範囲のbinding、初期model、gain、逐次RMS transcript、
学習対象hash不変を保存する。optimizer更新とcheckpoint復元を通じてbufferを固定する。

これは**forwardスケール/conditioningへの介入**で、初期化だけの因果試験ではない。
元のamplitude制約が|a|≤1なら、実効制約は|g_j a|≤g_jになる。
勾配スケールと、変更しないraw amplitude decayから実効関数への対応も変わる。
元の有界weight familyを保存せず、初期化だけが原因だったとの証明にはならない。
正gainがReLU符号を保つのは同じpre-gain入力に対してだけで、先行層が異なれば
学習後の符号パターン一致を意味しない。
同じparameterならbiasなし正方層/ReLU経路は正の斉次性を持ち、Dense head前の
初期出力は3gainの積でscaleされ、最後の出力biasは変わらない。
逐次合成に加えて、独立したCPUのproduct-gain Y/全勾配代数チェックも行う。

## データと実行前適格性

新depth用train/development payloadだけを再利用する。
SHA256 `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`、
manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`。
training raw [1000000,1100000)、development [1100000,1150000)。
正規化はtraining100kだけでFP64 fitting、FP32適用。
過去の集約特徴量統計にはこれらの行が含まれていたことを開示し、
特徴量やlabelが普遍的に未見だったとは主張しない。
将来のdepth confirmation [1150000,1200000)と公式test [10500000,11000000)は
開かず、pilot payloadにも含めない。[depth protocol](higgs-depth-position.ja.md)も参照。

**どの条件も学習を始める前に**、全3条件の新しい実機qualificationを通す。

- 全network Y/dX/全7parameter勾配を、独立したpublic CPU FP64 dense-weight合成と比較。
  upstreamは実際のFP32値をそのままFP64へ拡張する。
- 通常scaleの元elementwise tolerance 3e-5+3e-4|reference|を必須とする。
- 通常/1e12増幅upstreamの両方で、全勾配fieldと全packed atom列について
  relative L2とmax-error/reference-maxを≤3e-4とする。epsilonは使わず、
  referenceが厳密0ならactualも厳密0。微小列を含むzero/signflip置換を明示的に確認する。
- 増幅dXと各CST fieldはmaterialな非0referenceとactual信号を必要とする。
  upstream scaleでforward出力が変わらないことも確認する。判定前にraw tensorを保存し、
  増幅componentwise結果は**診断だけ**とする。
- stock optimizer 8更新と4+復元+4の全数値tensor byte、moment、buffer、clockを一致させる。
  7独立stateの完全所有、有限値、gain固定を確認する。
  AtomStateのlayout世代metadataはloadで進むが、数値tensor一致は必須。

以前のstock全層増幅componentwise stressは**失敗したまま保持**する。
後のfield/column norm基準はD10000にはpost hoc、他2密度ではprobe前に固定した。
本準備はこの限定されたnorm/coverage基準を使い、任意cotangentの成分ごとの精度を主張しない。
新qualificationが失敗したら本pilotを止め、toleranceを暗黙に変更しない。

## 出力と再現

entrypointは `cst_research.experiments.higgs_depth_gain`。
source/input/運用envelope/厳密GPU/runtimeを固定してreviewしてからだけ実行する。

```bash
PYTHONPATH="$CONSUMER/src:$TORCHCST_SOURCE/src:$TORCHCST_SOURCE" \
python -B -m cst_research.experiments.higgs_depth_gain \
  --consumer-commit "$CONSUMER_COMMIT" --consumer-manifest "$CONSUMER_MANIFEST" \
  --external-source "$TORCHCST_SOURCE" --external-manifest "$EXTERNAL_MANIFEST" \
  --data "$DATA" --data-manifest "$DATA_MANIFEST" --output "$FRESH_OUTPUT"
```

consumerは定めた3条件/予算だけを同一GPUで逐次実行し、resource確保は行わない。
repository/resourceルールに従う。運用receipt、厳密source/input manifest、結果archive SHAを保持し、
raw artifactはexternal evidence locatorのGit外へ保存する。

毎epochのraw model/optimizer/gradients、clock、order hash、activation/gradient診断、
同期した完全step学習秒、allocated/reserved allocator peakを保存する。
gain-aware activation traceはfull forwardと同じ位置で固定乗数を適用する。
CPU確認で最終trace出力および統計とfull forwardの一致を要求する。
full-train評価、development評価、診断（initial診断/development評価も含む）は
同期付きで個別の秒数を保存する。
peakは常駐data/evaluationも含みprocess memoryではない。
preparation/calibration/全条件qualificationは別に記録する。
evaluation/診断/checkpoint I/Oは学習秒から除外する。
オンラインtrain指標は変化中のweightを表すため、別にfull-training100kとdevelopment50kを
**同じpost-epoch checkpoint**で評価する。各条件は最初のmin-development CEを選び、
選択済みtrain/dev予測、final checkpoint、全曲線を残す。
曲線からの追加選択、最小alpha推論、正式NI、速度採用、GPU間比較は行わない。
改善してもこの複合介入をさらに調べる根拠にとどまり、3epochで否定的でも長期効果は否定できない。
