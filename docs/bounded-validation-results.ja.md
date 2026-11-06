# 有限のCST検証で得た観測、2026-11-06

[English](bounded-validation-results.md) | **日本語**

独立した探索的研究3件は、実行・回収・raw証拠の独立照合まで完了しました。
本命のL4・3 seed HIGGS非劣性検証は継続中です。この観測で本命を完了扱いにしたり、
その条件を変更したりしません。両研究とも確認用tailや公式testの指標は使っていません。

## 座標の学習率：序盤の改善は安定して残らなかった

モデルは28→1024→1024→2で、入力・出力は学習可能なdense、中間1層は1048 atomの
CST（公称alpha .001）です。
`q = positionの学習率 / ordinaryとcoupled polarの学習率`とします。
packed fieldはpolar_s、polar_t、output_center、input_centerであり、物理的なamplitudeと
radiusを独立したAdam学習率にした実験ではありません。基準LRは.0003です。
研究用optimizerは連続したpositionとmoment stateへ実際のposition LRを適用した後、
変更なしのpolar retractionを行います。q1はstock proposalと照合しています。
公開defaultの変更ではありません。

すべてB4、FP32、TF32 off、eagerで、同一A100 80GB PCIe MIG 3g.40gb profile
（42 SM、42,144,366,592 bytes）、Python 3.12.9、Torch 2.11.0+cu126、
Triton 3.6.0を使いました。本命HIGGSのtrain/development分割を使い、確認用tailは使いません。
seed17・3 epochのpilotでq=.1/1/10を比べました。q10はq1より選択development CEが
.00274229改善、AUCが.00232536改善し、固定済みの追加実験基準を満たしました。
q.1は悪化しました。q1/10だけをseed29/43・10 epochの有限の追加検証へ進めました。

| Seed | q10 − stockの選択CE | Accuracy差、ポイント | AUC差 |
|---|---:|---:|---:|
| 29 | +.0006189942 | −.2580 | −.0008525581 |
| 43 | −.0000753999 | −.1920 | +.0004390437 |
| 2 seed平均 | +.0002717972 | −.2250 | −.0002067572 |

各armはdevelopment CE最小、同値なら早いepochのcheckpointを選びます。
これらの対応するdevelopment差は記述的な結果であり、非劣性判定や信頼区間の主張では
ありません。序盤の利点から、最終精度の一貫した改善にはつながりませんでした。
stock q1を維持し、追加の比率・seed探索は自動では行いません。

CPU scalar演算、実CUDAのmoment・clock、q1とstockのbyte一致、8 updateと
4 update時点で保存・再開する照合はPASSしました。先行するqualificationは、oracleが
実際に保存されたFP32のbeta係数ではなくdouble精度の係数を使ったため、学習前に失敗しました。
失敗例と修正oracleの両方を保持しています。学習の数式と許容差は変更していません。

## 深さと層位置：圧縮できる程度は置換位置に依存する

28→1024→1024→1024→1024→2を使い、biasなしの正方隠れ層3個、ReLU 4個、
学習可能なdense入出力を用意しました。Dense、early/middle/lateの1層だけCST、
正方層3個すべてCSTを、公称alpha .0001/.001/.01（置換層あたり104/1048/10485 atom）
で比較しました。全13条件がseed17、3 epoch、LR .0003、stock optimizer q1、
B4 FP32/TF32 off/eagerで完了しています。

実機はRTX3090、CC8.6、82 SM、25,296,044,032 bytes、driver580.95.05、
Python3.12.9、Torch2.11.0+cu126、Triton3.6.0です。
train[1000000,1100000)、development[1100000,1150000)、別保存の未使用future
[1150000,1200000)は0始まりの半開区間です。今回の正規化は今回のtrain特徴だけで
fitしました。過去の集約特徴統計には今回の行も含まれていたため、全履歴を通じて未使用の
特徴とは主張しません。futureのラベルはモデル選択・評価に使っていません。

| 置換層 | alpha .0001: CE / AUC | alpha .001: CE / AUC | alpha .01: CE / AUC |
|---|---:|---:|---:|
| Early | .5627934 / .7783969 | .5484946 / .7982233 | .5482507 / .7994730 |
| Middle | .5485868 / .7982039 | .5429138 / .8024335 | .5433032 / .8029886 |
| Late | .5450923 / .8013952 | .5437870 / .8026811 | .5427784 / .8035260 |
| 全3層 | .6913939 / .5000000 | .5524291 / .7916114 | .5487499 / .7977808 |

Denseの選択CE/AUCは.5483097/.8014886でした。これらは3 epoch、1 seedでの
development選択値です。最小alphaや深さに対する単調な法則を確定するものではありません。
全3層alpha .0001は定数分類器のままでした。.001/.01では学習でき、まだ改善途中でした。
alpha .0001では、middle/late 1層の置換なら学習できても、3層すべての置換では
この予算内に学習できませんでした。

1層CSTの同期済み学習時間は3 epoch全体で約102–108秒、Denseは46.292秒です。
allocated peakは約87.8–88.6MB、Denseは104,513,024 bytesでした。
allocatorのpeakであり、process全体のメモリではありません。CUDA Graphは使っていません。
準備・評価・診断・checkpoint I/Oは学習時間の外です。最初のCST条件はcold compileのため
準備が約130秒、以降の1層CST条件は4.10–6.41秒でした。
密度の効果やL4での速度比較とは解釈しません。

独立した全networkのCPU FP64 Y/dX/全gradient照合、8 updateと4 update保存・再開の
照合はPASSしました。別に行ったcotangent増幅時の成分ごとのstress診断は、3密度すべてで
失敗しています。この負の結果は保持します。fieldとpacked columnごとのscale正規化判定は、
zero/sign-flip負例とmaterial gradient coverageも含めPASSしました。
alpha .0001へのこの判定の適用は事後的です。同じ判定を.001/.01のprobe前に固定しました。
元の増幅・成分ごとの失敗をPASSと言い換えません。

train probeで最初の置換層の初期出力RMSはCSTが約.0005、Denseが.2428でした。
conditioningの仮説を立てる根拠ですが、表現力・初期化・最適化・activation maskの影響を
切り分けた結果ではありません。別に固定した有限のgain実験の結果を以下に記録します。
各siteの範囲と最適化scaleは変わりますが、制約のない出力重みにより全体の理想実数関数クラスは
保存されます。初期化だけの因果実験ではありません。

## 固定gain：最小密度の全層置換でも学習するが、Denseには届かない

上記A100 MIG/runtimeで、新しいDense対照、stock全3層CST、固定gain全3層CSTを、
alpha .0001、seed17、3 epoch、LR .0003で比較しました。深層研究と完全に同じ
train/development行とstock q1 optimizerを使います。Dense基準もこのA100 cohortの
実測値を使い、RTX3090の値で置き換えません。密度はstock深層pilotの停止を観測した後、
gainの結果を見る前に選んでおり、探索的な選択であることを明示します。

各置換正方層に固定の正の出力gainを入れ、最初の256 train特徴で対応するDense層の
初期RMSへ合わせます。逐次的なCPU FP64・公開`dense_weight`合成で較正し、
ラベル・development行は使わず、clippingや目標値の探索もしません。
gainは学習しないFP32 bufferとして全学習中固定します。実際に実行したgainは
521.2747192382812、258.95751953125、945.0846557617188です。
保存したstock/gainの学習可能な初期値はbyte一致し、stock tableは先行する検証済み
深層研究とも一致しています。

| Arm | 選択development CE | Accuracy | AUC |
|---|---:|---:|---:|
| Dense | .5476290584 | .7199800014 | .7952768061 |
| Stock全3層CST | .6913939118 | .5295999646 | .5000000000 |
| 固定gain全3層CST | .5910087228 | .6868199706 | .7539428246 |

すべてepoch2のcheckpointを選択しています。gainはstockよりCEが.1003851891改善、
accuracyが15.7220ポイント改善、AUCが.2539428246改善しました。一方、Denseに対しては
CEが.0433796644、accuracyが3.3160ポイント、AUCが.0413339815劣ります。
この予算内で観測した定数分類器の停止を解消する介入でしたが、精度差は大きく残りました。
表現力の限界の切り分け、最小alphaの確定、gain較正のdefault採用を支持するものでは
ありません。追加のgain epoch・seed実験は自動では行いません。

全52 archive member、実source/data/runtime、較正、ordinary/stock/gain初期値、
通常のY/dX/全gradientとfield/column norm判定、8 update・4 update再開、
毎epochのparameter/gradient/moment/clock/CST optimizer store、選択checkpointの
train100k/development50k予測について、rawの独立照合はPASSしました。
Denseを含む3 armすべてで、増幅時の成分ごとの診断は失敗し、FAILED表示を維持しています。
そのfield/column相対L2誤差は固定済み3e-4のnorm基準内でした。確認用/testは使っていません。

readerだけの失敗2件も保持しています。local Torch2.13/Macで新しく構築したCST初期値が
実Torch2.11の初期値とbyte一致しなかったこと（最大差9e-9未満）と、100k train予測を
評価するhelperが50k行を前提にしていたことです。最終readerは保存した実初期値を
先行stock実測へ結び、そのtensorから較正を独立に再計算し、両方の行数に同じ二値指標の
数式を適用します。producerの数式、許容差、raw結果は変更していません。
runtimeをまたぐ初期化のbyte一致は主張しません。

## 正gainでも全体の理想関数クラスは変わらない

実際のbiasなし正方写像3個、ReLU、固定された正gainでは、正の斉次性から可逆な再parameter化を得ます。

\[
F_g(\theta)=F_{\rm stock}(T_g\theta),\qquad
W'_{\rm out}=G W_{\rm out},\quad b'_{\rm out}=b_{\rm out},
\qquad G=\prod_{j=1}^{3}g_j\approx127575095.34510244.
\]

この写像でhidden parameterは全て不変です。出力重みが制約されないため、siteごとの振幅範囲を
rescaleしても全体の理想実数関数クラスは拡大しません。部分的改善を、表現可能な関数クラスの違いや
不変の初期ゼロ列だけでは説明できません。同一保存tensorでも開始関数は異なり、勾配、Adam epsilon、
相対update軌道も異なります。decoupled weight decayの乗算自体は定数rescaleと可換なので、
単独の因果的役割を確定していません。有限FP32でのbyte同値は主張しません。

[完全な導出](positive-gain-equivalence.ja.md)は実初期保存状態と未変更69/975 sourceに結び付きます。
固定した合成linspaceとzero4×28入力による公開CPU-FP64 comparatorで、stockの出力重みを
変換した場合の最大絶対差は8.67e-19と0でした。dataset/label、タスク評価、GPU操作、新qualificationは
使っていません。解析report
`higgs-depth-gain-20261106/stats-positive-gain-equivalence-01/analysis.json` のSHAは
`a90fa2323c706095b4975e8b81bc29c745322ae13a56fbb467dd8c8f8c05f5c7` です。
既存のseed17・3 epochという限界と診断の失敗は変わりません。

## Sourceと証拠の識別情報

数値runtimeはすべて変更なしのtorchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`、external manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`です。
測定した研究consumerは次のcommitに固定したままです。

| 研究 | 測定consumer | 全manifest | PASSした集約audit |
|---|---|---|---|
| Coordinate LR | `83cb77fe84e91ae296c24f2144bd56ca9398f404` | `7ce3d76f3ea3e09084e5c2dcb37ade7581eebffbf0f22c1b74dae08b467511bd` | `b5be7229474a29e182b7978ba5270e665a941f34870d0ac9b27fcecf2ca45532` |
| Depth13 | `f6d08ad76001a69070b0965cefe7614f192ddf3f` | `aec42b259f37bd3922a3763f4c4f5e3af9b9a926263b10128861d09b18295a15` | `95b1bf35095028623900732dfd60eca41cf80d0a896a891dc3b0f2ab75050dd3` |
| Depth gain3 | `69d7f0636ce47db892096d7b7aacc2c2cb731b8b` | `b7b6428e80bbf18fc0cff22af2388e6818c3c9c28ad9221c98c352123666d193` | `4bfb2d41ed402fcce01b05648d52d601cee3fde820300edd6f1a15d8f80cbb7d` |

Coordinate sourceは[PR7](https://github.com/takumiecd/cst-research/pull/7)、
Depth sourceは[PR8](https://github.com/takumiecd/cst-research/pull/8)、
Gain sourceは[PR9](https://github.com/takumiecd/cst-research/pull/9)です。
再現には、この文書のcommitとは別に、測定commitのprotocolと固定inputを使います。
新しいoutput/stateを使い、対応する資源手順に従ってください。
CPU testやregistry validationだけではGPU結果の再現にはなりません。

[外部証拠の規約](reproducibility.ja.md)に従って次を保持しています。

- `higgs-coordinate-lr-20261105/stats-actual-a100-followup-seed29-01`と
  `stats-actual-a100-followup-seed43-01`はrawの独立auditです。
  `stats-verified-two-followups-02`には全curve CSV、epochを揃えた平均、PNG/PDFの
  重ね描きとSHAで結んだauditがあります。集約reportはPASS済みauditを再利用します。
- `higgs-depth-20261105/stats-verified-all13-01`には全13条件のcurve、case/activation
  summary、密度・位置・資源のPNG/PDF、auditがあります。先行raw auditの来歴と
  失敗したstress artifactもすべて保持しています。
- `higgs-depth-gain-20261106/stats-actual-a100-gain-03`には全raw audit、全curve、
  activation/gradient summary、同じcheckpointのtrain/developmentと資源のplotがあります。
  `presentation-final-01`は適用範囲を表示した学習PNG/PDFを、そのauditと保持した
  reader失敗の来歴へ結びます。result archive SHAは
  `0ed924e6511005a8a780764392dcd6d6d04f8cc48b25c7e725dfbb78da4cd92f`です。

Coordinate result archive SHAはseed29が
`6300c1efd82f4804a2b4de7ea7aea654f55f574d6a8e7bcc1520697e0672ba67`、
seed43が`653471726c8c714d6d873f743d33a82666b9a3a07cad46cf3227d8a45121ac8b`です。
Depth全3層のarchive SHAは密度.0001/.001/.01の順に
`c2db2ee4f8565929664b6989fefb4758ff0b13056f366c3362f08b7bdf4531ae`、
`31c8cb10aae104ded6bb4dab56cbf5615b8824cefd67799191c119f23bf5c10b`、
`53b19026996fe6ecd1520743dc8dafc016a3dd907bdb038ca01696cdfbc20d6a`です。
これらの結果はHub jobのreceiptを使います。
source、実運用deployment、raw state、選択、派生reportの識別情報は分けて保持しています。
