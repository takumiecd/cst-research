# 有限のCST検証で得た観測、2026-11-06

[English](bounded-validation-results.md) | **日本語**

独立した探索的研究2件は、実行・回収・raw証拠の独立照合まで完了しました。
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
切り分けた結果ではありません。別に固定した有限のgain実験を実行中で、この文書では
結果を主張していません。出力gainは実効的なweightの範囲、gradientと正則化のscaleも
変えるため、純粋に初期化だけの因果実験にはなりません。

## Sourceと証拠の識別情報

数値runtimeはすべて変更なしのtorchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`、external manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`です。
測定した研究consumerは次のcommitに固定したままです。

| 研究 | 測定consumer | 全manifest | PASSした集約audit |
|---|---|---|---|
| Coordinate LR | `83cb77fe84e91ae296c24f2144bd56ca9398f404` | `7ce3d76f3ea3e09084e5c2dcb37ade7581eebffbf0f22c1b74dae08b467511bd` | `b5be7229474a29e182b7978ba5270e665a941f34870d0ac9b27fcecf2ca45532` |
| Depth13 | `f6d08ad76001a69070b0965cefe7614f192ddf3f` | `aec42b259f37bd3922a3763f4c4f5e3af9b9a926263b10128861d09b18295a15` | `95b1bf35095028623900732dfd60eca41cf80d0a896a891dc3b0f2ab75050dd3` |

Coordinate sourceは[PR7](https://github.com/takumiecd/cst-research/pull/7)、
Depth sourceは[PR8](https://github.com/takumiecd/cst-research/pull/8)です。
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

Coordinate result archive SHAはseed29が
`6300c1efd82f4804a2b4de7ea7aea654f55f574d6a8e7bcc1520697e0672ba67`、
seed43が`653471726c8c714d6d873f743d33a82666b9a3a07cad46cf3227d8a45121ac8b`です。
Depth全3層のarchive SHAは密度.0001/.001/.01の順に
`c2db2ee4f8565929664b6989fefb4758ff0b13056f366c3362f08b7bdf4531ae`、
`31c8cb10aae104ded6bb4dab56cbf5615b8824cefd67799191c119f23bf5c10b`、
`53b19026996fe6ecd1520743dc8dafc016a3dd907bdb038ca01696cdfbc20d6a`です。
これらの結果はHub jobのreceiptを使います。
source、実運用deployment、raw state、選択、派生reportの識別情報は分けて保持しています。
