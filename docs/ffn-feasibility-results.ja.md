# FFNの有効性確認：初期観測

[English (canonical)](ffn-feasibility-results.md) | 日本語

開発用seed17の観測であり、3 seedの確認実験やスケーリング則の結果ではありません。
[検証計画](ffn-feasibility.ja.md)は進行中です。幅256・4 blockのcausal Transformerの
FFN全8射影に各4096 Atom、`A=ceil(8 sqrt(d_in d_out))`を用います。Attention、
embedding、重み共有する語彙出力はDenseのままです。FFNは256→1024→256、context128です。

## 最初の3条件pilot（R2）

各条件976 update・999424 tokenで学習しました。データ順序とcosine LRもこの短い
horizonです。Dense目標stdは.02、残差出力射影は.02/sqrt(8)、CSTは公開の二乗平均
初期化とq0を用います。条件間で共有するParameterは同じ初期化乱数系列です。
FP32、TF32off、RTX4090、Torch2.11.0+cu126、updateあたり1024 tokenです。

| FFN | Development CE | Perplexity | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| Dense | 3.72727 | 41.57 | 482.50 |
| CST | 4.17978 | 65.35 | 687.28 |
| なし | 4.23073 | 68.77 | 440.39 |

![初期pilotの学習曲線](assets/ffn-feasibility-20261107/pilot-r2-learning-curves.png)

CSTは学習しますが、この学習量ではDenseよりFFNなしに近い結果です。FFNの論理
Parameter数はDenseの6.48%ですが、実測GPU peakは大きくなっています。このrunは
実用上の速度・メモリ優位を示していません。native qualificationと別の学習済み
checkpoint監査で、Y、dX、全Atom座標VJP、optimizer moment/counterの有限性、
heldout CEの再計算を確認しました。記録した範囲での数値的な正しさを支えますが、
学習差の原因を説明するものではありません。

初期FFN出力RMSはDenseに近い値でした。CSTのdown射影RMSは.04–.05付近に留まり、
Denseは.12–.17へ増えました。その後の予定20M-token horizonのq0実験はownerの指示で
停止しました。最後の定期checkpointは2800 update・2867200 tokenです。未保存の
末尾を復元したとせず、このCEを短horizonのDense結果と直接比較しません。

## 半径の比較（R5）

q0/.25/.5を各512 update・524288 token、新しいrunで学習しました。LRとデータ順序の
予定horizonは共通20M tokenです。qごとに二乗平均を合わせ直しています。q変更は
物理半径に加えて振幅、空間相関、clippingも変えるため、半径だけの独立介入ではありません。

| 初期q | Development CE | update時間中央値 ms | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| 0 | 4.40716 | 419.35 | 687.28 |
| .25 | 4.39447 | 396.89 | 687.28 |
| .5 | 4.41436 | 222.69 | 687.28 |

![半径別の学習曲線](assets/ffn-feasibility-20261107/radius-r5-learning-curves.png)

損失曲線はほぼ重なり、この初期観測では半径拡大による大きな改善は見えません。
時間はq.5/0/.25の順に実行した1つの逐次jobでの同期updateの観測で、順序を変えた
反復比較はありません。数学的なworkloadの変更をkernelだけの高速化とはせず、
これらの観測を外挿しません。

別途seed17の`make_cst` CPU fixtureでは平均半径up/downがq0で2.49/3.21、q.25で
4.35/5.58、q.5で7.14/9.71でした。実モデルの各層は名前別seedなので、個別の
初期化reportを保持しています。q1のup fixtureは振幅上限1で目標二乗平均に届かず、
q1の学習は実行していません。拡大時に現在の振幅・半径の結合が制約になります。

停止したq0 checkpointの同じAtom IDを照合すると振幅RMSはup .2280→.2359、
down .1171→.1237でした。最終振幅上位10%の平均半径は.914/1.637、下位10%は
3.967/3.992で、qはほぼ0です。弱いAtomが広く、強いAtomが狭い状態が共存し、
全体の平均半径だけではこの違いを見落とします。始点と終点だけから振幅の累積
往復量は判断できません。

## native検証の範囲と残作業

R5では両FFN方向、B2 T128、q0/.25/.5でY/dX/全Atom VJP、polynomial quadrature、
optimizer/model resumeの検証を通りました。別途、範囲を限定した2048対応経路が
RTX4090でR3の拡大12条件すべてを通り、512↔2048と128↔2048を含みます。これだけで
幅512の言語モデルの有効性や、別consumer/q設定の検証済みを意味しません。

新しい5M-tokenのDense・FFNなし・CST q0/.25/.5比較でも予定horizonは20Mのままです。
CSTは各update後の振幅を読み、累積移動量と始終点の差、半径/qの履歴を保持します。
診断付きrunを速度・メモリの主比較には用いません。幅128/256/512、縦横比を変えた
学習、3 seed確認、最終的な速度・メモリ検証は未完了です。

## CPUでの追加機構確認

以下は合成条件・初期化の観測であり、言語モデルの結果や標準設定の変更ではありません。
PolarのノルムQ=s²+t²、公開q=(Q−1)/3、接線方向の更新提案δu、step size hに対し、
標準の活動量更新は次式です。

\[
Q_* = \operatorname{clip}_{[1,4]}(Q+\gamma\|\delta u\|^2),\qquad
Q' = \left[1-\frac{Q_*-1}{Q_*}e^{-4\lambda h}\right]^{-1}.
\]

接線速度vを固定してδu=h vとすると、活動量の加算はh²に比例します。既存の
`time_energy`はこの加算をhで割ります。h=.0003、||v||=.15、gamma1、lambda.1、
1000 stepのCPU fixtureでは`finite_chord`のFP64でq≈6.36e-7となり、独立した
scalar漸化式と一致しました。FP32ではq≈2.65e-9、`time_energy`ではq≈.00212です。
小さい角度更新を繰り返してもqがほぼ0のままになり、FP32丸めでさらに活動量が
小さくなる場合があります。実際のAdam接線速度や学習差の原因を特定したわけでは
なく、R6学習のclock modeは変更していません。

別の限定したCPU初期化gridでは振幅上限とw_cを一緒に変えました。上限4、w_c1.2、
q1のseed17 up/down fixtureは、同じDense目標に合わせて平均半径31.53/40.54、
4096 Atom中350/10個のclipとなりました。この大幅に広い設定は座標系の条件も変え、
学習前に新たなnative qualificationが必要です。準備した候補であり、成功したモデル
ではありません。raw script/resultは証拠locator内の`polar-activity-clock-diagnostic*`と
`broad-radius-initialization-grid*`に保持します。

## 来歴と保存証拠

TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`、tiktoken0.12.0の
GPT-2 encoding、story単位で重複を排除したsplitを用います。token manifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`です。

- R2/R5 operator：`9cb6de727ddbf5c2178fe18db6d580c9e181b926`。
- R2 consumer：`f75c1d2efe54767f6b9f65804c234924e6f6e191`。
- R5 consumer：`09e15b759fd796ad02c436e3580f1a223e098a92`。
- R5 native receipt SHA256：`4268f0d8ee42f4febd036e1f9846ecdd565231edaca330ef33fd5f4ddb49222b`。
- R5終了manifest SHA256：`17d42c7f9722373da24ab23ab3c4e3244e700ab38cb84791f4a64d88309e2b40`。
- 拡大operator：`2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`、
  consumer：`036d0c5beb4a6f675ce1dbeafd4224e6c270f342`。
- 拡大native receipt SHA256：`419ab7750182bca2302d7152c6fc01c1a961efe69f19bf5bb22355f3aa299ce6`。

raw receipt、各条件summary/curves、独立監査結果、描画scriptは
`$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106`から辿ります。R5のraw JSONと
curveは終了manifestとhashを照合済みで、学習済みR5 checkpointの独立監査は別工程です。
大きなcheckpointはhashとともにGPU hostに保持します。ここでのPNG/SVGは派生図で、
raw証拠ではありません。
