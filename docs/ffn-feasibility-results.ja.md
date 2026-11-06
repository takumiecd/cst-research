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

## 5M-tokenの機構比較（R6）

5条件すべてが4882 update・4,999,168 tokenを完了しました。LRとデータ順序の予定
horizonは共通20M tokenです。新しい実験であり、短horizonのR2の条件・観測を
置き換えません。探索条件はdevelopment CEで選び、confirmation CEは監査用に
保持して条件選択には使いません。

| FFN | Development CE | Perplexity |
| --- | ---: | ---: |
| Dense | 2.75219 | 15.68 |
| No FFN | 3.10782 | 22.37 |
| CST initial q=0 | 2.99281 | 19.94 |
| CST initial q=.25 | 3.01012 | 20.29 |
| CST initial q=.5 | 3.06285 | 21.39 |

![R6の学習曲線](assets/ffn-feasibility-20261107/mechanism-r6-learning-curves.png)

この学習量ではCSTの3条件ともFFNなしを上回りますが、Denseとの差は残ります。
初期qを増やしても最終development結果は改善しませんでした。開発用1 seedの観測で、
統計的優位や係数固定の幅スケーリングは未確認です。各update後の振幅CPUサンプリングが
wall時間に加わるため、時間軸の図は診断用であり、速度の主比較ではありません。

![R6の物理振幅・半径履歴](assets/ffn-feasibility-20261107/mechanism-r6-motion-curves.png)

物理振幅RMSだけでは移動量を見落とします。初期q0の層内RMS比を全8層で集計すると
最終値は1.076です。正規化した始終点差のRMSは.256、累積絶対移動量は1.719でした。
物理振幅a_iに対し、次のように定義します。

$$
D_{\mathrm{net}}=\sqrt{\operatorname{mean}_i(a_i(t)-a_i(0))^2},\qquad
D_{\mathrm{path}}=\sqrt{\operatorname{mean}_i\left(\sum_\tau|\Delta a_i(\tau)|\right)^2}.
$$

各層の統計量をその層の初期振幅RMSで割ってから、層間のRMSを取っています。
サンプリングした振幅の総変動を測るもので、振動周波数や無駄な移動の証明ではありません。
q0は最終でも約5.2e-8です。初期q.5では平均qが.500→.176、物理半径のinput軸up/down平均が
7.106/9.643→2.707/3.796 cellに減りました。初期q.25でも同様に縮みます。
広く始めた条件が学習中ずっと広いわけではありません。q0は振幅に依存する半径下限で、
物理的なhard minimumの.5とは異なります。弱い振幅ではbirth値4近辺の半径も残ります。

別途、終了後の読み取り専用監査で全5 checkpointのhash/counter、optimizer状態の
有限性、厳密なmodel load/cache generation、development/confirmation CEの独立再計算を
確認しました。学習済みCST全24射影で、B1 T8のY/dX/全Atom VJPもchunk化したCPU FP64
参照と一致しました。監査ではoptimizer更新や追加学習はしていません。記録した数値的
範囲を支えますが、学習差の原因が一つに決まったわけではありません。

### 固定promptの生成例

事前に決めた最初のpromptについて全文を載せています。複数生成から良い例を選んだ
ものではありません。sampling seed7000、temperature.8、top-k40、最大64新規tokenです。
固定した2つのprompt、token ID、sampling設定は各raw summaryに保存しています。
示す3モデルとも意味や文法の破綻があり、これだけで言語品質を評価しません。

**Dense**

```text
Once upon a time, there was a little girl named Lily. One day, Lily's mom saw a white red tree with a small leaves on it. Lily wanted to help the bug, but her mom said she was going to a good tree to put on the bench and put on the tree. Lily's head and went to the tree and saw that it was
```

**CST initial q=0**

```text
Once upon a time, there was a little girl named Lily. One day, Lily's mom saw a white red red car. The toy had lots of colors wanted to make a beautiful pink dress on the floor. Lily went to a garden and went outside, but she had a few more fun. She played with and played with the butterfly in the sun.
```

**No FFN**

```text
Once upon a time, there was a little girl named Lily. One day, Lily wanted to help her mommy and put on her.
Lily was playing with a little girl named Lily. She had a lot of fun with her toys and said, "Look, Lily?" Her mommy said, Lily and Lily was so happy. Lily looked around the
```

## 限定した設定比較（R7）

計画したCSTの5設定と、新しいDense・固定順序のDense・FFNなしの対照を、各976 update・
999,424 tokenで学習しました。c8、幅256、seed17、共通予定20M-tokenのLR・データhorizon、
Denseの二乗平均目標は共通です。各CST設定は先に、実FFNの両方向（downのstd.02/sqrt(8)を
含む）のnative検証、独立polynomial quadrature、普通のParameterとAtomを混在させた
optimizer参照比較、小型モデルのserialized resume、実4 blockの2 update検証を通しました。

| FFN recipe | Development CE | Perplexity |
| --- | ---: | ---: |
| Dense | 3.58196 | 35.94 |
| Dense, permuted | 3.57049 | 35.53 |
| No FFN | 4.03015 | 56.27 |
| CST reference | 3.96004 | 52.46 |
| CST birth1 | 3.93924 | 51.38 |
| CST wide q1/bound4 | 4.12193 | 61.68 |
| CST permuted | 3.95231 | 52.06 |
| CST Atom LR16 | 3.67351 | 39.39 |

![監査済み設定別学習曲線](assets/ffn-feasibility-20261107/recipe-family-r7-learning-curves.png)

この限定比較ではAtom LR16がdevelopmentで最も良い結果です。通常順序のDenseとのCE差は
.09155で、標準CSTの差.37808より小さくなりました。普通のParameterの基本LRは共通です。
倍率は中心や角度を含むAtom Parameter全体にかかり、Polar policyのouter timeも変えます。
振幅だけのLR効果や、16が最適であることの証明ではありません。共通予定20M horizonのまま、
新しい5M-tokenの追加比較に進める候補とし、公開の標準設定や3 seed確認済みとは扱いません。

広い設定の実際の初期input軸平均半径はup/downで31.601/40.325 cell、最後は12.432/16.255でした。
4はbirth側の設定で、hard upper boundではありません。この大幅に広い条件は1M時点でFFNなし
より悪く、半径を増やすだけでは改善を示していません。振幅上限とw_cも変えており、半径だけの
独立介入ではありません。

![FFN固定probeのactivation RMS](assets/ffn-feasibility-20261107/recipe-family-r7-activation-rms.png)

別の固定学習windowのprobeで、down出力RMSを4層のRMSとして集計すると、標準CSTは
.0457→.0531、Atom LR16は.0457→.1753、Denseは.0389→.2111でした。広い設定では、初期化の
期待weight energyを目標に合わせても、初期up/down probe RMSが既に.862/.380で、Denseの
.319/.039と異なります。期待二乗平均の一致だけでは、相関するタスク入力でのactivationの一致を
保証しません。機構の観測であり、独立したheldout測定や原因が一つに決まる証拠ではありません。
CSTの振幅サンプリングがwall時間に加わるため、pilotの時間軸から実用上の速度優位を主張しません。

終了後の監査で、8つのpilotすべて、学習済みCST全40射影のY/dX/全Atom VJP、厳密なpayload/
cache generationのload、development/confirmation CEの独立再計算を確認しました。追加の
2タスクupdateは使い捨てのコピー上だけで行い、メモリ内のserialize後の次stepの損失・optimizer
moment/counter・Parameterが一致しました。8条件すべてでParameterの最大差は0です。
元のcheckpointを再開したり書き戻したりはしていません。

最初の監査は、FFNなし対照で意図的に使われない`norm2`にも勾配を要求した不備で停止しました。
失敗した監査とlogは保持しています。CPUで構成とoptimizer状態から未使用集合を厳密に確認し、
修正版の監査02ではこのFFNなしParameterにだけ勾配・状態の欠如を認め、8条件をすべて再監査
しました。CST Atomや使う普通のParameterの確認を緩めたわけではありません。

## native検証の範囲と残作業

R5では両FFN方向、B2 T128、q0/.25/.5でY/dX/全Atom VJP、polynomial quadrature、
optimizer/model resumeの検証を通りました。別途、範囲を限定した2048対応経路が
RTX4090でR3の拡大12条件すべてを通り、512↔2048と128↔2048を含みます。これだけで
幅512の言語モデルの有効性や、別consumer/q設定の検証済みを意味しません。

R6とR7の比較・終了後の監査は完了しました。R7のAtom LR16を、native/task検証後の
新しい5M比較に進めます。公開の標準設定は変更していません。別のR8 consumerと明示的な
2048 operatorで、幅512の実タスクsmokeと学習済み状態監査も通りました。20 updateの
固定batch smokeはheldoutでの有効性や速度の主比較ではありません。幅128/256/512、縦横比を変えた
学習、3 seed確認、最終的な速度・メモリ検証は未完了です。

## CPUでの追加機構確認

以下は合成条件・初期化の観測であり、言語モデルの結果や標準設定の変更ではありません。
PolarのノルムQ=s²+t²、公開q=(Q−1)/3、接線方向の更新提案δu、step size hに対し、
標準の活動量更新は次式です。

$$
Q_* = \operatorname{clip}_{[1,4]}(Q+\gamma\|\delta u\|^2),\qquad
Q' = \left[1-\frac{Q_*-1}{Q_*}e^{-4\lambda h}\right]^{-1}.
$$

接線速度vを固定してδu=h vとすると、活動量の加算はh²に比例します。既存の
`time_energy`はこの加算をhで割ります。h=.0003、||v||=.15、gamma1、lambda.1、
1000 stepのCPU fixtureでは`finite_chord`のFP64でq≈6.36e-7となり、独立した
scalar漸化式と一致しました。FP32ではq≈2.65e-9、`time_energy`ではq≈.00212です。
小さい角度更新を繰り返してもqがほぼ0のままになり、FP32丸めでさらに活動量が
小さくなる場合があります。実際のAdam接線速度や学習差の原因を特定したわけでは
なく、R6学習のclock modeは変更していません。
これらは同じoperatorソースに対するローカルCPUのTorch2.13.0/Python3.11.15での
確認であり、remote CUDA updaterのFP32丸め動作を検証したものではありません。

別の限定したCPU初期化gridでは振幅上限とw_cを一緒に変えました。上限4、w_c1.2、
q1のseed17 up/down fixtureは、同じDense目標に合わせて平均半径31.53/40.54、
4096 Atom中350/10個のclipとなりました。この大幅に広い設定は座標系の条件も変え、
学習前に新たなnative qualificationが必要です。準備した候補であり、成功したモデル
ではありません。raw script/resultは証拠locator内の`polar-activity-clock-diagnostic*`と
`broad-radius-initialization-grid*`に保持します。

別の初期状態のGaussian入力確認では、各blockに独立な正規化入力512個を与えました。
4つのFFN出力共分散の和のparticipation ratioはDense184.13、CST q0で126.63、
CST q.5で62.09、隣接channel相関の平均は−.003、.419、.839でした。participation
ratioは共分散の統計量であり、代数的rankやタスク精度ではありません。blockごとに
出力channelを独立に並べ替えた共分散の和では、CSTの統計量は144.15/100.58でした。
これは共分散だけの反実仮想で、モデルを変更して学習した実験ではありません。
半径拡大で特徴の相関が強まる場合があり、全体RMSの一致だけではDense同様の特徴の
多様さを保証できません。広げる設定とAtom予算の確認に加え、birth側の半径下限を
狭める設定や固定の特徴順序も検証候補です。このfixtureで解決策が確認できたわけ
ではありません。`ffn-feature-covariance-initial*`を外部証拠として保持します。

## 来歴と保存証拠

TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`、tiktoken0.12.0の
GPT-2 encoding、story単位で重複を排除したsplitを用います。token manifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`です。

- R2/R5/R6/R7 operator：`9cb6de727ddbf5c2178fe18db6d580c9e181b926`。
- R2 consumer：`f75c1d2efe54767f6b9f65804c234924e6f6e191`。
- R7 consumer：`0e1d30279c213ad8f2ea9fc7895d3ccdea572c24`。
- R7終了receipt SHA256：`be725100159cc024f68d79895897357b143a5d6ba5949fca47a8e70127883180`。
- R7学習済み状態監査02 SHA256：`b5721388307aded0c3cd121ca8351c1e4fa5635576ce7a49bf64308b4a573cb6`。
- R7回収証拠archive SHA256：`66eb7c3df1e14e1ea87b528963b7af4625839cc7a3d6ed95959c4161a51a09e8`。
- R6 consumer：`f8de9b7f0e114e7c55035858f9544270ae381c21`。
- R6終了manifest SHA256：`2f2ff1be72d443f8a08212d5a559e177d55644fa488f3968d0361fd76f6122ed`。
- R6学習済み状態監査 SHA256：`b03d1bffa9a7f8d1017db47a7042aa3ca71aa748dbf46a1a28fd71a4552bbd67`。
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
