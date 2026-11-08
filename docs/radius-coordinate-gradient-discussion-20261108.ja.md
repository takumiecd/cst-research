# 半径・振幅・座標の勾配

[English（正本）](radius-coordinate-gradient-discussion-20261108.md) | 日本語

独立半径FFN予備比較中の議論を数式として残す。以下は恒等式・提案であり、
実装済み更新則や実験結果ではない。実行中の比較は[固定protocol](ffn-independent-radius-20261108.ja.md)
の3条件のままとする。

1つのAtomを `W_j=a_j K(r_j,mu_j)`、行列勾配を
`G=partial L/partial W`、内積をFrobenius内積とすると、

```math
g_a=\langle G,K\rangle,\qquad
g_r=a\langle G,\partial_rK\rangle,\qquad
g_\mu=a\langle G,\partial_\mu K\rangle.
```

半径と座標の勾配はkernel形状を通して既に関係する。両方とも振幅に比例し、
factorの微分が有限なら振幅が厳密に0のとき両方0になる。一方、振幅勾配は
非ゼロになり得る。ただし、この恒等式だけで個別の学習停滞の原因は特定できず、
「座標が動くなら半径を広げる」という制御則も導けない。
cell-massの微分はrouting/level所属ごとの区分的なものなので、分岐内部と
切替点を別々に検証する。内部の式から大域的な滑らかさを仮定しない。

## 半径まで微分するPolar拡張

`a=a(s,t)`、`r=L(a)+(U(a)-L(a))p(s,t)` なら、

```math
r_a=L'(a)+(U'(a)-L'(a))p,\qquad
\partial_s r=r_a\partial_s a+(U-L)\partial_s p
```

と書ける。tも同様。半径に損失勾配を通す場合は、

```math
\partial_s\mathcal L=g_a\partial_s a+g_r\partial_s r,\qquad
\partial_t\mathcal L=g_a\partial_t a+g_r\partial_t r.
```

現在のPolarは損失に対して派生半径をdetachしている。上記は**別の微分契約**であり、
既存backwardを黙って置き換えず、独立したparameterization宣言と検証が必要。
共有座標表現を保ち、L/Uの傾きも含める。半径項の一部だけを加えるのは、この
連鎖律とは異なる。今回の独立半径比較は物理値の直接学習を先に試すものであり、
このPolar拡張の実験ではない。

上記の線形envelopeは構成の一案であり、現在の実装は対数補間である。
厳密な算術では `r=L(a)^(1-q) U(a)^q` に相当し、内部の微分は

```math
\partial_s r=r\left[\left((1-q)\frac{L'}L+q\frac{U'}U\right)
\partial_s a+\log(U/L)\partial_s q\right].
```

現在の写像は `rho=sqrt(s²+t²)`、`a=a_max s/rho`、
`q=clamp((rho²-1)/3,0,1)`。特異点・clamp境界を除けば、
`a_s=a_max t²/rho³`、`a_t=-a_max st/rho³`、`1<rho<2` の内部で
`q_s=2s/3`、`q_t=2t/3`。qのclamp区間外の微分は0。
端点の扱いは内部の式から仮定せず、明示的に検査する必要がある。

更新則の問題もある。現在のPolar更新はoptimizerの生の提案を `(s,t)` の
円の接線へ射影し、その後に別のactivity/正則化則で半径方向の状態を進める。
したがって半径の損失微分をbackwardへ追加するだけでは、**qを通常の勾配降下で
学習する動作にはならない**。半径勾配を通すPolar実験では、微分と更新の両方を
宣言し、Parameter・moment・復元の検証も必要になる。

## 移動量は別の状態・更新則の提案

平行移動不変な周期格子では絶対座標で半径を変えるべきではない。
移動量を使うなら、勾配の生の大きさではなくoptimizer更新後の実際の周期的な
中心変位を使い、対応する半径・格子間隔で尺度を揃える。
共有半径の場合の無次元量と有界応答の一例は、

```math
D_t^2=\|\Delta\mu_t/r_t\|^2,\qquad
f(D_t)=\frac{D_t^2}{D_t^2+\tau^2}\quad(\tau>0).
```

`f(0)=0` で、大きい移動に対して1に近づく。時間平滑化にはoptimizer状態、
時間尺度、checkpoint要件が加わる。直前の更新を使う式は更新則であり、
その更新全体に通常の連鎖律を通す静的forwardの再parameterizationではない。
移動停止は収束だけでなく、弱い振幅、狭いkernelの平坦領域、小さいschedule LRでも
起きるため、それだけでは半径を縮めるべきとは判定できない。

envelopeの `p=f(D)` とすれば停止時に `r→L(a)` となり、常に大域的な
`r_min` に落ちるわけではない。`a→0` で両境界が `r_max` に合流するなら、
停止時でも広い半径を保てる。振幅のenvelopeを優先することで、2つの端点条件の
見かけ上の矛盾を解消できる。L/Uの厳密な形、履歴平均、脱出機構、損失のg_rとの
組合せは未決定。議論を理由に新しい移動量制御の実験は開始せず、現在の比較を見て選ぶ。

## CPU連鎖律検査と次の比較案

独立半径比較の完了後、q0.1/0.25/0.4/0.6/0.8/0.9の内部点を持つ6Atomで、
CPU FP64の完全なs/t半径連鎖律を検査した。手計算の式と自動微分の最大絶対誤差は
4.44089e-16、2種類の中央差分でも全4成分にわたり2.30e-9以内で一致した。
半径微分を通す勾配には正負両方の半径方向成分があり、detach版の半径方向は数値的に0。
半径写像のq0/q1では内側からの片側微分も検査し、clamp角の中央微分は主張していない。

既存の値計算helperはq>=1でUを直接選ぶため、q1での自動微分は0だった。
一方、検査した点の内側からの数学的な傾きは非ゼロ。現在の半径detach契約では
その傾きを使っていないが、新しい半径微分を通す契約では端点Jacobianを明示的に
実装する必要がある。既存helperのdetachを外すだけでは、その微分にはならない。

検査したsource依存を使い、
`python -m cst_research.experiments.polar_radius_chain_rule --output FRESH`
で再現できる。[プログラム](../src/cst_research/experiments/polar_radius_chain_rule.py)と
[CPU検証JSON](radius-coordinate-chain-rule-20261108.json)を保存した。
分岐内部の代数の確認であり、新しいCUDA backward・optimizer・境界routingや
LM学習の検証ではない。

次の学習比較案は、現在の振幅依存L/Uと対数半径写像を残し、**同じ更新則**で
半径の損失微分を通す／止めるを比べるもの。具体案は、通常のAdam後に
`1<=sqrt(s²+t²)<=2` へ射影し、中心は周期的に扱う。半径方向の提案を許可し、
LR・データ・初期値は共通にする。端点での内向き微分の約束と、Parameter・moment・
復元を含む完全な検査が必要になる。既存の接線更新・activity clockのPolarは参考に
残せるが、それに対してbackwardとoptimizerを同時に変えるだけでは半径勾配の
効果を切り分けられない。この議論の時点ではLM比較は未開始だった。

その後、Adaで[FFNの同条件比較](ffn-polar-radius-gradient-20261108-results.ja.md)を完了した。
976更新後のdetach/liveのDEV CEは3.42403589/3.42238586、平均半径は
2.058201/1.622871、平均qは0.078821/0.068468。初期Parameter/logitは完全一致し、
学習後VJP・復元監査にも合格した。既存のphysical Linearを合成する研究用実装であり、
公開PolarのCUDAやdefaultを変更したものではない。曲線は重なり、1seedで一貫した
改善を示す結果ではない。
