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
