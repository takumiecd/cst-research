# Fourierの2階層・2窓接続案

[English (canonical)](fourier-scale-connection-20261109.md) | 日本語

所有者が依頼した数式導出・CPU適合確認のみです。固定public Fourierセル積分を合成した
新しい数学定義案で、本番operator/Plan・モデル・optimizer・GPU・速度/学習の変更では
ありません。過去結果は元の定義に固定します。初期profileも変わるため、既存checkpointへ
互換な差替えではありません。

## 固定bankと幾何

T_l=2^l*T0と既存K3 bankのP_l=3T_lを使います。間隔1では固定窓nのunwrapped siteは
n*Tから(n+3)*T-1です。b_j^(T,n)(mu,R)を、その固定窓でpublic周期密度をwhole-cell積分し、
global site数のmoduloへ重複imageも加算したものとします。各枝は非負・総mass1で、
0<R<P/2ではmu,Rについて滑らかです。public integrated_cellsを呼び、profile評価器は
複製しません。L2ノルムfloorも導入しません。

S(t)=10t^3-15t^4+6t^5を[0,1]で用い、外側を0/1とします。
S'=30t^2(1-t)^2で、両端のS',S''はゼロです。各枝がC2ならC2接続になります。
数値評価は対称な端点式を使います。

Tを固定してq=(mu+1/2)/T-3/2、n=floor(q)、t=q-nとします。2つの固定窓の中心が
muを挟み、その最近傍窓の切替付近を

\[
 u=\operatorname{clip}\!\left(\frac{t-(1/2-\delta)}{2\delta},0,1\right),
 \quad m_j^T=(1-S(u))b_j^{T,n}+S(u)b_j^{T,n+1}
\]

で混ぜます。接続区間の外は1窓だけです。nが変わっても、非ゼロの窓と微分は両側で一致します。
全ての有効な窓で、muから近い側の端までの距離は少なくとも(1-delta)*Tです。
従ってcore全体を覆うにはR<=b*T、b<=1-deltaが必要です。この余白がない単純な2窓混合は、
有効coreを切る可能性があります。

## 半径の接続

0<delta<1/2、0<a<b<=1-delta、b/2<a<bを選びます。R<=b*Tを満たす最小Tについて、
a*Tより下はm^Tのみ、[a*T,b*T]では

\[
 v=\frac{\log(R/(aT))}{\log(b/a)},\quad w=S(v),\qquad
 m_j=(1-w)m_j^T+w m_j^{2T}
\]

とします。b*Tではw=1,w'=w''=0で、その直上ではprimary Tが2Tへ変わり、同じm^(2T)が
単独になります。a*Tではw=0で同様に微分がゼロです。2a>bなので隣接する接続帯は重なりません。
これにより半径と座標の経路境界でC2・global周期性・非負・総mass1を満たします。
1軸当たり最大2bank・bank当たり2窓で、同じbankの2窓は係数を共有できます。
入出力軸を混ぜても、1 Atomの積のfactorizationを維持します。

有効bankの比は、宣言した最小/base尺度領域を除きa/2<R/T<=bです。
先ほどの[1/2,1]を全ての補助bankにも厳密に課すと、接続用の幅がなくなるため緩和が必要です。
例のdelta=.1,b=.9,kappa=b/a=1.25ではa=.72、有効比は(.36,.9]です。
これは幾何と整合する例で、採用するハイパーパラメータではありません。
最小tile幅に達した後にRを下げる場合は、引き続き固定Mの集中限界があります。
今回の主な裾の主張はR>=.5に限定します。

半径微分には接続項を含めます。

\[
 \partial_R m=(1-w)\partial_R m^T+w\partial_Rm^{2T}
 +\frac{S'(v)}{R\log(b/a)}(m^{2T}-m^T)
\]

座標微分にもS'(u)/(2delta*T)×隣接窓の差が必要です。どちらもstop-gradientにしません。
接続幅を狭くしすぎると微分が大きくなり得るため、C2だけで条件数を保証しません。
新半径scheduleは有効幅と初期mass energyも変え、既存の初期化には別の適合確認が必要です。

## 実行前CPU protocol

1軸size128・間隔1、固定operator0d277ae6dc0ae14fe515c7559f39535ed0238950を使います。
FP64 M8では旧public経路、半径だけの接続、全接続、接続weightの微分を意図的に省く2条件を
比較します。M16/32は全接続だけです。20fixtureはT1/2/4のa*T/b*T両方向、旧半径.5/1/2、
座標接続帯の端・中央、座標と半径の同時変化、周期境界です。変位1e-2..1e-12の8尺度で
140条件・1120変位です。

裾の限定screenはdelta.1,b.9固定、kappa1.05/1.1/1.25/1.4、M8/16/32、半径.5..4の
対数等間隔41点、計492点です。public support_epsのcore外連続massを混ぜます。
core被覆の証明により、有効な座標窓でも同じunwrapped裾比になります。
総massの測定で、厳密な最大sidelobe・タスク許容差ではありません。
狭い接続帯は補助bankの漏れを減らす一方、接続の微分を大きくします。
幾何は49点、独立テストにはimageが重複するsize8も含めます。

検証は凸weight、端点の一次/二次微分ゼロ、外側の連鎖律一致、mass誤差1e-12以下、
core被覆・2bank/4窓上限、周期同一性、独立cell求積、境界での概ね二次のTaylor誤差縮小、
半径weight微分を省いた場合の一次誤差を要求します。
絶対誤差と誤差/パラメータ変位も記録し、微分が小さい・丸めが支配する場合のcosineや
相対Jv比だけでは判断しません。枝数が定数でも高速性の証拠にはなりません。
GPU・学習の判断前にsource/result hash、負のcontrol、制限を残します。

runnerはscripts/inspect_fourier_connection.py、証拠は
CST_RESEARCH_EVIDENCE_ROOT/fourier-scale-connection-cpu-20261109-01です。
