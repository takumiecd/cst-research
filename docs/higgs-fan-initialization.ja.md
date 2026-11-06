# He／Xavier互換のCST初期化pilot

[English (canonical)](higgs-fan-initialization.md) | 日本語

限定した実行・独立した全保存証拠の検証は完了しています。[結果と限界](higgs-initialization-results.ja.md)を参照してください。以下のprotocolは元の設計を保持し、準備中・将来形の記述は当時の記録です。

研究用の初期化であり、torchcstの公開defaultは変更しません。
`q`・`s`・`t`は小文字で表記します。0〜1で指定する割合qはPolarの
`alpha_init`に対応し、Atom密度alpha=A/N²とは区別します。学習前に
rho=sqrt(1+3q)、s=rho*a/amplitude_max、t=rho*sqrt(1-(a/amplitude_max)²)
と符号化します。物理半径はR=L(a)^(1-q)*U(a)^qです。q=0は振幅依存の下限、
q=1/2は上下限の幾何平均、q=1は上限です。既存の半径範囲とpublic975のruntimeを保持します。

## 数学的契約

W=sum_a a_a Phi_aとし、振幅を除くAtom行列Phi_aを固定し、振幅を独立・
ゼロ平均・分散sigma_a²とすると、E_K=sum_a ||Phi_a||_F²に対して
期待値E||W||_F²=sigma_a² E_Kとなります。実効fanをE_K/d_out、E_K/d_inと
定義すれば、He／Xavierに代入することで対応するDenseと同じ平均的な
順伝播・逆伝播の二乗平均目標になります。Denseの全要素を座標基底Atomで
表した場合は、実効fanが通常のd_in、d_outに正確に戻ります。

| 方式 | 目標E||W||_F² |
|---|---:|
| He・fan_in・ReLU gain=sqrt(2) | 2*d_out |
| He・fan_out・gain=sqrt(2) | 2*d_in |
| Xavier・gain=1 | 2*d_in*d_out/(d_in+d_out) |

一般にはHeはgain²*d_in*d_out/fan_mode、Xavierは
gain²*2*d_in*d_out/(d_in+d_out)です。utilityはgainとmodeを明示指定できます。
正方形で単位振幅Atomの平均エネルギーがeなら、形状固定のHe式は
sigma_a=sqrt(2/e)/sqrt(A/N)です。

実際のPolarカーネルでは形状が振幅の絶対値に依存するため、その依存と
クリップを含めて目標二乗平均を解きます。public975のq=0初期化から保存済みの
名目標準化Gaussian乱数zを取り出します。試行scaleごとに
a=amplitude_max*clip(scale*z/amplitude_max,-1+1e-6,1-1e-6)
を計算し、s,tを符号化します。実際のFP32 Atomsをpublic CPU FP64因子で評価します。

実現済みの絶対値と中心に対して振幅の符号を独立にランダム化した条件付き期待値は、
E_sign||W||_F²=sum_a a_a²||Phi_a(a_a,q)||_F²です。形状は振幅の二乗に依存するため
符号反転では変化せず、Atomが重なっていても成立します。ある1回の実現Wが
このノルムに一致することや、各行のエネルギー・相関・特異値がDenseと同じであることを
意味しません。各Atomのエネルギーは二つの因子ベクトルの二乗ノルムの積で計算し、
Dense W・TRAIN特徴・参照Denseモデルを必要としません。

scaleを0・.025・.05・.1・.2・.4・.8・1.6・3.2・6.4・12.8・32と試して
符号の異なる区間を見つけ、最大32回二分探索します。実測の目標誤差0.5%以内のみ
受け入れます。全体の単調性は仮定しません。探索が失敗したら履歴を保持し、学習前に
停止します。q・振幅上限・Atom数を変更したり、gainへ自動fallbackしたりしません。
この有限探索は探索経路外での数学的な到達不能性を証明しません。
対称クリップ後の振幅は厳密な正規分布ではありません。名目scale・物理振幅RMS・
全クリップ数を残します。ReLU gainを別の活性化関数や残差構造へ一律に適用しません。

## 完了した初期化実験との関係

以前のN1024/H8/A4096実験はすべてq=0でした。旧scale .1/sqrt(A)=.0015625と
幅対応 .1/sqrt(A/N)=.05はいずれもDEV AUC .5で停滞しました。その後のTRAIN特徴を
使ったDense-RMS合わせは、層ごとの名目scale .315625〜.359375、gain1・LayerNormなしで
AUC .765233、Dense .771360でした。1 seedの結果です。これは実入力と従来の
nn.Linearに合わせたもので、He／ReLUではありません。従来のDense内部重みの
集団分散は1/(3*d_in)、He／ReLUは2/d_inです。今回はDense内部初期化も変更します。

## 固定した最初の学習比較

GPUごとに独立したcohortとし、それぞれ新規の対応するDense／CSTを比較します。

- RTX4090：He、fan_in、ReLU gain=sqrt(2)、q=0。
- A100 MIG3g.40gb：Xavier、gain1、q=0。

両方ともN1024/H8・内部7層・各4096 Atoms・seed17、正規化なし・gain1、
対応する従来のDense入出力初期化、同じTRAIN100k/DEV50kの行、保存したseed117順序、
実B4・FP32・TF32無効・eager、stockq1 AdamW/CSTAdamW、LR.0003・eps1e-8・
通常/amplitude decay.01・生Atom decay0です。内部7層だけに選択した初期化を適用します。
Denseは明示seedで対応するPyTorchのnormal初期化を使います。
正方形なのでXavierのfan_in/fan_out選択は一致します。

各条件1 epoch・25,000実データ更新で最終固定選択し、HPO・best-DEV選択はしません。
新規の全ネットワークY/dX/全パラメータFP64 VJP、独立31Atom求積、scalarAdam、
通常8更新と4+load4およびobservedfirst+7の状態一致を先に確認します。
GPUごとの合成ネットワーク40更新とscalar8は、実データ50,000更新とは別です。
数値判定条件とoptimizer/loadの数式は変更しません。
0/1000/5000/10000/15000/20000/25000で全TRAIN/DEV予測と固定最初B4の独立copy上の
順伝播・勾配probeを保存し、実際の初回更新のmoments・採用変位も保持します。
実行時gainは1です。qは初期条件であり、学習中にradial activityを固定する拘束ではありません。

初期化費用、素の完全学習step時間とallocated/reservedピーク、評価/probeの費用、
job全体時間を分けて記録します。別GPUのHe／Xavier時間から初期化方式の高速化は
判断しません。精度を解釈する前にsource/data/config/runtimeの固定、正常終了、
全保存証拠の独立CPU検証、クリップ分布を残します。confirmationと公式testは採点しません。
1 seed・1 epochは探索的screeningです。成功しても非劣性や普遍的defaultの根拠には
しません。公開採用には別のAPI/契約変更とレビューが必要です。以前の生RMS方式を
独立の比較条件として保持し、Heと呼び替えません。

entrypointは`python -m cst_research.experiments.higgs_fan_initialization_run --help`。
`--scheme he|xavier`・`--q VALUE`と、不変のsource/input/runtimeパスを明示します。
目標到達や学習に失敗しても、追加のqやAtom予算を自動探索しません。
現在の許可対象はこのpilotです。広い研究goalは停止したまま、LM作業は待機します。
