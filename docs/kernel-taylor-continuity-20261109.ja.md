# Gaussian・Fourier区間のTaylor連続性診断

[English (canonical)](kernel-taylor-continuity-20261109.md) | 日本語

所有者が依頼したCPU数学分析のみです。学習・optimizer・本番kernel/既定値の変更・
速度benchmark・Gaussian採用は行いません。座標・幅の実変化が、変位を小さくすると
一次予測へ近づくか、現行ChebyshevのMを増やすと経路切替の跳びが変わるかを確認します。
実行前にこのprotocolとrunnerを固定します。

周期128x256・間隔1のgridで振幅1の1 Atomを使います。独立したGaussian対照は、
周期Gaussianを全固定cellで積分します。CDF差の裾をerfcで安定計算し、sigma>0、
固定image和-4..4を用います。最大幅2.01なので省略imageはFP64精度より十分小さく、
動的な打切りや最短image距離は使いません。独立に導いたPDFによる偏微分、64点の
Gauss-Legendre積分、総mass、非負性、中心の周期移動との同一性で検証します。
CDF定義は[SciPy ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html)です。

比較対象は固定したpublicの物理半径cell-mass reference、Chebyshev M4/8/16/32/64です。
Mは密度自体を変え、固定Gaussianの近似精度だけを変える値ではありません。
sigmaとradiusの数値を揃えるのは同じ幾何境界を見るためで、形・出力RMS・モデル品質を
揃える比較ではありません。referenceのM32/64はCUDA対応の証拠にはなりません。

事前固定19条件：物理幅境界0.5/1/2を両方向、入力・出力中心の境界は
(幅,中心)=(0.5,7),(0.5001,7.5),(0.75,7.5),(1.5,7.5),(0.75,0)、
枝内controlは(1,0.75,7.3,13.7)で幅・出力・入力中心を変えます。
半変位は1e-2/1e-3/1e-4/1e-5/1e-6/1e-8/1e-10/1e-12です。
Jvには名目変位でなく実際の符号化された端点差を使います。有限・予測ノルム、
Taylor誤差の絶対値と相対値、cosine、経路metadata、FP32の丸めによる無変化を記録します。
主分析FP64は全6profile、補助FP32はGaussianと現行M8で、合計152条件・1216変位です。
同じ区間内のcontrolも含め、境界切替と滑らかな枝内計算を分けます。

Gaussianの独立JVPとの相対誤差はFP64で1e-11未満を要求します。既知解テストでは、
積分絶対誤差2e-15以下、総mass誤差3e-15以下、周期同一性、h1e-4でcosine>0.99999、
h1e-3→1e-4のTaylor残差が概ね二次で縮小することを確認します。
極小hでは丸めや微分がほぼゼロなことにより相対比が不適切になり得るため、絶対誤差も
保全し、符号化上の無変化は滑らかさの証拠と解釈しません。Mの傾向を形を揃えた
近似収束と解釈せず、小さいfixtureから学習品質へ外挿しません。

consumer scriptはscripts/compare_kernel_taylor.py、operatorは
0d277ae6dc0ae14fe515c7559f39535ed0238950です。
raw証拠はCST_RESEARCH_EVIDENCE_ROOT/kernel-taylor-continuity-cpu-20261109-01です。
source/runtime hash、結果、失敗、英日結果を保全します。GPUは確保せず、既存学習と
証拠は変更しません。
