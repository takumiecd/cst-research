# RTX6000 AdaでのPolar半径損失勾配の2条件比較

[English（正本）](ffn-polar-radius-gradient-20261108.md) | 日本語

振幅依存L/U、q=clamp((s²+t²−1)/3,0,1)、対数補間の半径値を維持する。
`polar_detached` と `polar_live` を同じCartesian AdamWと更新後の
1<=norm(s,t)<=2への射影で比較する（FP32保存の許容差5e-7）。
半径方向の提案を許し、接線射影・activity clock・半径正則化・新LR探索は使わない。
中心は演算子内で周期的に扱う。原点・非有限値の提案は中止する。
Adam momentはCartesianのままで射影に伴うtransportはしない。
そのためdetach版のqも成分別Adamで動き得て、q固定の対照ではない。

liveの連鎖律はL/Uの振幅依存の傾きとq項を両方含む。
grad_st L = g_a grad a + g_r grad r。q0/q1では内部からの片側の傾きを使う。
明示したstraight-throughの値補正で、既存の厳密なforward値と制約内の端点Jacobianを
両立する。clamp角の無制約な両側微分という主張はしない。
detach版は振幅を通る項も含め半径の経路全体を止める。

同じ実現初期値のs/t/中心と通常Parameterを使う。幅512、pre-norm4ブロック、
8head、context128、FFN拡張4、CST8射影、各A=8 sqrt(512*2048)=8192Atom。
Attention、embedding、共有語彙出力はDense。minimum基準rational-zero-openingの
scale.001、birth.5、q0、振幅上限1。演算子は
88c2e8a7c4aa312f4f096c5169d712ace49ac979を変更せず固定する。
研究wrapperが元のPolar Atomを所有し、物理値Tensorに写像して、公開LinearBinding/
Dispatcherから検証済みphysical Planを呼ぶ。runtime実装を研究packageへコピーせず、
公開Polarのdefaultやbackwardも変えない。

AdamW baseLR.0003、Atom16倍、betas(.9,.95)、eps1e-8、decay0、globalclip1、
FP32/TF32無効。warmup200更新と20M-tokenのLR/データhorizonを共通にする。
seed17、batch2×128×蓄積4、各976更新・999424token。実機RTX6000 Ada48GBを1台使い、
detached→liveの固定順で実行する。token manifestは
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
DEV8192tokenだけを評価し、confirmation・生成・HPOは使わない。
従来Polar・独立半径は参考であり、同じ更新則の一次対照と混ぜない。
収束、非劣性、最良のLR/半径、Atom数のオーダーは求めない。

学習前に、内部点のCPU代数・中央差分、合成kernelの独立Gauss求積、両端点の傾きを
含むCPU/GPU写像一致、両FFN方向の実256非連続行・全8192AtomのY/dX/全4成分VJP、
再backward、初期値の完全一致、全射影のタスク更新、optimizer復元を確認する。
強制GCなしの完全backward32回でallocatedの幅を128MiB未満とし、functional bindingの
残留を検出する。演算子の全source byteと実機GPU/Torch/CUDAの一致を確認した場合だけ、
前回の回帰231件とnative buildを再利用する。合成写像・モデルの検査は新規に行う。

step0、1〜5、20ごと、976で固定probeを55回、更新タイマー外で記録する。
振幅、半径、L/U、q、latent normの統計・境界占有率、写像前の物理偏微分dLoss/dr、
伝播する半径寄与（detachでは0）、latent成分勾配とその半径方向成分を残す。
detachでも物理偏微分自体は非ゼロになり得て、その上流で止める。
これはscale/clip前の固定probe勾配であり、蓄積した学習勾配や座標の速度ではない。
学習・半径・q・半径方向勾配を重ねる。同期した完全更新時間、allocatorの
allocated/reserved学習ピークと診断/setupを含むwall timeを分けて残す。
functional binding/Python写像のoverheadを含む観測であり、新しい最適化kernelの
性能主張ではない。

976更新後、checkpoint独立復元、DEV厳密再計算、全wrapperのCPU参照VJP、
最終probeとcheckpointの一致、annulus制約、次のタスク更新・Adam state一致を確認する。
compact evidence回収・hash照合と所有worker非稼働を確認してからCOMPLETEDにする。
データ/checkpoint/native buildと失敗した凍結試行は保全する。
公開Polarは変更せず、採用は別途判断する。

runnerは `cst_research.experiments.ffn_polar_radius_qualification` と既存
`cst_research.experiments.ffn_run --kind cst --attention-kind dense --width 512
--heads 8 --linear-backend levels --radius-mode {polar_detached,polar_live}
--track-radius --development-only`。固定したdata/admission hashと上記全設定を明示する。

実行後の精度形式の明確化：同じ凍結sourceで、Parameterとphysical Linear Tensorは
FP32だが、Python写像はFP64で計算し、autogradはFP64の写像中間値を保存し得る。
Z/H/G response保存はFP32のまま。正しさ・学習の検証用adapterであり、全計算FP32の
最適化kernelではない。[結果](ffn-polar-radius-gradient-20261108-results.ja.md)を参照。
この説明のためにsource・データ・更新則・数値許容差は変更していない。
