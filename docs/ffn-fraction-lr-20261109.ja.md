# 初期fraction .5でフィールドLRだけを比較

[English](ffn-fraction-lr-20261109.md) | 日本語

ユーザーは初期radius_fraction=.5を固定し、そのLR倍率だけ1/4/16へ変える比較を承認した。
RMS校正は使わず、probe-RMS初期化は未実行の任意準備として保全する。
以前の.5開始では全Atomが動き、256更新後の平均絶対変化は.00384だったが、増減が
相殺されていた。勾配消失を示した結果ではない。

tau.003、振幅上限32、明示backend levels_wide_amplitude、幅512、4pre-normブロック、
8head、context128、expansion4、各FFN8192Atomを固定。Attention/embedding/共有headはDense。
公開moment初期化up分散.0004/down.00005、named seed・入力順序も共通。
CST3条件の全初期Parameter hashが完全一致することを要求する。初期出力RMSをDenseへ
合わせる必要はないが、LR3条件間は同一である。旧広い開始の診断をDenseとスケール一致した
比較へ読み替えない。

seed17のみ。新規1x/4x/16xとDenseを各976更新/999424tokens、256更新/262144tokensでも
観測・評価・checkpointを保存する。20M-tokenの順序/schedule、warmup200、batch2×128×
accumulation4、AdamW baseLR3e-4、betas(.9,.95)、eps1e-8、decay0、globalclip1を共通化。
振幅/中心は全CST16倍、fraction1/4/16倍の最大絶対LRは.0003/.0012/.0048で同じscheduleに追従。
Parameter/moments/物理行FP32、写像中間FP64、TF32無効。成績で予算を変えず、追加探索・
3seed確認・公開default変更を自動では行わない。

検証済みtoken manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`を使用。
DEV8192対象のみ、confirmationは使わない。旧1x曲線は文脈に留め、新sourceで1xとDenseも
実行する。operatorの68GPU検査は全source/GPU/runtime/hashを確認した場合だけ再利用。
consumerはLRごとに新規検査し、写像/Jacobian、複数forwardの保持、fraction実更新、
保存復元後の次のmodel/Adam state完全一致を確認。observerはParameter/勾配/moments/RNGを
変えず、sourceに結び付く実行許可に含める。

各更新で射影後のAtomごとのfraction増分、初期値からの正味変化、増減/不変割合、
絶対移動の累積を記録。step0、1..5、20刻み、256、976では実半径の変化も記録し、
r(a_t,f_t)-r(a_0,f_0)=[r(a_t,f_t)-r(a_t,f_0)]+[r(a_t,f_0)-r(a_0,f_0)]へ厳密に分解。
前者は現在振幅でのfraction成分、後者は初期fractionでの振幅依存境界の成分で、
学習軌跡の因果分解ではない。端点割合・固定probe各勾配と、保存FP32 moments/実フィールドLRから
FP64で求めるAdam提案の解析的推定も残す。推定は境界・丸め前であり、実raw提案やclip率の
厳密なtraceとは呼ばない。

更新/token/wall時間での学習曲線、半径/fraction分布、振幅bin、実activation RMSを保存。
observerのcopy/復号/reductionは更新timer外。wall時間には含み、速度/メモリは探索値で、
kernel採用用の主測定ではない。更新時間は従来範囲（蓄積/転送/前後伝播/clip/optimizerを含み、
zero_grad/LR設定/probe/評価を除く）。Graph・GPU process全体メモリは主張しない。

全4最終checkpointと256更新checkpointを監査。DEV再現1e-6以内、counter、FP32 moments有限性、
実上限を確認し、学習済みup/downの非連続7行Y/dX/全Atom VJPをFP32物理行/FP64参照に
既存atol1e-4/rtol4e-3で照合。raw/checkpoint/source/result hashと失敗を保全し、compact証拠を
検証回収、大きいcheckpointをremoteに保持、所有process終了を確認する。
1seedの機構探索であり、収束・普遍的最良LR・非劣性・公開default推奨は示さない。
