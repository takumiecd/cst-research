# 初期fraction0と.5の比較：CST共通LR32倍

[English (canonical)](ffn-fraction-zero-start-20261109.md) | 日本語

ユーザーの指示で、CST共通LR32倍・fraction0スタートを実行します。同一RTX6000 Ada
jobでfraction0・.5とDenseを新規比較。Atom数・tau・振幅上限・token順・学習予算・
非Atom LRは固定します。過去の曲線は参考とし、.5とDenseも今回のsource・観測処理で
再実行して対照にします。

TinyStories、幅512、pre-norm4ブロック、8heads、context128、拡大率4。FFN8行列を
各8192Atoms(A=8√(d_in d_out))へ置換し、Attention・embedding・共有headはDense。
tau=.003、振幅上限32、半径[.5,50]、M8、levels_wide_amplitude、公開moment初期化。
初期fractionのみ0/.5とします。同じseed・moment目標(up分散.0004/down.00005)を
使い、公開initializerが各fractionに対して振幅を初期化します。従ってfraction間の初期
Parameter hashは一致を要求せず、各設定のqualificationと学習のhash一致を要求。
中心の乱数とデータ順は固定。probe/output RMSの校正はしません。

CSTの振幅・入出力中心・fractionはすべて基準LRの32倍、最大.0096。Dense部分・
FFN bias・normを含む他のParameterは基準LR3e-4のまま。AdamW betas(.9,.95)、
eps1e-8、decay0、global clip1、warmup200、共通20M-token予定長。seed17、
batch2×128×蓄積4、各976更新/999424tokens、256更新/262144tokensでも観測・評価・
checkpoint保存。data manifestは
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
DEV8192targetsのみ、confirmation未使用。Parameter・moment・physical rowはFP32、
写像内部はFP64、TF32off、eager。

operatorの既存68検査は全byte・GPU/runtime・hash一致でのみ再利用。両fractionで
consumerを新規検査し、全写像VJP・未完了forward保持・fraction実更新・serialize後の
次の更新完全一致を確認します。source固定前に、振幅上限が広い場合のresume gateを
初期fraction0にも適用します。0での片側の範囲制約を自動的な勾配消失と解釈しません。

学習曲線、初期/最終activation RMS、固定probeのfield勾配、振幅別半径/fraction、
層別分布、各Atomのfraction実更新/net/pathを記録。初期0は内側への更新のみ可能で、
端点の割合だけでは適切な下限解と有用な更新の阻害は区別できません。observerは
読み取り専用で、更新時間の測定外です。

最終/256更新の6checkpointを監査：DEV差<=1e-6、FP32 Adam momentの有限性、
256/976counter、範囲、学習後up/downの8encoded-row FP64 oracle
(CST2条件×2checkpoint×2行列)でY/dX/全Atom微分を従来atol1e-4/rtol4e-3で比較。
4時点のobserver net変化を独立再現。失敗・source/checkpoint/result hashを保全し、
compact取得検証・remote checkpoint保全・所有PID終了を確認します。失敗条件の
再調整/再実行、追加探索の自動実行はしません。

時間は蓄積・転送・forward/backward・clip・optimizerを含み、zero_grad・LR設定・
probe・評価・observerを除きます。準備/wall時間とPyTorch allocated/reserved peakを
記録。同一jobでの逐次測定ですが順序反転はなく、速度比較は探索的。process全体/Graphの
メモリや公開defaultの主張はしません。1seedの初期化比較だけでは収束・非劣性・最適な
fraction・容量・Dense差の唯一の原因は特定できません。
