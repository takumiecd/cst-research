# CST全フィールドの学習率：16/32/64/128倍

[English (canonical)](ffn-all-fields-lr-20261109.md) | 日本語

ユーザーの指示で16倍より高いCSTのLRを検証します。振幅・入出力中心・radius_fractionを
同時に基準LRの16/32/64/128倍へ変更。他のParameterは全条件で基準LRのままです。
DenseのAttention・embedding・共有語彙head・norm・FFN biasも含みます。
同じjobでDenseと16倍も新規実行し、過去の非対応な時間対照は混ぜません。

前のfraction LR比較と同じ初期fraction=.5、tau=.003、振幅上限32、半径[.5,50]、M8、
levels_wide_amplitude、公開moment初期化。RMS校正はしません。TinyStories、幅512、
pre-norm4ブロック、FFN8行列各8192Atoms(A=8√(d_in d_out))、heads8、context128、
拡大率4。CST4条件の初期Parameterのhash完全一致を要求します。

seed17のみ。全5条件を976更新/999424tokensまで実行し、256更新/262144tokensでも
評価・checkpoint保存。batch2×128×勾配蓄積4、AdamW基準LR3e-4、betas(.9,.95)、
eps1e-8、decay0、globalclip1、warmup200更新、共通20M-token予定長。
CST各フィールドの最大LRは.0048/.0096/.0192/.0384。Parameter・Adam moment・
physical rowはFP32、写像内部はFP64、TF32off、eager実行。token順序を固定し、
data manifestは
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
DEV8192targetsのみ。confirmation評価・追加探索の自動実行はしません。

既存operator検査はtorchcst全byte・GPU/runtime・証拠hashの一致を条件に再利用可能。
consumer検査は今回の振幅・中心・fractionの実際のLRで新規実行し、未完了forward保持、
全写像VJP、fraction実更新、serialize後の次の更新の完全一致を要求します。
以前の検査helperは振幅・中心16倍固定だったため、source固定前に設定可能に変更します。
CPU回帰で16〜128倍の全field LRとscheduleの1回だけの適用を確認します。

学習曲線、固定probeのfield勾配、activation RMS、振幅別の半径・fraction、分布と
既存の読み取り専用observerによる各Atomのfractionのnet/path/更新を観測します。
sample更新時点のclip前global勾配normも記録します。生の勾配だけでなく実際の変化を
比較します。LR増加が勾配消失を直すという因果や、このAtom予算でDense同等に届くことは
前提にしません。非有限値のrunは失敗条件として保全し、再調整・再実行で置き換えません。
数値qualification失敗時は停止し、証拠を保全します。

最終・256更新の計10checkpointを監査：DEV差1e-6以内、FP32 Adam momentの有限性、
counterと範囲、学習後up/downの16oracle検査(CST4条件×2checkpoint×2行列)で
非連続7行のY/dX/全Atom微分を従来atol1e-4/rtol4e-3で比較します。8時点のobserver
net変化を独立に再現します。checkpoint/source/result hashを保全し、compact取得検証と
所有PIDの終了を確認します。

時間は勾配蓄積・転送・forward/backward・clip・optimizerを含み、zero_grad・LR設定・
probe・評価・observerを除きます。準備/wall時間とPyTorch allocated/reserved peakを記録。
Graph/process全体のメモリ・普遍的速度は主張しません。高LRで学習後の処理量も変わります。
開発1seedの機序検証であり、収束・非劣性・普遍最適値・公開defaultの結論にはしません。
