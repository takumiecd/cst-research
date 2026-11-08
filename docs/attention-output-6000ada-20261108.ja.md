# RTX 6000 AdaでのCST置換パイロット

[English](attention-output-6000ada-20261108.md) | **日本語**

Linear単体のGPU比較後、所有者が4条件の学習比較を承認した。
実機NVIDIA RTX 6000 Ada Generation 48GB・CC8.9・Torch2.9.1+cu128 / CUDA12.8を
1台使う。operatorは `b393ac653f3b3893c378acda1c1752919c1539d1`、consumerのcommitは
配置manifestで固定する。既存の永続launcher・資格検査・`ffn_run` 学習ループ・
学習済みcheckpoint監査を使う。稼働中snapshotの変更、無関係jobの再起動、
ハイパーパラメータ探索、confirmation評価は行わない。

## 固定比較

新規モデルを以下の固定順序で実行する。

| 条件 | Attention射影 | FFN | CST箇所数 |
| --- | --- | --- | ---: |
| Dense | Dense | Dense | 0 |
| FFN | Dense | CST up/down | 8 |
| QKVO | CST Q/K/V/O | Dense | 16 |
| QKVO+FFN | CST Q/K/V/O | CST up/down | 24 |

Transformerの全4ブロックを対象とする。この試験の「全部CST」はブロック内の全線形射影。
token/position埋め込みと共有語彙出力はDenseのまま。LayerNorm・bias・活性化・
causal SDPAも既存演算のまま。埋め込み・語彙出力の置換試験とは区別する。

幅512・4 pre-LNブロック・8 heads・context128・FFN expansion4・batch2・
accumulation4・seed17。各条件1M tokenを指定し、実際には976更新・999424 token。
20M-tokenのデータ順と学習率horizonを共通に固定する。
TinyStories/GPT2のmanifestは
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
trainとDEVだけを使い、共通の通常Parameterは名前ごとに同じ初期化を使う。

Atom予算は `ceil(8 sqrt(d_in d_out))`：正方形4096、FFN up/downは8192。
levels・FP32/TF32 off。variance scalingの目標標準偏差はQ/K/V/upが0.02、
O/downが `0.02/sqrt(8)`。q=0、rational_zero_opening、radius_birth=0.5、
zero_opening_scale=0.001、amplitude_max=1、w_c=0.3。
AdamWはbase LR3e-4・betas(0.9,0.95)・decay0・global clip1、Atom全体のLR倍率16。
既に選んだ設定を使い、条件ごとの調整はしない。単一の幅から線形スケールや
最小Atom数を主張しない。

## 資格検査と保全

同じ6000 Adaで既に回収・hash検証した回帰223件の記録は、operator全ファイル・
環境・log hashをreceiptと照合した場合だけ再利用する。再実行したとは記述しない。
新しいjob専用のnative buildと依存環境を使う。学習前に必要な3モデルの資格検査と
厳密なadmissionをこのGPUで新たに行う。実際のR256の全形状について独立CPU FP64の
Y/dX/全Atom VJPとprofile求積、手書きcausal Attention、未来tokenの勾配が厳密に0、
全CST箇所の更新、保存復元後の次更新とoptimizer状態を確認する。
Denseは資格検査済みの共通演算設定を使う。

各条件の学習後はcheckpointからDEVを厳密に再計算し、全学習済みCST射影の
有界行数FP64 VJPを検査する。独立した2回の復元から次の実タスク更新を行い、
optimizer状態まで比較する。失敗・途中結果も保持し、検査失敗時はその試行を止める。
自動retryや許容誤差の緩和はしない。学習は各60分、資格検査・監査は各25分を上限と
する。これは実行の上限で、終了予想ではない。

データ・ソース・checkpoint・native buildはclusterに保持する。コンパクトなreceipt・
曲線・geometry・summary・監査logを回収しhashを検証する。ソース・データ・設定の
同一性、checkpoint hash、コマンド時間、所有processを記録する。
launcherの切り離しだけで完了と判断しない。

## 測定

全条件のtrain/DEV CEとDEV perplexityを実token数・記録された経過時間に対して重ね、
記録された全点を残す。更新時間の中央値・tokens/秒・warmup後全sample・
ピークallocated/reservedを報告する。
既存の更新timerはzero_grad/LR設定と冒頭同期を除き、転送・4回のforward/backward・
clip・AdamW・末尾同期を含む。最初の10更新は除く。
評価・geometry・probe・checkpoint IOは更新timerの外で、wall timeには含む。
setupと資格検査は別記する。メモリはallocatorの学習時ピークで、GPU process全体の
使用量ではない。Linear単体benchmarkのtimerとも同一視しない。

1台・固定実行順序の資源観測であり、独立したhardware反復ではない。
1 seed・約1M tokenは学習挙動の確認で、収束・非劣性・学習完了LMの速度を示さない。
全4条件と失敗を公開する。接続不能となった以前の4090試行、停止したL4学習計画とは
別コホートとして扱う。
