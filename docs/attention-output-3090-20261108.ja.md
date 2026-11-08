# RTX 3090でのCST Attention/FFN全体検証

[English](attention-output-3090-20261108.md) | **日本語**

九工大Hubの復旧後、4090を確保できなかったため依頼者が3090を確保した。
読み取りで承認済み3090 profile、実機NVIDIA GeForce RTX3090、24GB級メモリ、
compute capability8.6、Torch2.6.0+cu126、CUDA12.6を確認した。
以前の4090 jobの保存先はこのサーバーから見えておらず、完了は未確認。
元実験を上書きせず、3090の結果へ付け替えない。

新しい専用ディレクトリとoperator
`b393ac653f3b3893c378acda1c1752919c1539d1` を使う。
consumerの変更は3090の明示的なGPU識別と研究登録だけとし、転送前に全ソースを
固定する。モデルの数式・許容誤差・既定GPUは変えない。専用の依存環境とnative
build先、CUDA architecture8.6を使い、GPU上では一度に1実験だけを動かす。
既存の永続launcher・資格検査・learner・checkpoint監査を使う。

幅512、pre-LN4ブロック、8heads、context128、expansion4、batch2、勾配累積4、
seed17、c8、minimum-001/q0、FP32/TF32off、AdamWのbaseLR3e-4・Atom群倍率16・
betas(.9,.95)・decay0・clip1を維持する。各mapのAtom数は
ceil(8sqrt(d_in*d_out))で、正方形4096、FFN8192。
公開variance scalingの目標stdはQ/K/V/upで0.02、O/downで0.02/sqrt(8)。
TinyStories/GPT2の固定manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`
を使い、train/DEVだけを参照する。confirmation・生成・HPO・最小Atom探索はしない。

まず回帰223件とlevels5条件、24か所levels_stream64の資格検査を行う。
実R256の独立CPU FP64 Y/dX/全Atom VJP、Gauss求積、手書きcausal Attentionと
未来勾配0、全箇所の更新、保存復元後の次更新とoptimizer状態の検査を要求する。
その後、Dense・FFN・QKV・QKV+FFN・QKVO・QKVO+FFN（CST0/8/12/20/16/24か所）を
それぞれ新規に976更新・999424tokens学習し、データ順とLRのhorizonは20Mtokensに
固定する。token/position embeddingと共有語彙出力はDenseのまま。
最終checkpointはDEVの厳密再計算、学習後の全CST mapの行数を制限したFP64 VJP、
独立2回の復元後の次更新とoptimizer状態の一致で監査する。
cache generationは明示した無効化・load契約に従う。

小さいlog・曲線・geometry・監査証拠を回収してhashを照合する。
大きいデータ・checkpoint・ソース・native buildは名前を付けたcluster保存先に
保持し、どのpayloadを回収したか明示する。途中・失敗結果を保存し、自動再試行は
しない。他の作業を再起動・停止しない。全学習曲線を重ね、更新時間の各sampleと
allocated/reservedピークを残す。既存learnerのタイマーはzero_grad/LR設定・準備・
評価・probe・checkpoint IOを除き、転送・4回のforward/backward・clip・AdamW・
同期を含む。wall timeは除外した作業も含む。allocatorピークはGPUプロセス全体の
使用量ではない。1seed・約1Mtokensの開発用試験であり、収束・非劣性の主張ではない。
3090のコホートと時間をL4のLinear単体benchmarkや4090の結果と分けて記録する。
