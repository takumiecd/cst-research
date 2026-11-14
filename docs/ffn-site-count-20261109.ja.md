# FFNの置換箇所数と合成の診断

[English](ffn-site-count-20261109.md) | 日本語

全8写像CSTのLMは学習するが同予算DSTに届かなかった。以前のHIGGSの1層成功とは
タスク・構成も違うため、その比較だけでは深さの影響を特定できない。
今回は同じTinyStories LMを4ブロックに固定し、FFNの置換箇所のみを変える。
新しい探索的研究であり、準備をGPU実行・結果として扱わない。

## 固定比較

幅512・8ヘッド・文脈128・拡張4。同じseed17・データ・順序・通常層の名前別初期値、
batch2・蓄積4。新規初期化で5M-token prefix（4882step・4999168tokens）を学習し、
LRと接続更新のhorizonは以前の20Mのままにする。新たなLR/tau/Atom探索、DEVによる
早期停止・選択、confirmation評価はしない。

ブロック番号は0始まり。各置換箇所でCSTと同じ有効slot予算のDSTを比較する。

| 条件 | 置換する写像 | 個数 |
|---|---|---:|
| up | block1のup |1|
| down | block1のdown |1|
| block | block1のupとdown |2|
| two_blocks | block1・2のupとdown |4|
| all | 全4ブロックのupとdown |8|

新規Denseを加え計11run。全8写像CST/DSTも同じcohortで新規実行する。
旧20M jobは学習終了後、ローカル検証済み回収前のサーバー停止でcontainer-homeの
証拠が見つからなくなったため、そのprefixは借用しない。
attention・embedding・共有head・未置換写像はDenseで全パラメータを学習する。
Denseモデルから対象mapのみ、元と同じ名前別seedで置換する。全8箇所なら過去の
全CST/DSTモデルとstateが完全一致しなければならない。公開runtimeは変更しない。

CSTは写像あたりA8192・envelope tau.003・fraction0・振幅上限32・半径.5..50・
levels_wide_amplitude・moments初期化・全4フィールドLR32倍を維持。
DSTは写像あたりK32768・疎LR.003・通常LR3e-4。有効FFNスカラー総数は各箇所条件の
CST/DST間で一致するが、置換箇所数の間では圧縮量が変わる。
FP32・TF32無効・Adam betas(.9,.95)・eps1e-8・WD0・clip1・warmup200・20M cosine。
DSTは4750まで19回接続更新し、4882学習stepで実Adam更新4863。

## 原因の手掛かりと検査

初期・10・100・256・976・200ごと・最終stepに固定TRAIN probeを保存する。
全4ブロックと8射影の入出力RMS・出力cotangent RMS、FFNの生勾配、field別勾配、
振幅ごとの半径/fractionを追う。異なるパラメータ化の生勾配の大小は、更新scaleの同等性を意味しない。
同じ7件の固定Gaussian入力に対する1stepの実際の射影変化と、Atom各fieldの変化
（座標差はchart周期でwrap）も測り、座標表現に依存せず関数の変化量を比較する。
clip前norm・時計・有効field LRを記録。同じ8192 DEV targetsのCE/perplexityと
teacher forcing次トークンtop1を測り、TRAIN CEは別記する。

実行前に独立FP64の初期写像微分、保存・再開した通常/接続更新の完全一致、
診断あり/なしで更新が変わらないことを検査する。CPUでDenseの未置換値・全8箇所の
過去state一致・有効スカラー予算・RNG・read-only probeを確認。
全33 checkpointでDEV CE/top1・有限FP32 moments・時計を監査し、学習済み写像oracle・
source/checkpoint/compactハッシュを通してから科学的検証完了とする。

速度・メモリは10step後の同期した非計測介入stepのみを使い、固定probeはその範囲外。
準備・wall時間は別記。過去の全8箇所の時間は同一job対照ではない。
証拠は永続NFSに保存し、大きなcheckpointを保持。compact証拠は定期的にローカル回収・独立検証する。開始前にsourceを固定。
証拠：CST_RESEARCH_EVIDENCE_ROOT/ffn-site-count-4090-20261109-01。

仮説は信号/勾配の減衰、実現する更新量の不適合、構造化写像を合成する際の制約。
診断を次の介入に使うが相関だけで原因確定はしない。中央箇所だけでは位置に依存しない
振る舞いは示せず、固定予算・1 seed・5M pilotで表現不可能や普遍的な深さの法則を証明しない。
