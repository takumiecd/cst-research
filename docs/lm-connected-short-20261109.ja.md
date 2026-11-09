# Connected M16の短期タスク学習と有限更新の予測一致度

[English](lm-connected-short-20261109.md) | **日本語**

実行前に固定。ユーザーが短期学習比較と座標・半径更新の予測とのズレの確認を許可した。
1seedの探索的タスク実験であり、confirmationやAtom数下限の研究ではない。

## 固定した比較

既存の物語単位で分離したGPT2-tokenized TinyStoriesデータを使用。幅512、pre-normの
因果4ブロック、8head、context128、FFN拡張4倍、Dense attention、共有Dense埋め込み・
出力層とする。CST/DSTではFFNの全8写像を置換する。CSTは各写像8192Atoms
（A=8 sqrt(d_in*d_out)）。Denseと行数均衡RigL型DSTを新たに対照として実行する。
DSTはdense保存で各写像4Aの有効scalarを持ち、既存のsource-qualifiedアルゴリズムと
選定済み疎LR.003を使用。新しいLR探索、教師行列・出力の再構成、データ選択はしない。

Dense、全DST、従来の非接続M8-envelope、検証済み行連続FP32実装のconnected M16-envelope
の4条件を初期から実行する。両CSTのraw初期Parameters（振幅・割合を含む）は完全一致。
全条件で共有する通常Parameterもbyte一致を要求する。M16のRMS再調整は行わず、初期出力・
CEの違いを記録する。Fourier次数とconnectionを同時に変更する比較なので、学習差を
connectionだけ、または実装の高速化だけの効果とは解釈しない。

seed17、tau.003、半径[.5,50]、初期割合0、振幅cap32。通常AdamW LR.0003、CSTの4fieldは
割合を含めすべて32倍。betas(.9,.95)、epsilon1e-8、decay0、全体gradient clip1。
従来20M-tokenのLR/topology horizonとshuffle seed905を維持。2contextのmicrobatch4回で
1024 targets/updateとし、976更新・999424 targetsで停止する。DEV8192 targetsを
0/200/256/400/600/800/976で評価し、confirmation/testで選択しない。TRAIN CE、DEVの
CE/perplexity・次token正答率を記録する。同期した非計測追加のLM全体更新時間・
allocated/reserved peakは、準備・評価・checkpoint・probeと区別する。
DSTのtopology eventはAdam更新に数えず、dense保存の時間を疎kernelの効率とは主張しない。

## 更新予測の診断

実更新1/256/976で最初のTRAIN microbatchの先頭32token行を、block1のup/down
（0始まり）で記録。局所入力、元のタスク余接ベクトル（accumulation1/4を含む）、
更新前後のraw Atom行、native出力を保存する。直列化・再開の事前検証で、hookが次の完全更新と
model・optimizer状態を厳密に維持することを確認する。新しい更新規則は定義しない。

biasと入力Xを固定し、Jvとf(theta+v,X)-f(theta,X)を、raw振幅・radius_fraction・
出力中心・入力中心・全Atom fieldそれぞれで比較する。振幅に誘発される半径変化も維持。
周期上の最短変位、実際のprojection後endpoint、尺度1/.1/.01を用いる。
FP64の物理factorには正確なFP32符号化後の物理写像を渡し、写像の両微分を維持する。
小さな有限差分ではcast/丸めが支配し得るため限界として記録する。scratch layerでnativeの
有限変化も別に計算し、参照との一致を絶対最大2e-4以内または相対L2.01以内で要求する。
結果を見てこの基準を変えない。予測/実変化cosine、実変化/予測norm比、
||実変化-Jv||/||Jv||、局所タスク内積/cosineを記録する。予測ゼロの比率はnull。
タスク方向との一致と予測への忠実さは別の測定であり、局所probeを全損失の因果分解とはしない。

## 実行ゲートと証拠

学習前にconnected envelopeの長方形独立積分Y/dX/全raw勾配、実FFN両方向の全A oracle、
全条件のmodel/optimizer直列化後の次更新一致とhook不変性を検証する。
CSTの各最終状態でも全A oracleを再実施。実モデルの許容差は従来atol1e-4/rtol4e-3、
独立primitiveの3e-4ゲートとは区別する。source・archive・data manifest/order、GPU/runtime、
command・PID、checkpoint、raw曲線/probeを固定・hash化。失敗試行と過去証拠を保全する。
予約済み4090で直列に実行し、追加資源の確保や公開既定値の採用はしない。
各repoのPRを通し、mainと旧実行は変更しない。結果は未取得。

## 学習前の実行上の修正記録

試行01はnative78テストを通過したが、runnerのoptimizer import漏れで検証・学習前に停止。
直接importとGPU不要の起動テストで修正し、ハイパーパラメータ・許容差は変更しない。
試行02ではDense/DST/従来CSTの検証とconnectedの全A数値oracleを通過した後、
次更新のbit一致で停止した（最初の埋め込み最大差7.451e-9）。
別の順序固定direct-level physical/包絡実装で、浮動小数atomicのsite集約を
階層ごとのscratchと固定順のsite加算へ変更する。演算順序の変更であり、profile、
初期化、LR、optimizerは変更しない。試行03でも厳密な再開・hook検証ゲートを維持する。
失敗試行と固定sourceを保全し、いずれの試行にもタスク学習結果はない。
