# FFNの長期比較

[English](ffn-long-horizon-20261109.md) | 日本語

976ステップでは損失がまだ低下中だったため、固定したDense・DST・CSTを20M tokens
まで延長して学習経過を観察する。新たなLR・tau探索はしない。seed17の探索的なDEV
評価であり、独立した精度確認ではない。RTX4090で検査を通してから学習する。
準備完了は実験結果を意味しない。

## 固定条件

4ブロック・幅512・8ヘッド・文脈128・FFN拡張4。8個のFFN写像のみを置換し、
attention・embedding・共有語彙headはDenseのまま。同じハッシュで指定した
TinyStoriesデータ・通常パラメータ初期値・seed17・窓のシャッフル順を使う。
新規初期化から19531ステップ（19999744 tokens）、microbatch2・蓄積4で学習。
256・976・4882・9765・19531でcheckpointを保存し、200ステップごとと各節目で
8192 tokensのDEV CEを測る。confirmation評価・RMS補正・seed追加・再調整・CEによる早期停止は行わない。

- Dense：通常パラメータの最大LR0.0003。
- CST：公開moments初期化、tau0.003、初期radius_fraction0、振幅上限32、
  半径0.5..50、levels_wide_amplitude、写像あたりA8192。
  全4 atomフィールドのLRを32倍。初期出力RMS合わせはしない。
- DST：行ごとに均衡したRigL型のdense保存参照、写像あたりK32768（4A）、
  疎重みの最大LR0.003は以前の短期DEV調整で選んだ値に固定。通常LR0.0003。
  CSTとDSTの有効FFNスカラー数はbias・残差スカラー込みで272384。

共通でFP32・TF32無効、Adam betas(0.9,0.95)、eps1e-8、weight decay0、clip norm1、
warmup200、20M-token cosineスケジュール。DSTは19531ステップの75%まで250ごと、
250..14500の58回で接続更新する。そのステップは全パラメータのAdam更新を置き換え、
最終Adam時計は19473。旧短期実験の接続更新250・500のみの状態からは再開しない。

## 実行・証拠

同じ実機RTX4090でDense・DST・CSTを順番に実行。実行前にoperatorとconsumerを固定。
68件のoperator検査、CSTモデルの検査（optimizer保存・再開の一致を含む）、
DSTの通常・接続更新・更新期間終了後の分岐検査を実施し、失敗した場合も証拠を残す。
データ不要のTrainingPlan検査でhorizon・接続更新・checkpoint時計・prefix整合を固定。
3条件すべてでシャッフル順のハッシュを記録する。

同期した完全学習ステップ時間とallocated/reserved peak memoryを測り、準備時間と
総時間は別記する。両runnerでmicrobatch間の返却logitsの寿命を統一した。
旧短期DSTは返却値を直ちに解放していたため、旧速度・メモリは今回の対照ではない。
DSTはdense保存・matmulによる精度参照であり、疎kernelの効率の証明には使えない。
DEV評価・checkpoint・幾何診断はステップ計時に含めず、CST診断は総時間に含まれる。
tokens軸・経過時間軸の曲線を残す。

15個の異なる節目checkpointを監査：optimizer時計・moment・DEV再現・source/config/data一致、
学習済みCST/DST写像の出力・入力勾配・全パラメータ勾配を独立FP64 oracleで検査。
最終aliasを含む18個のcheckpointファイル名のハッシュを残す。ログ・receipt・sourceの
compact archiveをハッシュ検証して回収し、大きなcheckpointはGPU側に残す。
実機・runtime・検証結果・制限は別の結果文書に記録する。
証拠：CST_RESEARCH_EVIDENCE_ROOT/ffn-long-horizon-4090-20261109-01。

## 追加のトークン正答率

Dense学習中の所有者の依頼により、同じ8192 DEV targetsでteacher forcingによる
次トークンtop1正答率を15個すべての節目checkpointから追加集計する。
source固定後の追加評価であり、従来のCE・perplexityを補足する。学習・主指標・
選択・confirmationの扱いは変えない。ステップ計時に干渉しないよう、全学習と
学習済み監査の後に、別途ハッシュを残したevaluatorを実行し、CEが元ログと一致することも確認。
トークン正答率は文全体の完全一致率や自由生成の品質ではない。
DEVは学習とstoryが重複しないが、8192 targets・1 seedの小さな開発実験である。
計測した学習ステップ累積時間と実経過時間を分けて報告する。
Dense/CSTは200ステップごとにcheckpointを保存する一方DSTは節目のみであり、
CSTには幾何observerもある。このためwall曲線は診断用で、疎演算の公平な効率比較ではない。
