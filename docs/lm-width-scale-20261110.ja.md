# c64固定M8 FFNの幅・学習量パイロット

[English (canonical)](lm-width-scale-20261110.md) | 日本語

## 問いと固定条件

各写像を A=64 sqrt(d_in d_out) として、幅を増やすとDenseとのvalidation精度差が
縮むか、長く学習すると差がどう変わるかを調べる。Atom下限の研究ではなく、
タスク上の実現可能性を調べる探索実験である。

| 幅 | FFN up/downの次元 | 各写像のAtom数／DSTマスク1の数 | Atom密度 |
| --- | --- | ---: | ---: |
| 512 | 512↔2048 | 65,536 | 6.25% |
| 1024 | 1024↔4096 | 131,072 | 3.125% |
| 2048 | 2048↔8192 | 262,144 | 1.5625% |

pre-LNデコーダ4ブロック、8 heads、context128、GPT2語彙50257、FFNの8写像を
すべて置換する。Attention、埋め込み、正規化、bias、埋め込みと共有する語彙出力は
通常のDense。次トークンの正解ラベルで学習し、Dense行列を教師にしない。
seed17、名前ごとの初期化、shuffle905、1更新1024 targets（2文脈×4蓄積）、
FP32、TF32無効を固定する。

物語単位で分離済みのTinyStories manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`
を再利用する。TRAINは20,000,001 tokens。DEVの8192次トークンtargetsはvalidationで、
model.eval()/no_grad()で評価する。confirmation32768 targetsの性能は評価しない。
選択や仮説はDEVに基づくため、形式的同等性や独立テスト結果とはしない。

M8、物理spacing1、radius-envelope tau.003、fraction0、radius[.5,50]、振幅上限32。
幅512の前実験と同じ分散初期化を使う。通常重みの標準偏差.02、残差出力写像は
sqrt(2*layers)で割る。この固定レシピは幅不変のHe初期化でも、容量だけを変える
介入でもない。幅とAtom数により出力・勾配・初期半径の尺度も変わり得る。
CST通常peak LR.0003、4フィールド32倍、AdamW(.9,.95)、eps1e-8、weight decay0、
global clip1。fractionフィールドの既存LR方針も維持する。本実験ではカーネル再設計や
CSTのLR探索は行わない。

## 短期選択と長期継続

各幅でDense LR {.0001,.0003,.001}、CSTc64、Dense保存DSTの疎重みLR
{.0003,.001,.003,.01}を新規に学習する。DSTの通常パラメータのpeak LRは.0003。
選択されたDenseのLRがCST/DSTの通常LRと一致するとは限らない。疎重みLRは別に
探索する。全候補を残し、範囲端で選ばれた場合も記載する。最適LRの特定とはしない。
DSTはDense重み＋BooleanマスクのままとしCOO/CSRへ変換しない。各写像で
マスク1の個数=Aとし、再成長直後の値0も数える。rawスカラー数と実測メモリは別記する。
行ごとの有効数を保つRigL型250更新ごとの配線変更、active-fan初期分散、残存moment、
成長時の重み0・moment reset規則は維持する。

短期は976更新=999,424学習targets。各幅で最終DEV CE最小のDenseとDSTを選び、
同値なら小さいLRを採る。**3つの幅すべて**で、選択した対照2群とCSTを4882更新
=4,999,168 targetsまで延長する。短期976のモデル・optimizer・RNGを復元し、
同じデータ順、20M-token cosine horizon、DSTの20M-token配線horizonで継続する。
warmup、LR、配線スケジュールを継続時にリセットしない。長期LRは短期で選んだ値を
固定し、5M時点で再最適化したとは扱わない。

DEV評価は0/200/256/400/600/800/976、その後1200から400更新ごと、および4882。
checkpointは256/976/2441/4882。step計時外の固定TRAIN4文脈で、各層の出力・勾配RMS、
特徴量・残差統計を記録し、offset0のtensorも保存する。層ごとの振幅・半径・fractionと
Adam尺度も記録する。

学習targetsに対するDense/CST/DSTのvalidation CE、PPL、次トークンTop1を重ねる。
TRAIN曲線は分離し、短期LR全候補を残す。同期したcomplete-step時間、peak allocated/
reservedを区別し、評価・probe・checkpoint/oracle I/Oと各プロセス区間の最初10stepを
計時から除外する。CE+.01、Top1−.5ポイントは記述上の目安として残す。
1 seedとDEV選択から収束・同等性・普遍的スケーリング則は主張しない。

## 実行前検査・保存・実行

所有者が確保した4090を1台使い、実CC8.9とメモリを記録する。追加GPUは確保しない。
専用経路は上記6組の方向つきshape/countとrows1..256だけを許可する。
既存ordinary/c16/budget/公開dispatchは変更しない。数値カーネルは共有し再調整しない。
metadata往復、nativeの非連続入力・retain backward、全Aの独立FP64 Y/dX/4フィールド
oracleを初期・短期後・長期後に確認する。各tensor/fieldでmaxabs<=3e-4かつ
relativeL2<=1e-4を固定し、元の成分別違反数も残す。診断probeの不変性とserialize後の
次更新完全一致、checkpoint hash、Adam時計、有限FP32 moments、DSTのmask/0/個数、
最終digest、復元validationを監査する。

リモート実行前に両repoのtracked sourceとprotocolを凍結する。raw checkpoint/tensorは
リモート保存し、compact証拠をhash確認して取得する。検査失敗時は停止して失敗を残す。
許容誤差の緩和や別演算への暗黙fallbackはしない。各arm16時間を安全上限とする。
全幅の短期を先に完了し、その後長期へ進む。途中曲線は未完了と明記する。
公開defaultの採用やmain mergeは含まない。

Runner: `python -m cst_research.experiments.lm_width_run --help`。
Plotter: `python scripts/report_lm_width.py --run RUN --output PLOTS`。
registryとignoredの外部証拠locatorにsource commit、archive hash、remote directory、
永続driver PIDを記録する。既存実験は凍結したまま残す。
