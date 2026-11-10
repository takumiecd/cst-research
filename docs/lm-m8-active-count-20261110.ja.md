# 有効数を揃えたマスクDSTとのM8容量比較

[English](lm-m8-active-count-20261110.md) | 日本語

準備段階で結果ではない。ユーザー指定の主比較は **CSTのAtom数A = 各LinearのDSTのマスク1の数**。
DSTは既存の密な重み+boolマスクを維持し、COO/CSR化しない。
raw変数数4Aを揃えた比較は別の補助であり、以前の結果は元の基準のまま保全する。

新規条件はM8 c16（各map16384 Atom）、およびDST各map8192/16384有効位置×sparse LR.001/.003/.01。
DSTは各有効数で終端DEV CE最小、同値なら小さいLRを選ぶ。全候補を保持しDEV選択と明示する。
confirmationは使わず、旧4A用に選んだLRをnnz=Aの調整済み証拠として流用しない。

`lm-geometry-epsilon-capacity-3090-20261110-02`の監査済みDenseとM8 c8（8192 Atom）をread-only対照として
再利用する。旧4A DSTは全て補助。M8の8192 Atom対DSTマスク1が8192、M8の16384対DST16384で比較する。
CSTのraw値は引き続き4A（振幅/fraction/座標2個）であり、raw変数数ではなく有効な表現単位を揃える。
再成長直後の値0の重みもマスクが1なら有効数に数える。数値が非ゼロかではなくマスク1の数が基準である。

4block・幅512・FFN512→2048→512の8mapをCST/DSTとしattention/embedding/共有語彙出力はDense。
非接続M8・tau.003・fraction0・半径[.5,50]・振幅上限32・通常LR.0003・CST4field LR32倍・
全epsilon1e-8・AdamW(.9,.95)・decay0・clip1・seed17・shuffle905・FP32/TF32無効を固定する。
DSTは既存の行予算を保つRigL型で250/500/750に更新し、active-fan分散補正、行/全体のmask数を保つ。
DSTの同じ有効数のLR候補間で初期mask/重みを一致させ、通常重みは旧対照と一致させる。
CST増加時は同じ目標分散・seedの初期化族を使い、Atomを複製しない。c8 rawの旧M8一致、
c16 rawの旧c16族一致、sourceとloopの同一性を確認する。

データmanifest5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01。
976更新・999424 target、20M schedule、context128、2context×4microbatch、DEV8192 target。
評価0/200/256/400/600/800/976。固定TRAIN4箇所を0/256/976でprobeし、層出力/cotangent/勾配、
幾何/momentとcheckpointを残す。confirmationは未使用。実Adam時計はCST976、DST973。

runtime修正では通常c8 binaryのguardを保ち、専用RADIUS_LM_C16 buildとextension-binding hookを追加する。
既存runtime fileの差はpacked_rows.cuhの条件付きhost guardとpacked.pyのextension bindingだけ。
既定preprocessing/ASTの同一性、数値kernel/初期化/optimizerの不変を確認する。拒否された旧sourceは固定保全。
学習前にnative/回帰98件、実機3090/24GB/CC8.6、初期up/down全AのFP64 Y/dX/raw各fieldを確認する。
各maxabs≤3e-4かつrelativeL2≤1e-4を要求し、旧component違反数も保持して基準を緩和しない。
新7条件すべてで保存後次更新とobserverの完全一致を要求する。M8終端の全A gate、有限FP32 moment、
時計・topology不変条件と256/976 checkpoint/DEV監査も必須。

速度/メモリは実際のdense-mask実装を対象とし、密重み・mask・dense-gradient buffer・optimizer状態を含む。
準備/probe/先頭10更新を除いた完全stepのallocated/reserved peakと時間を記録する。
これをCOO/CSR実装の値や最適化された疎保存全般の証明として扱わない。旧/新jobの速度比較は探索的。
source/data/result hashと失敗証拠を保全し、公開/既定値採用は行わない。

入口はcst_research.experiments.lm_m8_active_count_runのqualify/train/audit。
描画に明示的な旧対照と--dst-run、有効数検証を要求する。主曲線には選択DSTを、別HPO重ね図には
全LR候補を出す。1seed短期・DEV選択なので容量比較は探索的であり、十分なAtom数・収束・同等性・
多層化の主因を確定しない。Atom増加で改善しても、それだけで多層化が悪化する原因の証明にはならない。
