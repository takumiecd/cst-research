# M8のAtom数だけを増やす追加比較

[English](lm-m8-capacity-20261110.md) | 日本語

状態は準備段階で、新たな学習結果ではない。ユーザーは通常の非接続M8でAtom数を2倍にし、
容量効果をM16の幾何特性から切り分けることを依頼した。新規実行は全FFN・c16・各map16384 Atomの
1条件のみ。`lm-geometry-epsilon-capacity-3090-20261110-02`の検証済みM8 c8・8192 Atomを対照にする。
同じ4block・幅512・expansion4のLMでFFN8mapを全てCSTとし、attention/embedding/共有語彙出力はDense。
下限や深さのスケーリングの主張ではない。2倍後もA=c√(d_in d_out)を保つ。

通常LR.0003、Atom4fieldのLR32倍、AdamW(.9,.95)、全epsilon1e-8、decay0、clip1を固定。
非接続M8・tau.003・fraction0・半径[.5,50]・振幅上限32・名前別分散初期化・seed17・shuffle905・
FP32/TF32無効も固定する。Atom増加時も同じ目標分散を使い、旧Atomを複製しない。
通常重みのbyte一致、c8 raw初期値の旧M8との一致、c16 raw初期値の旧c16 M16族との一致を確認する。
最後の一致はraw値であり、関数が同じという意味ではない。

TinyStories GPT2のmanifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`を共有。
976更新・999424 target、20M schedule horizon、2context×4microbatch、context128、DEV8192 target、
confirmation未使用。評価0/200/256/400/600/800/976、固定TRAIN4箇所のprobeを0/256/976で実施。
半径/fraction・層ごとの出力/cotangent/field勾配・moment尺度・checkpointを保存する。
先頭10更新を除く同期した完全stepとallocated/reserved peakも記録する。

既存の同GPU対照を明示したread-only入力として使う：Dense、DST-c8/c16、旧M8-c8、epsilon1e-8の
M16-c8/c16。manifest/監査/source/data/通常・raw初期値/active scalar予算を検証して再利用する。
DSTはmapごと4A active weightと以前選択・転用したsparse LR.003で、c16で新たに調整しない。
Dense保存である。旧対照と新実行を区別し、別job間の速度は探索的とし、新しい対応付き速度主張にはしない。
描画に明示的な`--dst-run`を要求する。完了済み対照を再学習しない。

runtimeは7a7dc11689d6905f905965ecdac3b41a839548e8（[PR163](https://github.com/takumiecd/torchcst/pull/163)）。
c16のmetadata/envelope adapterとbenchmark登録のみ追加し、CUDAを含む既存runtime sourceは全て
95f647fとbyte一致する。元のc8制限も変更しない。研究用の学習loopにmodel/optimizer factoryを渡せる
ようにするが、step/data/probeの数学は変更しない。両sourceを実行前に固定し、実行後もhash検証する。

学習前に実機3090/24GB/CC8.6を確認し、c16のretained/multiple forwardを含む関連native/回帰96件、
初期up/downの全A FP64参照を通す。Y/dX/raw各fieldそれぞれmaxabs≤3e-4かつrelativeL2≤1e-4を要求し、
旧component基準の違反数も保持する。基準は緩和しない。保存後次更新とobserverの完全一致、
epsilon/LR/raw/通常重み/予算も必須。終端全A oracle、有限FP32 moment・時計976、256/976 checkpointの
監査と終端DEV再現を要求し、失敗証拠も変更せず保全する。

入口は`cst_research.experiments.lm_m8_capacity_run`のqualify/train/audit。
正確なdata SHA・`--previous-run`・qualification receipt・未使用outputを渡す。
1seed短期の改善はこの条件での容量効果を支持するが、多層化の主因や十分なAtom数の証明にはならない。
短期改善がなくても容量の限界を確定したことにはならない。

## 停止した実行と主比較基準の変更

最初の実行は旧C++ c8 guardで学習前に停止した（8失敗・88合格）。失敗compact f99dd0bf295ee5533003b351d88add72f244ba984dbe62718d47a9a5396a1bfdを保全し5payloadを検証済み。その後ユーザーはDSTのマスク1の数=Atom数を主比較とし、dense-mask保存を維持しCOO/CSR化しないと指定した。[active-count protocol](lm-m8-active-count-20261110.ja.md)で継続し、旧4A DSTは主比較ではなく補助にする。
