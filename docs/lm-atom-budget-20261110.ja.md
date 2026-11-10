# Denseに近づくM8 Atom予算探索: 手順

[English](lm-atom-budget-20261110.md) | 日本語

準備段階で未実行。所有者はM16だけでなくAtom数がタスク差の原因だと考え、確保したRTX4090でM8の予算探索を指示した。
未mergeのM16実装PR159〜162はcloseし、branch/結果は保全した。同予算のM8/M16結果は証拠として残すが、
唯一の原因の証明とはしない。新しいM16実行は選択しない。

## 固定タスク

4block・幅512・head8・context128・expansion4、全8FFN mapをCST/DSTとする。
attention/embedding/共有語彙出力はDense。seed17・shuffle905・976更新/999424学習target・20M schedule・DEV8192 target。
非接続M8・envelope tau.003・fraction0・半径[.5,50]・振幅上限32・通常LR.0003・CST全field32倍、
AdamW(.9,.95)・epsilon1e-8・decay0・clip1・FP32/TF32無効を固定。
初期化方針は同じだが、envelopeが振幅に依存するためAで平均初期半径は変わり得る。

実機4090/24GB/CC8.9でDenseとM8 c16 anchorを新規測定する。旧3090は履歴対照であり、paired速度基準ではない。
データmanifest5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01を固定し、confirmation/testは未使用。
既存の共通学習・checkpoint bodyを明示的factoryでそのまま再利用する。通常初期化は監査済み旧対照と一致。
private AtomBudgetConfigで通常設定をModelConfigにより検証し、このpilotの予算だけ許可する。通常config guardは変えない。

## 事前に定める適応的pilot

A=c√(d_in d_out)。coarse c16→32→64→128は各Linearの16384→32768→65536→131072 Atom。
訪問する各予算でCSTと、dense-mask DSTのsparse LR.003/.01/.03を事前検証・学習・監査する。
DSTのマスク1をAと揃え、4Aにはしない。密重み+boolマスクを維持し、COO/CSRは使わない。
同数のLR間で初期mask/値を一致させ、行予算を保つRigL型更新250/500/750を使う。
実Adam時計はDST973対CST976。DSTは終端DEV CE最小・同値なら小LRを選び、
全候補を保全しDEV選択と明記する。grid端の候補が最良かも記す。

記述的なDense近傍の基準を、終端DEV CE≤新Dense CE+.01、かつ次token Top1≥新Dense Top1−.005（0.5ポイント）とする。
CE基準はPPL約1%以内に対応するが、統計的同等性や収束の検証ではない。
最初の合格でcoarse拡張を止め、c16より上なら直前不合格と最初の合格の中間を1つだけ追加する（c24/48/96）。
試した中で最小の合格CST予算を選ぶ。それ以上の二分探索やAtom下限の主張はしない。
c128まで合格がなければ、この上限内で適合する予算なしと報告する。
曲線を見た後のDEV探索として設計し、独立confirmationとはしない。
候補後の長期/複数seed確認は別段階であり、ここでは実施済みと主張しない。

## 検証と証拠

libraryはtorchcst PR164のprivate非接続M8予算routeを使い、公開選択/中央fixture認定とはしない。
数式・optimizer・旧c8/c16 admissionは不変。RADIUS_LM_BUDGET host guardだけを除いてsource比較し、
他の既存runtime、共有research module、train/audit bodyのbyte一致を確認する。
nativeのretained・複数forward・非連続入力を学習前に検証する。
各段階で全A・FP64のup/down Y/dX/raw各fieldでmaxabs≤3e-4かつrelativeL2≤1e-4を要求し、
旧component違反数も保全する。保存/observer後の次更新完全一致、有限FP32 moment/時計、
checkpoint/DEV再現、明示的有効数監査を要求。許容差を緩和せず、失敗時は停止してsource/結果を保全する。

同期した完全stepは4microbatch・clip・optimizer/topologyを含め、先頭10更新・入力準備・評価・probe・checkpoint・oracleを除外。
allocated/reservedは別に記録し、reservedにはobserver履歴が残る。層出力/cotangent/特徴、幾何/moment、
256/976のraw checkpointと0/256/976の固定TRAIN4箇所probeを保存する。
実行した全段階・DST全LR候補・source/data/result hash・全適応判断・失敗を保持し、固定snapshotは変更しない。
所有command完了後も予約Hubは維持する。merge/既定採用や未測定の速度主張は行わない。

固定前検証: library CPU2288合格/2677任意skip、research CPU241合格/61skip。Ruff/registry/diffも合格。最初のresearch全件実行はPYTHONPATHにscriptsがなく14 import errorだったが、コードを変えずに実行経路を直して全合格した。

実行記録: joint snapshot e77a8b6ed48d24a8dc44600f3f6459d44774391194b44b858665c27d5abd5a07（追跡2237file）、consumer9b9e2eb1e4579fe035ca2442b36bad99f1df029d/library0bed48655b1d988e0c59b4442005aaec8c463a31を実機確認済み4090で実行中。native前にruntime278fileの一致、通常guard/packed AST、共有学習/監査body一致に合格した。作業checkoutに後処理/記録を追加しても実行snapshotは不変。
