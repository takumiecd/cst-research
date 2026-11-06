# HIGGSの座標学習率に関する限定pilot

[English (canonical)](higgs-coordinate-lr.md) | 日本語

この別研究consumerは、変更しないtorchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`、そのprivate HIGGS benchmark、同じ
SHA256固定の学習100k・検証100kデータを使う。公開runtimeと固定済みCST/DST比較は
変更しない。有限pilotと条件付き追加検証は実測consumer83で**COMPLETED**となった。
初期のq10改善は一貫して持続せず、stock q1を維持する。
[実観測・audit・失敗attemptの範囲](bounded-validation-results.ja.md)を参照する。
以下は固定した再現protocolであり、研究の自動拡張を指示するものではない。

## 問いと固定予算

結合したPolar成分に対し、座標のLRを変える価値があるかを見る。中間層CST、N1024、
divisor1000（1048 atoms）、batch4、FP32、TF32off、全層学習、同一初期化・提示順を使う。
3条件ともeagerで、stock対照1と座標倍率0.1/10を各3 full epochs、seed17で学習する。
学習は100k、開発評価は検証データ先頭50k。通常層と結合PolarのLRは0.0003、振幅減衰は
0.01のまま。新しいLR grid、構造・密度変更、holdout・公式test参照は行わない。
Graphのexact-type資格条件は維持し、このsubclassを公開Graph optimizerとして認めない。
時間は追加座標計算と標本診断を含むeager研究参照値であり、速度最適化とは主張しない。

## 更新を変える境界

packed atomは **[s,t,出力中心,入力中心]**。s,tは有界振幅とactivityを結合して表し、
そこから物理的なsupport radiusを復号する。この実験は物理振幅とradiusのLRを独立に
調整するものではない。各物理中心μについて、

\[
\hat m_t=m_t/(1-\beta_1^t),\quad
\hat v_t=v_t/(1-\beta_2^t),\qquad
\mu_{t+1}=\operatorname{wrap}_N\!\left(
\mu_t-\eta_\mu\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right),
\]
\[
\eta_\mu=q\eta_{\rm polar},\quad q\in\{0.1,1,10\},\qquad
\eta_{\rm polar}=\eta_{\rm ordinary}=0.0003.
\]

Adamの勾配とmoment漸化式は倍率を掛けず、master時計は1回だけ進める。通常のPolar
retraction/activityと旧振幅減衰には元のLRを使う。座標のraw decayは0であり、物理的な
トーラス座標を維持する。

元のFP32提案を丸めてから変位に倍率を掛ける方式は**採用しない**。物理中心が512付近
では、丸めにより0.1条件の実効移動量が約2倍になることがある。private subclassは、
更新前の座標・勾配・momentと旧時計をcontiguous scratchへ退避し、元のnative AdamW
提案を呼んだ後、更新前の状態から直接の座標LRで再計算する。通常retractionの前に
座標parameterとmomentだけ戻す。fused AdamWへstrided列viewを渡さない。倍率1は
引き算・掛け算・追加copyなしで直ちに元実装へ委譲し、pilot対照には実際のstock classを
使う。stepごとのGPU scalar読み出しは追加しない。checkpoint契約は倍率を保存し、
異なる倍率でのloadを拒否する。

## Admissionと保存する証拠

consumerと外部Git inventory、archive・dataのhash、実際のL4 runtimeを照合する。
実物receiptと結び付いた975資格確認（136 tests、10数値oracle、16 eager/Graph epoch
pairs）はsource・data・runtimeが一致する場合だけ再利用する。変更しないkernelの全
suiteを再実行したり、既存guardを弱めたりはしない。

新しいCPU検証は、独立scalar AdamとPolar/torus retraction、実際のFP32中心ULP、
倍率1/native一致、強制再計算1のoracle、通常層・Polar提案とmomentの不変性、時計1回、
境界通過、保存後再開、constructorと異なる保存済みLRの復元、倍率不一致拒否を含む。
pilot前の短いCUDA eager検証では、直接LR/ULP、contiguous momentと列の完全性、
8-step軌跡と4-step保存/load後の継続を調べる。倍率1はstockのtensor軌跡とも一致させる。
失敗すれば学習を開始しない。

外部runnerは明示したfactory/checkpoint/step計測で再利用し、全benchmark globalを
`finally`で戻す。初期状態、全epoch、開発最良・最終の生model/optimizer/gradient
checkpoint、最良dev予測、曲線、artifact hashを残す。少数の更新前後の生標本と、正しく
名付けた物理出力/入力中心、復号振幅とradius、最短符号付きtorus変位、移動ゼロ率、
時計を保存する。結合radiusの値はタスク学習に伴って変化し得るが、更新則は変えない。

## 事前に決める条件付きの次段階

各条件はdev CE最小のcheckpointを選び、同値なら早いepochとする。AUC条件を満たす
候補では最小CEを優先し、対照とCEが同値なら対照を選ぶ。非対照が対照よりCEを0.001
以上改善し、AUC低下が0.001以内で、有限性・正しさ・再開gateに通る場合だけ進める。
非対照同士の有効CEが完全同値なら小さい倍率を選ぶ。

進める場合は、選択倍率とstock対照を**seed29・43で各10 full epochs**学習し、合計4 runs
とする。元LR・data・構造は変えない。runtime上限に必要ならfresh job1件につき1 seedの
対を実行する。所有者がpilot証拠を検証し、正確なdecisionファイルSHAを渡してから
次段階を提出する。runnerは自動拡張しない。これは開発データを使う限定的な探索再現で
あり、正式な非劣性や新しい未見評価ではない。それ以外は停止し、「この短い予算での
証拠なし」とだけ結論する。座標LRが将来も無効だとは結論しない。

pool scriptは`src/cst_research/experiments/higgs_coordinate_lr_driver.py`。
`--stage pilot`はadmission後に3条件を実行する。`--stage followup`にはさらに
`--seed 29|43`、`--selected-ratio 0.1|10`、検証済み`--pilot-decision-sha256`が必要。
実際のCLIではflagと値を別argumentとして指定する。GPU前にconsumer manifestを
固定し、draft PRを作る。
