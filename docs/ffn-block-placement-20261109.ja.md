# 各ブロックへのCST分散とFFN内の合成

[English](ffn-block-placement-20261109.md) | 日本語

所有者の指示によりDense教師・重み再現の案は取り下げる。目的は本来の言語モデルの
正解トークンでタスクを学習できるか。Denseは比較対照として使う。
HIGGSでも単層/複数層の差はあったが、タスク・正規化・初期化が変わった今回に
同じ原因を仮定しない。

同じ4ブロック・幅512のTinyStories LM、通常層初期値、seed17、5M-token prefix、
20M LR/接続更新horizon、CST写像あたりA8192・LR32倍・tau.003/fraction0/cap32、
DST写像あたりK32768・疎LR.003を維持する。追加HPO・confirmation・RMS補正・
層の追加はしない。置換で写像の形状を変えない。

元の固定site-count cohortは上り1個・下り1個・1/2ブロックの上下両方・全4ブロックの
上下両方を含む。別研究として次の6条件を新規初期化から追加する。

| 配置（0始まり） | FFN構成 | 写像数 | 方式 |
|---|---|---:|---|
| scattered_up | 全ブロック：CST→GELU→Dense |4|CST/DST|
| scattered_down | 全ブロック：Dense→GELU→CST |4|CST/DST|
| scattered_mixed | block0/2はup、block1/3はdown |4|CST/DST|

1個だけの条件と各ブロック1個を比較し、それらと全ブロックの上下両方も比較する。
さらにscattered_mixedとblock1/2の上下両方を比較する。両方4写像・up2/down2・
有効スカラー総数が一致し、個数を揃えた分散/FFN内連続の診断になる。
各配置のCST/DSTは同じ有効FFNスカラー予算。4個と8個の間では全体圧縮率が変わる。

本来のタスクラベルで新規の名前別初期値から学習する。TRAIN/DEV CE・perplexity・DEV次token top1、
固定TRAINの値/出力cotangent/field勾配、clip前norm、実際の固定ランダム入力に対する更新、
振幅/半径/fraction、有効field LRを残す。初期/保存再開/oracle/診断非干渉検査と全18 checkpoint
監査はsite-count runnerを使う。68件のoperator検査を再利用するのはoperator全bytes・runtime・
GPU・ログ一致を確認した場合のみ。前cohortのGPU処理終了後に実行し、稼働中の固定sourceは変更しない。

以前の検証済み対照とsource core/model/data/runtime/order/seed/budgetの一致を確認する。
数式sourceの差はsite catalog追記と明示qualification family選択だけで、既存条件の初期値はテストで保護。
両source archiveを保存し、旧固定familyへの事後追加ではなく所有者による新研究として記録する。
永続NFSの証拠と不変の途中compact archiveをローカルへ回収・ハッシュ検証する。

同じ学習token数でタスク性能を比較したうえで、分散と連続の値・cotangent・実更新推移を照合する。
単層の差が小さく、連続時に差が増えれば合成の分析に進む根拠になるが、勾配消失やkernel不具合の
証明ではない。mixed比較も方向の個数は揃うが位置が変わる。
位置の反転、LR/gain介入、seed追加は次段階として別記録し、自動追加しない。
証拠：CST_RESEARCH_EVIDENCE_ROOT/ffn-block-placement-4090-20261109-01。
