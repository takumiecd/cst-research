# 全CST Attention・FFN：機能の事前検証

[English](attention-output-20261108-admission.md) | 日本語

幅512、pre-LayerNorm4block、8head、FP32/TF32無効、A=ceil(8√(d_in d_out))で
機能の事前検証に合格した。収束・精度同等の結果ではない。token embeddingと共有語彙出力、
学習位置embeddingはDenseを維持。Attentionの内積・mask・softmaxは通常の演算を保つ。

| 構成 | CST射影数 | 全演算VJP | 因果性 | モデル更新・再開 |
| --- | ---: | --- | --- | --- |
| FFN | 8 | PASS | PASS | PASS |
| QKV | 12 | PASS | PASS | PASS |
| QKV+FFN | 20 | PASS | PASS | PASS |
| QKVO | 16 | PASS | PASS | PASS |
| QKVO+FFN | 24 | PASS | PASS | PASS |
| QKVO+FFN、stream64 | 24 | PASS | PASS | PASS |

独立CPU FP64の全Atom比較は、実256行の正方・up/down・O演算を含む。
各caseの最大絶対差はY1.35e-6、dX2.27e-6、全Atom勾配2.69e-5で、既存の要素ごとの
atol3e-5/rtol3e-4基準で合格。別途Gauss積分でprofileの数式も確認した。
手計算masked AttentionとY/dX/QKVO勾配が一致し、未来token勾配は厳密0。
全CST siteが有限非0の勾配を受け、実2×128tokenのモデル更新で変化する。
保存後の次タスク更新でParameter最大差0、同一勾配optimizer/moment/clock再開も厳密一致。
これは少数更新の事前検証であり、予定する976更新の学習比較ではない。

数値torchcstは`df24415afe442b9916e902d0f590fa5207016f66`、検証consumerは
`73751439a7643989adaab10889bdf6d547e67e3d`、RTX4090/Torch2.6.0+cu126。
回帰campaignはscope/CUDA test223件に合格、skip0。試行03 archive SHA256は
`e1ebb4914dd52adfb1c493a25c9c53b4ba453735812264a796641b51c0304103`。
27原本を照合し、当該試行の所有PID10個の終了も確認した。
[集計JSON](attention-output-20261108-admission.json)に指標を記録。
元source/logと失敗は外部証拠`attention-output-20261108-02`/`-03`に保全。
符号付き座標への修正と以前の失敗はtorchcst研究記録・PR155に残した。

試行03はreceiptに既定のclosed-envelope powerが欠け、モデル構築前に学習開始を拒否。
consumer91ef5a6で全constructor設定を明記し、生成側と照合側の回帰を追加した。
物理設定・数値基準は不変。試行04はbyte同一・hash検証済みoperator source/buildと
回帰証拠を使い、consumer検証を再実行。levelsの5条件と厳密な学習器照合は合格し、
最後に観測したのはstream64検証。その後の学習状態は未確認。
実行接続ReadTimeoutの後、HubがHTTP503を返し、VPNは画面上で接続済みだった。
耐切断controller・serverの停止や再起動はしていない。
学習曲線・比較CE/PPL・速度・メモリの結果はまだ主張しない。
監視の制限を記録し、研究状態はRUNNINGのままとする。

初期最適化PR154・計画PR34はmerge済み。今回の実装・証拠はtorchcst PR155と
cst-research PR35で、全実行・レビューの完了待ちである。
