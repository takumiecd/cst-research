# 幅2048で Atom 総数を揃えた Q・K と V の配置比較

[English](lm-qk-v-equal-budget-2048-20261118.md) | 日本語

RTX 4090 で2種類の Atom 総数に対し、Q・K だけを置換した条件と V だけを
置換した条件を比べる。[部分置換の比較](lm-qk-v-partial-2048-20261118.ja.md)の V c64 と
Q・K 各 c48 を使い、Q・K 各 c32 と V c96 を seed 17・23・42 で4,882更新まで
追加学習する。各配置は全4ブロックで対象を置換し、残りの Dense 行をビット単位で
同じ値から始める。

| モデル全体の Atom 数 | Q / K | V | 新規条件 |
| ---: | --- | --- | --- |
| 524,288 | 32 / 32 | 64 | Q・K |
| 786,432 | 48 / 48 | 96 | V |

前回の TinyStories モデル、バッチ順、オプティマイザ、FP32 半径デコードを
使い、新しい Atom 数を幅拡張の対応表に追加する。全 Dense 初期値を照合する。
先頭射影の出力・入力勾配・Atom 勾配を独立参照と比較し、全置換射影の初回・
最終 q 更新を確認する。予算ごとに、検証 CE と別ストーリーの確認 CE の
Q・K 条件から V 条件を引いた差を各 seed と平均で報告する。同期した全更新時間、
アロケータのピーク、前回ジョブから参照した条件も記録する。

幅4096の予備比較には、2予算・3 seed の平均確認 CE が低い配置を選び、
同値なら平均更新時間が短いほうを選ぶ。その配置の2通りの射影係数で比較する。

## 再現方法

登録したソースを SM89 用にビルドし、TORCH_EXTENSIONS_DIR を指定する。
REFERENCE_RUN は元の有理式射影アーカイブ、DENSE_ANCHOR は元の幅2048の集計。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qk_v_equal_budget.py --reference-run "$REFERENCE_RUN" \
  --dense-anchor "$DENSE_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力と出力先に --resume を追加する。
