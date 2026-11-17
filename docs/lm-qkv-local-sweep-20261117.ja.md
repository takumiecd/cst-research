# 幅2048 QKV の c64 周辺の Atom 数比較

[English](lm-qkv-local-sweep-20261117.md) | 日本語

均等配分の QKV c48・c56・c72・c80 を、
[幅2048の前回比較](lm-qkv-2048-20261117.ja.md)の同じ seed の c64・全 Dense と比べる。
4条件を seed 17・23・42 で初期状態から4,882更新まで学習する。c64 の seed 17 も
GPU 初期化で再走する。新規は合計13走。4ブロックの Q・K・V を置換する。

| c | 射影あたりの Atom 数 | Atom 数／Dense 重み数 |
| ---: | ---: | ---: |
| 48 | 98,304 | 2.344% |
| 56 | 114,688 | 2.734% |
| 64 | 131,072 | 3.125% |
| 72 | 147,456 | 3.516% |
| 80 | 163,840 | 3.906% |

幅2048、8ヘッド、文脈128、FFN倍率4。前回の TinyStories GPT-2 トークン、
データ順 seed 905、1更新1,024ターゲットを使う。学習率の基準は通常0.0001、
Atom 0.0096で、19,531更新のスケジュール、半径、tau も前回と共通にする。
各 seed の全 Dense の初期値を前回のハッシュで照合し、seed 17 の c64 再走では
初期値全体を照合する。GPU 初期化は CPU の乱数列と FP64 の半径・タイル選択を使い、
エネルギーを GPU FP32 で計算する。

学習前に、新しい Atom 数それぞれの seed 17 先頭 Q で CPU・GPU 初期値のビット一致を
調べる。独立 CPU 参照と出力・入力勾配・Atom の4勾配を比較し、全12射影の初回 q 更新を
確認する。各学習の初回と最終の q 更新も確認する。

13走すべてを集計する。主指標は同 seed の c64 に対する検証 CE 差。ストーリーを
分離した確認 CE と検証 top-1 も測る。4条件・3 seed の結果および seed 17 の c64
再走との差を報告する。同期済み eager 全更新時間と PyTorch のピークメモリは
前回と同じ方法で測る。条件順は runner で事前固定する。

## 再現方法

前回の幅2048のソースに、登録済み結合マニフェストに従って torchcst の変更を重ねる。
元の有理式射影のアーカイブ、凍結データ、FP32 アダプタを再利用する。SM89 用に
幅拡張をビルドし、TORCH_EXTENSIONS_DIR を指定する。PAIR_ANCHOR は前回の
evidence/summary.json を指す。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_local_sweep.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力と出力先に --resume を追加する。
