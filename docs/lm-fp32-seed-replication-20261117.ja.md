# FP32 半径デコードによる TinyStories FFN down の seed 追加比較

[English](lm-fp32-seed-replication-20261117.md) | 日本語

幅1024、4ブロック、8ヘッド、文脈長128、FFN倍率4に固定する。
各条件は初期状態から1更新あたり1,024ターゲットで4,882更新を学習する。
オプティマイザ、19,531更新の学習率スケジュール、q の初期値、Atom 数、
データ順は [seed 17 の置換比較](lm-rational-projection-20261116.ja.md) に従う。

seed 23 と 42 で全 Dense と、4ブロックの FFN down をそれぞれ196,608
Atom に置換する条件を比較する。各ペアで置換箇所以外の Dense 初期値と
ミニバッチ順を共有する。全 seed が保存済みのデータ順 seed 905 を使う。
seed 17 は新実装で FFN down 条件を再走し、初期値と結果のハッシュを照合
した前回の全 Dense と FFN down を比較に使う。実行順は seed 23 の
全 Dense、down、seed 42 の down、全 Dense、最後に seed 17 の down。

チェックポイント計測時の研究用 FP32 半径デコードと重み共有アダプタを
使う。Fourier 重み生成と振幅上限の CPU 丸めは、torchcst PR #168 を
kernel/lm-rational-projection-admission にマージしたコミットを使う。
各更新で4層の重みを一度だけ組み立て、マイクロバッチ間で重み勾配を
加算してから Atom の VJP を求める。評価も同じ FP32 半径デコードと重み
生成を使う。長い学習の前に seed 23 の初期モデルで独立した FP64 因子
計算と初回の q 更新を照合する。全 FFN down 条件で初回と最終の q 更新を
記録する。

主指標は step 4882 の検証 CE（8,192ターゲット）のペア差（FFN down から
全 Dense を引く）とする。ストーリーを分離した固定の確認セット
（32,768ターゲット）の CE も測る。検証 top-1、途中の学習曲線、更新時間、
ピーク allocated と reserved を記録する。条件を固定し、seed ごとの差を
記述する。

`REFERENCE_RUN` には trainer・結果・ソースを含む保存済みディレクトリ
`lm-rational-projection-20261116-02` を指定する。`SNAPSHOT` には保存済み
ソースの独立コピーを使い、`torchcst` と当時の `cst-research` を並べる。
データのパスは参照マニフェストに従う。
`31f40e64241eb866e95305401b94cc8da4ac5a7e` から
`6703a18a55cd949564be8a0ceca1ca1b813f7fc5` への torchcst の変更ファイルを
重ね、SHA-256 を記録したマニフェストを渡す。PyTorch 2.9.1+cu128 の環境で
そのソースから拡張をビルドするか、同じソースハッシュで検証済みのビルドを使う。
runner はデータ、ソース、ネイティブバイナリの識別情報、実行の識別情報、
復元したチェックポイントを検証する。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$EXTENSIONS"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python - <<'PY'
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import width_extension

width_extension()
PY
python scripts/run_lm_fp32_seed_replication.py \
  --reference-run "$REFERENCE_RUN" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力に `--resume` を追加し、最後に記録したチェックポイントから
続ける。実験登録に計測結果の保存先を記載した。

## RTX 4090 の結果

新規5走が4,882更新を完了し、最終チェックポイントを復元して検証 CE と
top-1 を再確認した。seed 17 の全 Dense は前回の実行を使う。FFN down の
再走と前回の down は初期パラメータのハッシュが一致した。CE の差は負値が
FFN down 優位を表す。

| Seed | 全 Dense 検証 CE | FFN down 検証 CE | 差 | 全 Dense 確認 CE | FFN down 確認 CE | 差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 2.53073 | 2.51144 | -0.01929 | 2.45098 | 2.44584 | -0.00514 |
| 42 | 2.50373 | 2.52609 | +0.02235 | 2.43854 | 2.44764 | +0.00910 |
| 17（全 Dense は前回） | 2.51733 | 2.50175 | -0.01559 | 2.44804 | 2.43966 | -0.00838 |

追加した2 seed は、検証・確認ともに1勝1敗だった。追加2 seed の平均ペア差は
検証で +0.00153、確認で +0.00198。seed 17 の
FFN down は、前回から検証 CE が 2.49768 から
2.50175（+0.00407）、確認 CE が
2.43741 から 2.43966
（+0.00225）に変わった。幅とデータ順は固定した。

seed 23 の down 1層（196,608 Atom、入力3件）で、CPU FP64 の独立した
radius cell mass 参照と Y・dX・Atom の4勾配を比較した。相対 L2 誤差の
最大値は 1.66e-06、絶対誤差の最大値は 4.65e-06。各 down 実行の
初回更新で4層すべてに正の q が生じた。最終 q とオプティマイザ状態は
チェックポイント検証を通過した。

| Seed | 条件 | 検証 top-1 (%) | 更新中央値 (ms) | 初期化 (s) | ピーク allocated / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 23 | 全 Dense | 46.77 | 36.65 | 0.35 | 2.161 / 2.441 |
| 23 | FFN down | 46.66 | 44.01 | 297.70 | 4.654 / 4.838 |
| 42 | 全 Dense | 46.53 | 36.74 | 0.28 | 3.433 / 3.723 |
| 42 | FFN down | 46.59 | 43.89 | 314.49 | 2.819 / 3.006 |
| 17 | FFN down | 46.84 | 44.01 | 324.68 | 3.453 / 3.723 |

時間は更新11〜4,882の forward・backward・オプティマイザ更新を同期して測った。
メモリはプロセス全体のピークであり、先行条件や照合用コピーの常駐分が
含まれる場合がある。同じ全 Dense 構成でも seed 間で値が異なるため、
この実行順での実測値として記録する。

保存した実行 ID は `lm-fp32-seed-replication-20261117-01`。
RTX 4090、PyTorch 2.9.1+cu128、CUDA 12.8、
Triton 3.5.1、TF32 無効で実行した。取得した summary、
5本の曲線、各 seed の結果、5個の最終チェックポイントのハッシュを
検証した。チェックポイントは学習ホストに保存している。
summary の SHA-256 は `6a4a062cd6b7d5628b8cbdd6a0c4f823cd91d3b52b376b045c13d22661db5c41`。
