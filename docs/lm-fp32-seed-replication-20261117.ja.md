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

保存済みソースとデータの独立コピーを使用する。31f40e64241eb866e95305401b94cc8da4ac5a7e
から 6703a18a55cd949564be8a0ceca1ca1b813f7fc5 への torchcst の変更ファイルを
重ね、SHA-256 を記録したマニフェストを渡す。保存済み有理式アダプタと元の
trainer は前回の実行と照合する。TORCH_CUDA_ARCH_LIST=8.9 を指定して
マージ済みソースから拡張をビルドする。runner はデータ、ソース、ネイティブ
バイナリの識別情報、実行の識別情報、復元したチェックポイントを検証する。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
python scripts/run_lm_fp32_seed_replication.py \
  --reference-run "$REFERENCE_RUN" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力に --resume を追加し、最後に記録したチェックポイントから
続ける。実験登録に計測結果の保存先を記載した。
