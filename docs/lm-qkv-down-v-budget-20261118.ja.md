# 幅2048の FFN down 併用時における QKV の Atom 数

[English](lm-qkv-down-v-budget-20261118.md) | 日本語

[前回の併用条件](lm-qkv-down-2048-20261118.ja.md)の Q/K/V の係数を (56, 56, 80) に
して初期状態から再走し、保存済みの (48, 48, 64) の併用条件と比べる。seed 17、
4,882更新、同じ RTX 4090、凍結済みの TinyStories の分割、バッチ順、
オプティマイザと学習率を使う。4つの down 射影はそれぞれ393,216 Atom とし、
初期値のハッシュを前回と照合する。モデル全体の QKV Atom 数は1,310,720から
1,572,864に増える。QKV の初期値を[前回の QKV 単独](lm-qkv-v-allocation-20261117.ja.md)
と照合し、共通の Dense 初期値も確認する。

前回と同じ幅拡張、FP32 半径デコード、GPU 分散初期化を使う。学習前に
先頭 Q の出力と勾配を独立参照と比較する。全16射影で初回と最終の q 更新を確認する。

基準の併用条件との差を、最終の検証 CE と別ストーリーでの確認 CE で報告する。
同期した全更新の時間中央値と PyTorch のピーク allocated・reserved を記録する。
保存済み対照を使う1 seed の比較とする。

## 再現方法

幅2048の併用比較のソースと拡張を使い、完了済みの集計を PILOT_SUMMARY とする。
他の比較基準も前回と同じにする。TORCH_EXTENSIONS_DIR には前回の SM89 ビルド、
OUTPUT には新規の出力先を指定する。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_down_v_budget.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --qkv-anchor "$QKV_ANCHOR" \
  --pilot-summary "$PILOT_SUMMARY" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力と出力先に --resume を追加する。

## RTX 4090 の結果

seed 17 の (56, 56, 80) の併用条件が4,882更新を完走した。4つの down の初期値は
前回の併用条件と一致し、12の QKV の初期値は同じ配分の QKV 単独と一致した。
先頭 Q の出力・勾配の最大相対 L2 誤差は 2.25e-06。全16射影の初回・最終 q、
全学習ログ、最終チェックポイントのハッシュを確認した。

| down と併用した Q / K / V | 検証 CE | 確認 CE | 検証 top-1 (%) | 更新中央値 (ms) | 準備 (秒) | ピーク allocated / reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 / 48 / 64 | 2.63859 | 2.56407 | 45.36 | 135.13 | 1.35 | 5.398 / 5.986 |
| 56 / 56 / 80 | 2.65907 | 2.57685 | 44.76 | 136.82 | 1.55 | 5.442 / 6.025 |

前回の併用条件に対して、検証 CE 差は +0.02048、確認 CE 差は
+0.01278 だった。(48, 48, 64) の QKV 単独（確認 CE 2.39050）
からの確認 CE 差は、前回の併用で +0.17358、今回の併用で +0.18635。
(56, 56, 80) の QKV 単独の確認 CE は 2.39101 だった。

前回と同じ拡張バイナリで、実行環境は RTX 4090、PyTorch 2.9.1+cu128、
CUDA 12.8、Triton 3.5.1。凍結した runner のコミットは `78524a819f3114b10fe54e77dd0e45ff1f2f9b5d`、
スクリプトアーカイブの SHA-256 は `fc69b03262caaa0a99e9fbfd6c12c23a6ee2cf6ad5aa262b4bcbcbe36206119d`。

照合済み集計の SHA-256 は `2850900ae00f2deffc950268ea59ba0783dabd5cb3c90caa19e34de1022bc837`。
