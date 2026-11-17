# 幅2048 QKV の V 寄りの Atom 配分

[English](lm-qkv-v-allocation-20261117.md) | 日本語

[配分の予備比較](lm-qkv-role-budget-20261117.ja.md)では、seed 17 で V に Atom を
追加した条件の CE が最も低かった。今回は均等 c64 と同じ総数で V に多く配分する
条件と、総数を下げる条件を試す。各条件で4ブロックの Q・K・V を置換する。
同じ RTX 4090 で、3条件を seed 17・23・42 の初期状態から4,882更新まで学習する。

| 条件 | Q | K | V | モデル全体の Atom 数 |
| --- | ---: | ---: | ---: | ---: |
| 均等配分 | 64 | 64 | 64 | 1,572,864 |
| 同じ総数 | 56 | 56 | 80 | 1,572,864 |
| 少ない総数 | 48 | 48 | 64 | 1,310,720 |

[均等配分の比較](lm-qkv-local-sweep-20261117.ja.md)のモデル幅、文脈長、凍結した
TinyStories の分割とデータ順、FP32 半径デコード、GPU 初期化、オプティマイザ、
学習率スケジュールを使う。c64 の再走では3 seed とも初期パラメータ全体のハッシュを
前回と照合し、全9走で共通の Dense 初期値も照合する。seed 17 の先頭 Q では、
独立参照と出力・入力勾配・Atom の4勾配を比較する。新しい2配分の先頭 Q で
CPU・GPU 初期値のビット一致を確認し、各条件の全12射影で初回と最終の q 更新を確認する。

同じジョブで再走した c64 を基準に、確認 CE、検証 CE、全更新時間の差を
seed ごとおよび平均で報告する。確認 CE は検証とは別のストーリーで測る。
両方の分割で、各配分が基準を上回った seed 数を記録する。3 seed の結果は探索的な
比較とする。検証 top-1、モデル準備時間、PyTorch アロケータのピークも記録する。
実行順は runner で宣言し、seed ごとにローテーションする。

## 再現方法

均等配分の比較と同じ凍結ソース、元の射影アーカイブ、結合マニフェスト、ネイティブ
拡張、比較基準を使う。TORCH_EXTENSIONS_DIR に既存の SM89 拡張ディレクトリを、
OUTPUT に新規の出力先を指定する。

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_v_allocation.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

再開時は同じ入力と出力先に --resume を追加する。
