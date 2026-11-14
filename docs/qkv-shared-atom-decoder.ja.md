# QKVのAtom共有デコーダ

[English](qkv-shared-atom-decoder.md) | 日本語

対照はTinyStoriesの幅1024・c64・seed 17でQKVだけを置換した有理式包絡の
凍結済み実験です。各Atomの4個の生の値を潜在ベクトルの初期値にします。
全Atomで共有する残差型の4 → 16 → 4 SiLU MLPが振幅、半径比率、出力中心、
入力中心を復号します。同じ有理式包絡で比率を実半径に変換します。
最終層をゼロ初期化し、最初は元のAtom値と同じ値を出します。通常の次トークン
予測CEでMLPと潜在ベクトルを同時に学習します。

MLPの学習スカラーは148個です。QKVの12射影すべてからタスク勾配を受け取り、
MLPの勾配ノルムを個別に0.1でクリップし、ベースの学習率を使います。
他のパラメータには対照と同じクリップ1.0を適用し、Atomの学習率倍率は32です。
初期化、トークン順、学習・検証データ、4882ステップの予算と学習率の長期
ホライズンを対照と揃えます。手で指定した半径包絡を使う残差型の表現を
1 seedで測る実験です。

対照の凍結済みPythonソースを `PYTHONPATH` に置き、同じプロトコルを使います。対照と同じCUDA環境で、継承する
ソース変更マニフェストを変更順に2つ指定します。出力先は新規ディレクトリです。

```bash
python "$CHECKOUT/src/cst_research/experiments/qkv_shared_decoder_run.py" \
  --reference "$QKV_REFERENCE" --protocol "$WIDTH_PROTOCOL" \
  --source-changes "$RADIUS_CAP_MANIFEST" --source-changes "$SQUARE_QKV_MANIFEST" \
  --output "$OUTPUT"
```

学習前に初期タスク損失と潜在パラメータの勾配を、直接学習のQKVと照合します。
検証CE、振幅帯ごとの実半径中央値、勾配ノルム、時間、メモリ、チェックポイント
のハッシュを記録します。最終チェックポイントは復元して再評価します。
