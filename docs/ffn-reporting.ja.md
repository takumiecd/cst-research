# FFN観測の集計

[English (canonical)](ffn-reporting.md) | 日本語

`ffn_report`はTorch・GPUなしで回収済みのR8形式の証拠を読みます。終了command receipt、
summary/curveのhash、合格した学習済みcheckpoint監査を結びます。途中終了・固定batch・
診断付きrun、異なるsource/data/recipe/horizon、比較条件の欠落・重複、不完全なcurve grid、
異なるAtom group LR、学習済み射影・resume確認の欠落を拒否します。receiptの集計であり、
GPUの数式の独立再実行や、任意のファイルの真正性認証ではありません。実際に監査した
sourceとraw証拠を保持してください。

[request template](ffn-report-request.example.json)をコピーし、evidence rootからの相対pathと、
実際の終了receipt・合格auditのSHAに置き換えます。template自体は完了した結果ではありません。
各shape・seedにDense/CST/FFNなしが必要です。`development`はseed17、`confirmation`は
正確に29/43/71を使い、request内で選んだrecipeとtoken予算を揃えます。recipeの選択に
confirmation scoreを使わないでください。

```bash
python -m pip install -e '.[reporting]'
PYTHONPATH=src python -m cst_research.experiments.ffn_report \
  --evidence-root "$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106" \
  --request evidence/report-request.json --output evidence/new-report --plots
```

新しい出力directoryを使います。JSONは全run、固定promptの両方の出力、source/data/request/
evidence hash、reporter source hash、描画環境を保持します。同等の英日Markdownには指標と
最初のseed・最初の固定promptを載せ、良い生成例を選別しません。PNG/SVGは全seedの学習/
development曲線と平均・標本SDの帯、complete-update/allocated/reservedの観測を描きます。
optional reporting extraが必要なのは`--plots`だけです。

SDは独立したseedのrun間のもので、1 run内の数千update間のSDや信頼区間ではありません。
seedを対応させたCST−対照のCE差も各seedの値を保持します。開発用1 runではseed SDを
推定しません。Parameter数は宣言したFFN予算と照合します。幅の観測点だけでruntime/memoryの
漸近オーダーや形式的な非劣性を証明しません。研究reportに含める前に、描画と生成例の失敗を
確認してください。raw data/checkpointはGitの外に保持します。
