# cst-research

[English (canonical)](README.md) | 日本語

CSTのスケール、精度、学習過程、速度・メモリを再現可能な条件で調べる研究用repoです。
研究計画と実験の識別情報を先に固定し、観測結果・不確実性・未検証の仮説を分けて残します。
ライセンスは[Apache-2.0](LICENSE)です。

## torchcstとの役割分担

| repo | 役割 |
|---|---|
| [torchcst](https://github.com/takumiecd/torchcst) | 公開Module・演算の数学的契約・backend・autograd・kernelの正しさ |
| cst-research | 研究計画、モデル・タスク、比較条件、学習曲線、統計、実験の来歴と再現手順 |

このrepoはtorchcstのruntimeを複製しません。公開APIを使う研究runnerを今後ここへ追加します。
公開APIだけでは不足する実験用Graph adapterは、そのprivateな依存先・対象commit・
独立検証範囲を明示し、公開APIへの採用と実験内の利用を別に判断します。

## 現在の研究

**HIGGS低密度CSTはRUNNINGです。結果は未完了で、学習runnerの移管も未実施です。**
現在の実行元はtorchcst [PR #141](https://github.com/takumiecd/torchcst/pull/141)の
固定commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`です。
28→N→N→2の中間層だけをCSTへ置換し、N=1024/2048、
α=0.001/0.002/0.005/0.01を調整済みdenseと比較します。
入力・出力はdenseのまま全層を学習します。3確認seedを使い、accuracy −0.2ポイント、
ROC-AUC −0.002の許容差を実行前に固定しています。

[機械可読な研究計画](src/cst_research/studies/higgs-alpha-confirmation-20261105.json)と
[再現・証拠の扱い](docs/reproducibility.ja.md)を参照してください。
過去の研究やraw証拠は元のtorchcst履歴・外部アーカイブに残し、ここへ一括コピーしていません。

新たに承認された[Static/RigL比較](docs/higgs-dst-comparison.ja.md)では、
両疎参照のLRを独立に調整し、3つのpaired学習seedと同期間controlで比較します。
最初の実装範囲は資格確認とtuningで、必要なcontrolが全てそろうまで最終freeze・
confirmation・assessmentを拒否します。元のCST主研究は変更せず、既存seed17 DST pilotは
探索結果として扱います。

次の承認済み方針は[層位置の比較計画](docs/higgs-layer-position-plan.ja.md)です。
幅・深さを固定して入力寄り/中央/出力寄りの隠れ層を比べ、有限の密度探索から立てた予測を
検証します。現在の実験完了後に開始し、runnerと実行プロトコルはまだ固定していません。

## すぐに使う

Python 3.11以上が必要です。レジストリCLIは標準ライブラリだけで動作し、GPUを使いません。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cst-research list
cst-research show higgs-alpha-confirmation-20261105
cst-research validate --json
python -m unittest discover -s tests -v
```

インストールせず確認する場合は `PYTHONPATH=src python -m cst_research list` も使えます。
`validate`は登録内容と固定条件の検査です。学習結果、外部証拠の存在、GPUの正しさを
検証したことにはなりません。正式なHIGGS campaignは固定した外部ソースで実行します。
独立した数値admission後に変更なしのbenchmarkを呼ぶ
[限定A100 MIG環境比較adapter](docs/higgs-gpu-comparison.ja.md)を別に用意しています。

torchcstのライブラリが必要な場合だけ、固定commitのoptional extraを使えます。

```bash
python -m pip install -e '.[torchcst]'
```

これはライブラリ依存の導入です。現在のbenchmark runnerを使うには、
[固定torchcst sourceの取得手順](docs/reproducibility.ja.md)に従ってください。

## 外部証拠を参照する

データ・checkpoint・raw logs・pool状態・認証情報はGitへ入れません。
`CST_RESEARCH_EVIDENCE_ROOT`の下にstudy IDのディレクトリを置くか、
Git管理しない `research.local.toml` で保存先を設定します。

```toml
[evidence]
root = "./evidence"
```

```bash
cst-research --config research.local.toml show higgs-alpha-confirmation-20261105 --evidence
```

設定は所在を解決するだけで、コピー・ダウンロード・削除は行いません。
`--evidence`の出力にはローカル保存先が含まれるため、公開用レポートには含めないでください。
環境変数のrootが設定されている場合はTOMLより優先します。

新しい研究・変更は[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)に沿ってPRで追加します。
