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

このrepoはtorchcstのruntimeを複製しません。研究consumerとworkflow controllerをここで管理し、
演算実装は固定した外部sourceから利用します。
公開APIだけでは不足する実験用Graph adapterは、そのprivateな依存先・対象commit・
独立検証範囲を明示し、公開APIへの採用と実験内の利用を別に判断します。

## 現在の研究

**HIGGS campaignは科学的admission・評価が未完了で停止中です。学習jobは完了し、primary/DSTの学習は動いていません。**
現在の実行元はtorchcst [PR #141](https://github.com/takumiecd/torchcst/pull/141)の
固定commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`です。
28→N→N→2の中間層だけをCSTへ置換し、N=1024/2048、
α=0.001/0.002/0.005/0.01を調整済みdenseと比較します。
入力・出力はdenseのまま全層を学習します。3確認seedを使い、accuracy −0.2ポイント、
ROC-AUC −0.002の許容差を実行前に固定しています。

[機械可読な研究計画](src/cst_research/studies/higgs-alpha-confirmation-20261105.json)と
[再現・証拠の扱い](docs/reproducibility.ja.md)を参照してください。
過去の研究やraw証拠は元のtorchcst履歴・外部アーカイブに残し、ここへ一括コピーしていません。

統合したconsumer実装と測定履歴を以下に示します。元のsource/data identityを保持し、
統合を再実行や再検証とは扱いません。[統合の来歴](docs/integration-provenance.ja.md)を参照してください。

| 研究 | 状態 | 範囲 |
|---|---|---|
| [HIGGS primary](docs/reproducibility.ja.md) | RUNNING | 外部975、最終30モデル評価は未完了 |
| [HIGGS static/RigL比較](docs/higgs-dst-comparison.ja.md) | RUNNING | Stage1/2 consumer、共通期間の78モデルgateは未完了 |
| [A100 MIG比較](docs/higgs-gpu-comparison.ja.md) | COMPLETED | 限定1-epoch環境比較 |
| [座標LR](docs/higgs-coordinate-lr.ja.md) | COMPLETED | 短期pilotと有限2-seed追加検証、stock q1を維持 |
| [深さ・層位置](docs/higgs-depth-position.ja.md) | COMPLETED | developmentのみの13条件、限定的なstall証拠 |
| [固定output gain](docs/higgs-depth-gain.ja.md) | COMPLETED | 探索的3条件、部分改善後もDenseに劣る |
| [Covertype転移](docs/covertype-transfer-results.ja.md) | COMPLETED | 9モデル固定・heldoutを1回評価、CST両密度はDenseに劣る |
| [8更新の正規化診断](docs/higgs-depth-normalization-diagnostic-results.ja.md) | COMPLETED | 9×8 task更新の機構診断。品質選択・既定値採用なし |
| [正規化の3 epoch追試](docs/higgs-depth-normalization-learning.ja.md) | STOPPED | 元ea0はmain300k後にexit1・未完了を維持。保持証拠の再利用は別recovery系譜だけ |
| [正規化load契約のrecovery](docs/higgs-depth-normalization-learning-results.ja.md) | COMPLETED | 探索的9条件を結合完了。A104のRMS/LN救済後もDenseに劣り、正規化の既定採用なし |
| [線形Atom数：Stage B結果](docs/higgs-linear-atom-results.ja.md) | COMPLETED（限定Stage B） | seed17の1 epoch全16終点。H8 CSTはbias-only、Atomオーダ・最小cの主張なし |
| [scaleとLayerNorm](docs/higgs-initialization-results.ja.md) | COMPLETED | GPUごとの6条件を独立検証した2 cohort |
| [幅対応・TRAIN-RMS初期化](docs/higgs-initialization-results.ja.md) | COMPLETED | 旧方式と幅対応は停滞、実測Dense-RMSで救済。1 seed |
| [He／Xavier二乗平均初期化](docs/higgs-initialization-results.ja.md) | COMPLETED | q=0の対応比較と明示公開API。普遍的精度の主張なし |

[観測結果と限界](docs/bounded-validation-results.ja.md)で、完了した座標・深さ・gain結果を
実測commitとraw auditへ結びます。developmentで選んだpilotは最小alphaや正式な非劣性を
確定しません。HIGGS primaryとDSTは最終固定familyのadmissionと評価が完了するまでRUNNINGです。
測定jobは元のsource snapshotを保持します。統合をjobの再開や未完了の科学的評価の完了とは扱いません。

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

[幅を考慮したamplitude初期化実験](docs/higgs-width-amplitude-init.ja.md)では、gain・LayerNormなしの生の`0.1/sqrt(A/N)`を、従来初期化と以前の固定gain対照と比較します。限定したGPU実行と独立検証は完了しました。[結果](docs/higgs-initialization-results.ja.md)を参照してください。

[生のamplitudeをDenseのRMSへ合わせる追加実験](docs/higgs-rms-amplitude-init.ja.md)は、学習前のamplitudeを校正し、実行時gain1・正規化NONEを保ちます。限定したGPU実行と独立検証は完了しました。[結果](docs/higgs-initialization-results.ja.md)を参照してください。

[He／Xavier初期化pilot](docs/higgs-fan-initialization.ja.md)はデータ不要のAtom二乗平均目標と小文字`q`の明示指定を使い、新規の対応するDense／CSTを比較します。CPU準備・native検証・学習結果・公開default採用は区別します。

完了した[初期化の結果](docs/higgs-initialization-results.ja.md)に、旧方式・幅対応・TRAIN-RMS・データ不要のHe/Xavier・正規化対照をまとめます。明示的な公開APIは[torchcst初期化ガイド](https://github.com/takumiecd/torchcst/blob/main/docs/initialization.ja.md)を参照してください。
