# 実行元と証拠の再現

[English (canonical)](reproducibility.md) | 日本語

## 現在のHIGGSはtorchcstで実行中

登録したstudyは、torchcstの[固定source](https://github.com/takumiecd/torchcst/tree/97580582ed28c4e494bf5c2656bbe94a85c544c3)で
実行中です。研究計画の正本はそのcommitの
[`higgs-alpha-confirmation-20261105.ja.md`](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/docs/research-history/cuda-linear/higgs-alpha-confirmation-20261105.ja.md)です。
このrepoのJSONは運用上の入口と索引であり、既存runのsource identityを書き換えません。

```bash
git clone https://github.com/takumiecd/torchcst.git torchcst-higgs-source
cd torchcst-higgs-source
git checkout --detach 97580582ed28c4e494bf5c2656bbe94a85c544c3
python -m pip install -e '.[dev,cuda]'
python -m benchmarks.training.higgs_confirmation_campaign --help
```

optional extraだけのインストールでは、source checkoutにある`benchmarks`までは導入されません。
新たな再現実行は別の出力・state・識別情報を使い、現在動いているrunへ混ぜません。
GPU作業前にtorchcstのCONTRIBUTING、研究運用規則と対応resource skillを読み、共有L4では
共有poolに提出します。ここで新しいVMを確保したり、既存poolの状態を直接編集したりしません。

## データと段階

元のHIGGSは11,000,000行です。training10,400,000行だけで算出した標準化を使い、
学習は先頭100,000行、公式validation先頭50,000例をdevelopment、後半50,000例を
confirmationに分けます。公式test500,000例は今回評価しません。
必要なprepared NPZのSHA256はstudy JSONに固定されています。
元CSV・NPZはこのrepoには含みません。データ準備・検証の経路は固定sourceの既存runnerと
外部data-manifestに従い、別の分割や標準化で同じ結果と名乗らないでください。

固定sourceのentrypointは次のものです。

```text
python -m benchmarks.training.higgs_confirmation_campaign
  --stage qualify|tune|freeze-width|confirm|assess
  --data PREPARED_NPZ --output FRESH_OUTPUT --campaign-state CAMPAIGN_STATE
  --source-identity FROZEN_SOURCE_MANIFEST_SHA256
```

これはオプションの対応表です。実際のpool driverはimmutable source cache、依存環境、
source/data/runtimeに結び付くstate、hash付き回収を管理します。
rootとなるstateを新規作成し、`qualify`を通してから各stageを順番に実行します。

1. `qualify`: 独立oracleとeager/Graphの完全stepを確認する。
2. `tune`: 各widthのdense＋4密度をdevelopment seed17で調整する。候補LRを全て10epoch走らせる。
3. `freeze-width`: 5構成の開発結果から共通10/20epochを決め、必要なら上位2LRを20epochで再比較する。
4. `confirm`: 固定LR・予算でseed29/43/71を新規学習する。1つのjob内にdense＋4密度を含める。
5. `assess`: 全30checkpointを先に凍結し、その後でconfirmation50kを評価する。testは開かない。

具体的な`--width`、`--kind`、`--divisor`、`--seed`はsourceの`--help`とstudy JSONを参照してください。
`confirm`は独立学習seedのstage名です。このstageのlearning curveも**development指標**であり、
held-out confirmation評価ではありません。

## 比較・統計の範囲

accuracy −0.2ポイントとROC-AUC −0.002を両方通すIUTを使い、2幅×4密度の8構成に
Bonferroni補正します。16指標のtwo-sided同時区間は別の診断です。
3seedでは判定保留が残り得ます。最小のsupportedな検証点と、閾値が特定できたかを区別します。
α=0.001が通っても、それ未満の必要密度は未検証です。

時間は同期した完全学習stepを測り、準備込み時間を別に残します。評価・checkpoint/hash診断の
時間は学習時間から除きます。HPOの別jobは別L4 VMにまたがる可能性があり、CPU hostモデルも
記録されていないため、その時間曲線の重ね描きは探索的です。
主要な速度・目標精度到達時間は、各幅・独立seedで同一job内にあるdenseと4CSTのpaired比較を使います。
学習損失とdevelopment損失の曲線・末尾の改善量は、数学的な最適化収束の証明ではありません。
allocator allocated/reserved peakとGPU process全使用量も区別します。

## 外部保存と移管

`CST_RESEARCH_EVIDENCE_ROOT/<study-id>/`は証拠の索引先です。
環境変数、またはGit管理しないTOMLの`[evidence] root`で設定できます。
rootが相対パスならTOMLの置かれたディレクトリから解決します。
`CST_RESEARCH_CONFIG`でTOMLを指定でき、`CST_RESEARCH_EVIDENCE_ROOT`が優先します。
CLIは所在を表示するだけで、存在・SHA・実行の完了を保証しません。

元のpool job result manifest、recovery archiveとreceipt、source/data identity、curve、
checkpoint・predictionsのhashを保存します。回収・再利用の前に外側と内部memberのhashを検証します。
公開用のノートは相対locator、source/result SHA、job IDと要約を持ち、ローカルの絶対パスやsecretは含めません。

今後研究runnerをここへ移す場合は、公開torchcst APIに依存する部分とprivate実験用Graph adapterを
明示し、独立oracleと完全step、optimizer状態、時間・メモリの同条件比較を再検証します。
公開APIの変更はtorchcst側の別PRです。既存のprivate adapterを公開APIと呼び替えません。

旧実験は元のtorchcst commit・記録・ignored archiveに残っています。履歴をコピーしただけで
新repo上の実験結果・再現済み証拠とは扱いません。外部証拠を整理するときも、元のsource・
data・hardware・runtime・workload・jobの対応と復旧経路を保持してください。
