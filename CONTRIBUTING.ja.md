# 研究への参加

[English (canonical)](CONTRIBUTING.md) | 日本語

変更前にこの文書と[AGENTS.md](AGENTS.md)を読みます。
研究設計、実装、観測、統計的な主張を区別し、別の人が条件を追える最小の記録を残してください。

## 変更とレビュー

- `feat/<目的>`、`fix/<目的>`、`docs/<目的>`、`refactor/<目的>`、`test/<目的>`など、目的に沿うbranchを使います。
- GitHub PRでレビューしてから統合します。`main`へ直接pushせず、ローカルmainの統合代替も行いません。
- kernel・公開演算・backendの変更はtorchcst側で、そのrepoの手順に従って扱います。
- 実行中の固定sourceを編集して実験を続行しません。変更が必要なら別version/studyとして、既存証拠と区別します。
- dataset、secret、raw logs、checkpoint、VM/queueの内部状態、絶対ユーザーパスをcommitしません。
- 作業treeを片付ける前に、commit・未commit変更・ignored証拠と復旧経路を確認します。

## 研究を登録する

`src/cst_research/studies/`へJSONを追加し、IDとファイル名を一致させます。
source repoと完全commit、runner entrypoint、公開API/private adapterの境界、
データ・選択/評価の分離、比較条件、乱数seed、時間・メモリの計測範囲、証拠locatorを記録します。
レジストリ形式を拡張する場合はvalidatorと意味のある検査も更新します。

主張に必要な許容差・比較family・選択規則は結果を見る前に固定します。
変更は実施前の変更と事後の探索を区別し、理由・時点・影響を記録します。
3seedの低検出力や、条件付きbootstrapの範囲を、普遍的な最適値やjoint保証として報告しません。
RUNNINGをCOMPLETEDへ変えるのは、必要な実行と回収・検証が実際に完了してからです。

既存HIGGSの条件を変える場合は新しいstudy IDを使います。現在の学習はtorchcst側にあり、
このrepoでの登録やCLI検査を、学習移管・再実行・結果採用と扱いません。

## LM学習曲線の対照

所有者の指示により、新しいLMの学習曲線比較にはDenseとsourceを検証したDST対照を
含めます。データ・モデル/通常初期化・学習token予算を揃えます。
所有者指定の主な表現予算は、各LinearのCST Atom数とDSTのboolマスク1の数の一致です。
raw学習変数数は別に報告し、以前のscalar数を揃えた結果は元の基準の補助比較として保全します。
DSTは密な重みとboolマスクを維持し、この比較でCOO/CSRへ変換しません。
DST方式・mask/更新規則・LR調整を記録。全候補を保全し、DEV選択と明示します。
作図の--dst-runと互換性検査を使用し、HIGGSや不適合なタスク/source/予算の曲線を
代用しません。半径/fraction診断はCSTのみです。Dense保存のDST精度参照から、
疎kernelの速度・メモリ効率を主張しません。

## 実験と証拠

実行前にsource/data/config identityを固定します。ownerのGPUを使う場合は、
そのresource skill・共有pool・割当の規則を読み、既に許可された範囲で進めます。
repoへの書き込み権限はGPUの新規割当権限を意味しません。

raw証拠はignored保存先か管理された外部storageへ残し、公開ノートには
再現コマンド、source/data/result SHA、job識別子、hardware/runtime、完全step時間、
準備時間、allocated/reserved peak、失敗・不確実性を記録します。
同じGPU種でも別VMのHPO時間は探索的です。主要な時間比較には同一job内のpaired比較を使います。
過去の結果は元のcheckout・workload・runtimeへ結び付け、履歴の移動だけで再検証済みとしません。

## 検査

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m cst_research validate --json
```

CLI検査にGPU・PyTorchは不要です。学習runnerを追加した場合は、独立oracle、
完全step、optimizer状態、データ分離と同じ条件のdense baselineを別途検証します。
PRには変更の目的・実際に動作した範囲・未実施の検証・証拠の所在を簡潔に記載してください。
