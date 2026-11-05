# 研究workflowの統合と実測sourceの保持

[English (canonical)](integration-provenance.md) | 日本語

個別にreviewされた研究consumerを共通repo treeへ統合します。元commitをmerge祖先として
保持し、実行中ジョブのsource snapshotやdata/protocol identityは変更しません。GPU再実行、
raw結果の再検証、primary評価完了、torchcstへの実験的変更の採用ではありません。
operator/backend/kernel runtimeも複製しません。

## 正確なproducer identityを引き続き正本とする

| Consumer | Measured source commit | Frozen consumer manifest SHA256 |
|---|---|---|
| DST Stage1 | `22f2810b1bc51a599e51c8936acac310ca9455ef` | `8e0b689790e387bb7adcadc6723905ed210befce3105735df6bf1f16f8b126a6` |
| DST Stage2 | `0f8160a8d9206100455dc6f8d05aab858e4a88d5` | `e7db6b9b13a898b7e64a62c4418fa6bd514afd8cf70ebb52c1bc7640590c4983` |
| Coordinate LR | `83cb77fe84e91ae296c24f2144bd56ca9398f404` | `7ce3d76f3ea3e09084e5c2dcb37ade7581eebffbf0f22c1b74dae08b467511bd` |
| Depth/position | `f6d08ad76001a69070b0965cefe7614f192ddf3f` | `aec42b259f37bd3922a3763f4c4f5e3af9b9a926263b10128861d09b18295a15` |
| Fixed depth gain | `69d7f0636ce47db892096d7b7aacc2c2cb731b8b` | `b7b6428e80bbf18fc0cff22af2388e6818c3c9c28ad9221c98c352123666d193` |
| Covertype | `bfa58dce18da85f9d2cb3a791db6d6a5126805d7` | `8a3c45f7677001c66cc666009d3dfa4a9027d0e1a7571eb571032be75df59d48` |

Stage1からdepthまでは祖先chainです。gainとCovertypeはdepthf6から分岐したconsumerで、
両履歴を保持します。以前の座標admission失敗source/attemptと修正83 oracleも履歴・外部証拠に
残します。固定worktree、branch、manifest、raw archive、recovery capsule、失敗attemptを
この統合で削除したり書き換えたりしません。

primary HIGGS benchmarkは外部torchcst `97580582ed28c4e494bf5c2656bbe94a85c544c3` で
動き続けます。DST cellは外部 `f37729e61dc11a57e76103aa2457c2be11c4c306`、original controlは
975を使います。座標・深さ・gain・Covertypeの演算も未変更975です。研究ローカルのmodel、loop、
統計、private benchmark adapterはconsumer境界を明記し、private Graphや座標LRの対応を
公開runtime保証とは扱いません。

## 完了は事前に限定した範囲に対するもの

座標LR、depth13、固定gain3は有限の実行・回収・独立raw検証が完了したためCOMPLETEDとします。
実source、manifest、passed-summary audit identityはregistryの `execution` 記録と
[観測結果と限界](bounded-validation-results.ja.md)にあります。完了は肯定結果を意味しません。
座標q10は長い追加検証で一貫した改善がなく、最小密度全層stock depthはstallし、gainはそれを
部分的に改善しただけです。成分診断の失敗、post-hoc admissionの範囲、reader失敗、
conditioning/capacityの不確実性を引き続き開示します。後続の
[正gainの補足](positive-gain-equivalence.ja.md)は、この構造で全体の理想実数関数クラスが
同一であることを示すデータ不要の解析であり、別の学習結果ではありません。

HIGGS primaryとmatched DST比較はRUNNINGで、primary30/comparison78のadmissionと
評価は未完了です。Covertypeは実all9 seal/raw-admission gate、admitされたheldout評価1回、
保存予測の独立監査が完了した後だけCOMPLETEDとしました。
[限定された転移結果](covertype-transfer-results.ja.md)は試したCST両密度で負でした。
original familyとcomparison familyを分離し、評価結果で固定した選択規則を変えません。
公式testは予約したままです。

## 実測runは実測treeから再現する

表の正確なconsumer commitと対応する外部source/data manifestを使い、resource手順を読み、
新規output/stateを選びます。統合treeのsource identityは新しく、古いconsumer manifestのidentityへ
付け替えて通すことはできません。既存jobの継続は未変更の元snapshotを使います。統合treeでの
再現には別consumer manifestの固定とadmissionが必要で、別の証拠として扱います。

CPU test、registry/package検証は統合codeとmetadataの検証です。明示的に設定した実975 CPU
fixtureは公開modelの所有権、optimizer/retraction、checkpoint契約を確認しますが、GPU軌跡を
再現したり評価データを再び開いたりしません。raw data、model、認証情報、machine固有の
運用helperは[証拠規約](reproducibility.ja.md)に従ってignored外部証拠へ残します。歴史的protocolの
数学・予算は維持し、更新した状態の前書きから後続の実測結果を参照します。
