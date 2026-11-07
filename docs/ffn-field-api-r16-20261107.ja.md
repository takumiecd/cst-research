# 公開フィールド別学習率APIの検証（R16）

[English](ffn-field-api-r16-20261107.md) | 日本語

絶対値`field_lrs`の公開APIはnative・実LMの検証を通り、
[torchcst PR151](https://github.com/takumiecd/torchcst/pull/151)で統合しました。
merge commitは`c3b68dcf6275f96d29a21ff2aa6e989d42de2cff`、検証operatorは
`a6c5f12a31442a03afb79cf47ebdb88db13afbb7`です。最終PR head21ac2029は検証文書だけを追加し、
runtime・testsのバイトは同一です。最終CPU CI42324636658も成功しました。
新しい`[ffn]` extraは統合後のライブラリを固定し、旧`[torchcst]`はHIGGS975のまま別の環境で使います。

フィールド別LRの区別を再現する場合、通常groupを0.0003、Atom groupを0.0048に分けます。
公開APIの指定は次の形です。

```python
optimizer = CSTAdamW([
    {"params": ordinary_parameters, "lr": 3e-4},
    {"params": atom_parameters, "lr": 4.8e-3,
     "field_lrs": {"amplitude": 2.4e-3, "input_center": 4.8e-3,
                   "output_center": 4.8e-3}},
], model=model, weight_decay=0.0)
```

設定可能な初期絶対学習率であり、普遍的な推奨設定ではありません。
未指定フィールドとPolar外部時間はgroup LRを使います。通常のschedulerは初期group基準値に対する
共通係数を与え、`get_field_lrs()`で現在の実効指定値を確認できます。
フィールドはkernelのparameterization・geometry宣言から決まり、raw列番号では指定しません。
Atomごとに1つのParameter・モーメント表・カウンタを保ち、q・activityと振幅運動の連成も保ちます。
[正本optimizer契約](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/cst-optimizer.ja.md#field-learning-rates)を参照してください。

実RTX4090 CC8.9・24GB級、Torch2.11.0+cu126、Python3.12.9、NumPy2.0.2でremote CPU25テストと
恒久native CUDA8テストをskipなしで通りました。CUDA側はnative/wrapper、ピークfield LR、
CPU FP64とのParameter・モーメント・カウンタ比較、0を含むschedule、シリアライズ後の次更新を含みます。

次に使い捨ての新規4ブロックN256/FFN4N/c8モデルで、native/wrapper × 振幅8/16/32/64、
中心・外部時間16の8設定を確認しました。全8 FFN投影がCSTで、attention・embed・outputは通常のDense、
batch2/context128/accumulation4、FP32/TF32off、AdamW(.9,.95)、epsilon1e-8、decay0、
global clip1です。通常warmup20更新と、次の順序付きbatchでのピークLR stress5更新で、
損失・Parameterが研究用アダプタとすべて完全一致しました（最大差0）。勾配・モーメントは
記録したFP32許容差を通り、カウンタは一致しました。field指定なしで構築した公開optimizerへ
checkpointの設定を復元し、26回目のピークLR実タスク更新もパラメータ差0でした。
学習後の全64投影でY/dX/全Atom VJPのCPU FP64分割reference検証を通りました。

これは正しさの証拠です。stress列は通常20Mの学習軌跡や品質・速度・メモリ実験ではありません。
元の学習の再開・checkpoint上書き・再試行は行っていません。R12–R15の品質結果は古いoperator・
アダプタの来歴を保ちます。この範囲のeager field LRを検証し、stock Graph benchmarkはcold
qualificationでfield指定を拒否してeagerへ戻ります。Graph・他hardwareへは一般化しません。

manifestを含む回収8ファイルのSHA256を照合しました。

- Archive: `64ae9bac51b1565d35cae76b875f45096a9145de663a2cb0f65ffd98367dfc06`。
- Manifest: `c37a46db68037dbdc5b3b36a1d484df8047251e222f4b86719e2a399137471f2`。
- Full-LM result: `49125fd92525bde1fba4ee1220fbe92a5fdcb9534852b626ccc5d1453b880800`。
- Tested operator archive: `aeac783c7053f97374337f1fb0a918ae5d69e90337b05d4a0df3ed9cc8c6af7a`。
- Consumer: `351617dc81f529ce10e964e5a762f336b7c24b39`。
- Data manifest: `5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。

登録した外部証拠保存先にcontroller・request・固定ソース・JUnit/ログ・
`qualify-public-field-api-r16-02.py`と完全結果を保持しています。記録した数値環境と固定operator・
consumerを使い、`--data TOKEN_DIR --output NEW_DIR`で実行します。
恒久の小規模テストはtorchcstの`tests/test_optimizer_field_lrs*.py`です。
[上流検証記録](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/research-history/optimizer-field-lrs-20261107.ja.md)
に再現コマンドとローカルCPU・buildの範囲を記録しています。
ローカル2121成功・2474skipでGPU検証を代替していません。
所有する検証workerは終了し、ユーザー所有のHubは起動したままです。
