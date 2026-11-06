# Covertypeの限定的なタスク移行結果

[English (canonical)](covertype-transfer-results.md) | 日本語

この有限条件では、調べたCSTの2密度はいずれも独立に保留したCovertype行でDenseを
下回った。対応する3つの学習seedすべてで、alpha=0.001と0.01のaccuracyは低い。
これはN=1024でのタスク移行の負の結果であり、CST全体や十分なalphaについての
普遍的な結論ではない。

構造は54→1024→1024→7で、入出力はDense、正方中間層1つはbiasなし、ReLUは2つ。
全層を学習する。Alphaはatom予算A/N²（整数切捨てA=1048/10485）であり、
dense重みの非ゼロ率ではない。全モデルでeager FP32、TF32 off、B=4、10完全epoch
（250,000 minibatch/Adam更新）を使う。実行環境はRTX3090 1基、CC8.6/82SM/
25,296,044,032 bytes、Torch2.11.0+cu126/CUDA12.6、Python3.12.9、driver580.95.05、
Triton3.6.0、NumPy2.0.2、NVCC12.6.85。
[固定protocol](covertype-transfer.ja.md)と
[登録済みexecution identity](../src/cst_research/studies/covertype-transfer-20261106.json)を参照する。

ソース凍結済みのランダム行分割はtrain100k/development50k/heldout50kで、連続値10列の
正規化はtrainだけでfitした。各方式でseed17、LR1e-4/3e-4/1e-3を3epoch探索し、
所定の端点LR（3e-5または0.003）を最大1つ追加した後、真の上位2LRを初期状態から10epoch再学習した。
Denseは0.0003、CST両群は0.001を選択した。Checkpointはdevelopment CE最小の
最も早いepochで選ぶ。その後、固定した10epoch条件を新しい対応seed29/43/71で学習。
9つすべての選択checkpointをまとめて封印し、独立監査してからheldout評価を1回行った。
実行総数は学習26cell/183epochで、全学習曲線を保存している。

## Heldout結果

共通のheldout50k例で個別に採点した3 seedの指標を平均する。記述的な値であり、
正式な非劣性・有意性検定や地理的な
汎化についての主張ではない。

| Method | Mean CE | Mean accuracy % | Mean Macro-F1 | Mean Macro OvR AUC | Paired mean accuracy delta vs Dense (pp) | Paired mean AUC delta |
|---|---|---|---|---|---|---|
| Dense | 0.303929 | 87.9207 | 0.804058 | 0.986685 | 0 | 0 |
| CST alpha=0.001 | 0.399907 | 83.3613 | 0.688082 | 0.976923 | -4.5593 | -0.009762 |
| CST alpha=0.01 | 0.371187 | 84.6973 | 0.734105 | 0.980509 | -3.2233 | -0.006176 |

選択された全9checkpoint（heldoutによるcheckpoint再選択はしていない）:

| Method | Seed | Selected epoch | Heldout CE | Accuracy % | Macro-F1 | Macro OvR AUC | Train s (10 epochs) |
|---|---|---|---|---|---|---|---|
| Dense | 29 | 9 | 0.305178 | 87.8280 | 0.812309 | 0.986473 | 126.775 |
| Dense | 43 | 9 | 0.304472 | 87.8000 | 0.801412 | 0.986803 | 132.435 |
| Dense | 71 | 10 | 0.302136 | 88.1340 | 0.798453 | 0.986779 | 124.913 |
| CST alpha=0.001 | 29 | 9 | 0.394169 | 83.5260 | 0.728886 | 0.977612 | 305.716 |
| CST alpha=0.001 | 43 | 7 | 0.406623 | 82.9900 | 0.670625 | 0.975728 | 300.621 |
| CST alpha=0.001 | 71 | 10 | 0.398929 | 83.5680 | 0.664735 | 0.977428 | 298.131 |
| CST alpha=0.01 | 29 | 9 | 0.368974 | 84.8140 | 0.762324 | 0.980720 | 313.569 |
| CST alpha=0.01 | 43 | 9 | 0.375746 | 84.5100 | 0.745164 | 0.980640 | 313.507 |
| CST alpha=0.01 | 71 | 10 | 0.368841 | 84.7680 | 0.694826 | 0.980166 | 318.843 |

対応するCE/F1差の平均はalpha=0.001で+0.095978/−0.115976、
alpha=0.01で+0.067259/−0.069954である。

対応するaccuracy差はすべて負で、alpha=0.001は−4.8100〜−4.3020ポイント、
alpha=0.01は−3.3660〜−3.0140ポイント。完全精度の値、seed別CE/F1/AUC差、
平均・SD・範囲を独立監査に保存し、[all-nine-quality-resources.csv](covertype-all-nine-quality-resources.csv)に全リソース・
精度行を記録している。

## 記録されたリソース

上表の時間は、選択checkpointが途中であっても同期した**10epoch全体の学習**秒数。
次は該当3seed jobの除外項目別平均秒数で、学習clockに含めていない。

| Method | Startup s | Qualification s | Preparation s | 10 dev evals s | Selected train eval s | Selected dev eval s | Heldout eval s |
|---|---|---|---|---|---|---|---|
| Dense | 1.841 | 0.518 | 0.080 | 7.523 | 1.500 | 0.752 | 0.789 |
| CST alpha=0.001 | 1.844 | 0.980 | 0.086 | 31.384 | 6.130 | 3.060 | 3.095 |
| CST alpha=0.01 | 1.847 | 2.591 | 0.092 | 39.101 | 7.715 | 3.852 | 3.826 |

学習時間にはminibatch全体のforward/backward/optimizer処理を含め、startup、
qualification、preparation、評価、生checkpoint・ファイルI/Oを除く。
評価時間にはforward・転送・指標計算を含む。同じ物理3090 cohort・runtimeだが、
方式は別々の逐次phase jobなので時間比較は探索的であり、HPO時間を主たる速度結果と
して扱わない。

| Method | Trainable scalars | Atoms | Allocated MiB mean [min,max] | Reserved MiB mean [min,max] |
|---|---|---|---|---|
| Dense | 1,112,071 | — | 67.05 [67.05, 67.05] | 88.00 [88.00, 88.00] |
| CST alpha=0.001 | 67,687 | 1048 | 52.12 [52.12, 52.12] | 60.00 [60.00, 60.00] |
| CST alpha=0.01 | 105,435 | 10485 | 53.12 [53.12, 53.12] | 62.00 [62.00, 62.00] |

学習可能数は実際の保存済み初期パラメータshapeの合計で、atom当たりpacked scalar4個と
通常の入出力重み・biasを含む。Buffer・optimizer状態は学習可能数に含めない。
メモリは同期学習直後に記録したepochごとのPyTorch allocated/reserved peakの最大値。
常駐train/development配列、モデル、optimizerを含み、reservedは前の評価のallocator
cacheを反映しうる。プロセスメモリでも、heldout評価だけのpeak測定でもない。

## 証拠と制限

- Consumer: `bfa58dce18da85f9d2cb3a791db6d6a5126805d7`; public torchcst: `97580582ed28c4e494bf5c2656bbe94a85c544c3`.
- Training-data SHA256: `010f150bab3a700fbcde1106daf21e83cf28fe9024732d29c25756144a07e416`; heldout: `7220f0d7282c83732c203325f523da180e4701d8a001647ea0eff4cdc0dbee4f`.
- All-nine freeze: `029a5e9dd7e8f71357ef49f9e31a22b26a945a94fa6c9cd8f301640dd8468030`.
- Strict training/seal audit: `ebfb2a99a973aef6fc68dde0b731d42e21b9d995b5711489126f67629be5870d`.
- Once-only assessment archive: `6949031e3d7a4a88ba15ff7af7108982d39a5637d89d4a06ea1079c247fd6760`.
- Independent prediction/metric audit: `6534b81fb590e23b49a14c5e810b24ac467d682c7221445ce33d6529d10cd1eb`.

9つすべての予測配列、共通ラベル・行ID、CE/accuracy/macro-F1/softmax確率による
macro-OvR-AUCを独立に再計算した。ReaderはGPU forwardを実行せず、heldout特徴配列や
HIGGSのtail/testを開いていない。評価1回はexit0で完了し、retry・推論の繰返しはない。
失敗したreader準備attempt01は保存している。直接module起動を想定したcommand guardで
予測を読む前に停止し、その後、検証済み凍結driverの`--mode assess` dispatchを許可した。
ソース・モデル・数値許容差は変えておらず、修正履歴をpresentation provenanceに保持する。

結論は固定タスク分割・構造・2密度・有限LR探索・10epoch予算・3seedに限定される。
一般的な最小alpha、最適化の収束、速度上の利点を示したり、追加実験を承認したりする
ものではない。HIGGSのprimary/comparison研究は別のまま維持する。
`paired-seeds.png`にはdevelopment学習曲線9本とepoch対応平均を、
`heldout-summary.png`には選択モデルだけを記録し、seed SDを信頼区間と区別している。

[外部証拠の規約](reproducibility.ja.md)に従い、
`covertype-transfer-20261106/stats-verified-through-phase17-01`、
`stats-heldout-independent-02`（`presentation-01`図を含む）、
`stats-covertype-result-note-03`、`assessment-after-all9-01`を、raw receipt/archiveと
失敗readerの来歴と共に保持する。公開する9行CSVは派生summaryであり、raw checkpointと
dataは外部に置く。Consumer manifest:
`8a3c45f7677001c66cc666009d3dfa4a9027d0e1a7571eb571032be75df59d48`、
外部975 manifest: `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`。
Assessment envelope: `7e4210589a17f8d3a2cdf96d679cd8731baab2710df4caebdae5df251f75f040`。
