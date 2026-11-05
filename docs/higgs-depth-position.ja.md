# HIGGSの深さ・CST層位置pilot

[English (canonical)](higgs-depth-position.md) | 日本語

変更しないtorchcst `97580582ed28c4e494bf5c2656bbe94a85c544c3`を使う別研究consumerで、
開始状態はPLANNED。cohortは`higgs-depth-position-v1`で、旧HIGGS単一中間層cohortと
区別する。公開runtimeとGraph資格条件は変えない。CPU検証とデータ準備は、深いモデルの
GPU結果を意味しない。

## 固定構造と最初のscreen

**28→N→N→N→N→2**、N1024を使う。入出力はbias付きdense、正方hidden operatorは
3層ともbiasなし、ReLUは4個、全parameterを学習する。dense対照1条件と、正方層の
早期・中央・後期だけをCSTにする条件を比べ、曲線・勾配を確認してから3層全てをCSTに
する。CSTごとにalpha0.0001/0.001/0.01（divisor10000/1000/100）を使い、各CST層の
atomsは104/1048/10485。計13条件をseed17・3 full epochs、学習100k、batch4、FP32、
TF32off、全条件eagerで比較する。

通常parameterとpacked CSTのLRは一律0.0003、座標倍率は1。振幅減衰0.01とstockの
結合Polar更新を維持する。密度ごとの4LR grid、自動幅探索、gain校正、Graph資格条件の
迂回はしない。座標pilotが有効でも、採用には証拠レビューと新たな明示的source freezeが
必要。1 site専用private座標optimizerを3 siteモデルへ流用しない。

`torch.random.fork_rng`内で、入出力seedはseed+1/+2。正方dense層jはseed+100+j、
その位置のCST atomsはseed+1000+jを使う。他のどの位置を置換しても、共通dense重みと
同じ位置のCST初期値は条件間で一致する。公開CST初期化を維持する。活性・勾配の崩壊は
観測結果であり、無断でモデルを再スケールする理由にはしない。

## 新データとexposure境界

HIGGS TRAINの0始まり・半開区間を使う。

| 役割 | raw行 | 件数 |
|---|---|---|
| 学習 |[1000000,1100000)|100k|
| 新development |[1100000,1150000)|50k|
| 将来confirmation、別payload |[1150000,1200000)|50k|

raw source SHA256は
`ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698`。
母平均・母標準偏差は**新学習100kの特徴量だけ**でfloat64計算し、std floor1e-6を使い、
`(raw_float32-mean_float32)/std_float32`を全条件へ共通に適用する。新developmentや
将来confirmationをfitに使わない。

固定pilotファイル`train-development.npz`は
`x_train,y_train,x_development,y_development`の4配列だけを含み、SHA256は
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`。
manifest SHA256は
`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`。
別の`future-confirmation.npz`は**pilotへ転送せず、開かない**。公式testの
[10500000,11000000)は予約したままにする。

既知の過去retained先頭1Mと過去検証[10400000,10500000)を避ける。ただし、全く未接触の
raw特徴量ではない。過去の正規化fit[0,10400000)には新しい行も含まれた。manifestは
rawモデル・ラベル使用境界と、過去の集計特徴量exposureを分けて記録する。過去のCST研究
全てでのラベル使用について普遍的に未使用だとは主張しない。旧100k検証・confirmationは
新しい正式主張へ再利用しない。この適応的な粗いpilotに正式confirmationの主張はなく、
将来の主張には別途固定したprotocolと完了したexposure監査が必要。

## 実装とgate

`higgs_depth_study.py`が新構造・optimizer group・学習loopを担当する。外部975からは
公開CST構築と汎用`ordinary_step`、`evaluate`、設定・tensor helperだけを再利用する。
stock CSTAdamWが全CST tableを所有し、通常/dense parameterは通常AdamW groupに残す。
dense対照は通常AdamWを使う。旧単一中間indexの仮定やcohort名を流用しない。

各条件の学習前に、実device上の新network Y/dXと全parameter勾配を、公開CPU FP64の
`dense_weight`合成と比較する。続いて8 eager stepsと4-step保存/load後継続、有限性、
全parameterのoptimizer所有、時計を確認する。新しい104-atom条件も対象であり、過去の
L4資格確認を新device・新depth admissionの代用にしない。device名、capability、SM数、
memory、Torch/CUDAを残す。所有者が承認profileを使い、source準備をresource所有権と
みなさない。

初期値、全epoch、開発最良/最終の生model/optimizer/gradient checkpoint、新development
予測とraw row ID、全tensor hash、提示順hash、train/dev曲線、時間、allocated/reserved
peakを保存する。新学習の固定先頭256行で層別の活性RMS/max/ゼロ率を記録し、全parameter
勾配も同じ診断を行う。評価・probeでmodelとoptimizer tensorを変えない。各epochで全勾配と
moment時計を調べる。微小・ゼロ信号と負結果も証拠として残す。

選択は新development CE最小、同値なら早いepoch。時間はCUDA同期したeagerの完全学習stepで、
評価・診断・I/Oを除き、準備時間を分ける。memoryは常駐data/model/optimizerとprobeの影響を
含むが、GPU process総量ではない。精度科学優先の探索であり、device間速度採用の主張はしない。
このrunnerに新holdout evaluatorは存在しない。

確認したjobごとに1条件を実行し、consumer/外部manifestを検証し、出力は不変consumer sourceの
外へ置く。denseと単一位置の結果を確認してから全3層へ進む。追加seed、長期学習、大きいN、
新密度/LR探索は、次の明示的な設計として扱う。
