# Covertype転移研究（限定された探索範囲で完了）

[English](covertype-transfer.md) | **日本語**

低密度CSTの挙動が別タスクへ移るかを調べる限定的な探索研究である。固定済みHIGGS研究を
置き換えず、正式な非劣性・普遍的な最適alpha・公式benchmark再現・地理的汎化を主張しない。
有限RTX3090 campaignは、実測consumerbfaで9-model seal、厳密な独立raw audit、
別途admitしたheldout評価1回、保存予測の独立監査を経て**COMPLETED**となった。固定した
予算内ではCST両密度が全3 seedでDenseに劣った。
[結果・seedごとの資源・証拠](covertype-transfer-results.ja.md)を参照する。以下の固定designは
変更せず、統合をcampaignの再実行とは扱わない。

## 結果を見る前に固定するデータ

[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype)は581,012行、
54特徴（連続10・binary44）、7classを持つ。出典はBlackard, J. (1998)、
[Covertype, DOI10.24432/C50K5N](https://doi.org/10.24432/C50K5N)、CC BY4.0。
公式zip SHA256は
`89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d`、
内部covtype.data.gz SHAは
`614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`。

NumPy PCG64 seed20261106で全行をpermuteし、先頭100,000行をtrain、次50,000行を
checkpoint選択用development、その次50,000行を別のfuture heldoutとする。残りは使わない。
random row splitなので地理的に近い観測がpartitionをまたぐ可能性がある。
全ての過去研究に対しlabelがglobally unseenだったとは主張しない。

raw整数をFP32としてparseする。最初の連続10特徴だけ、選択したtrain行からfloat64で
平均・母標準偏差をfitし、stdに1e-6のfloorを置く。float64で変換してFP32へcastする。
44binary特徴は保存し、label1..7を0..6へ写す。各splitに元のrow IDを保存する。
機械的準備でfuture行を抽出してもfuture精度指標は計算しない。学習へ渡すのは
train-development.npzだけであり、future-heldout.npzはfamily seal後の明示評価までstagingしない。

準備済みdata pin:

| Artifact | SHA256 |
|---|---|
| train-development.npz | `010f150bab3a700fbcde1106daf21e83cf28fe9024732d29c25756144a07e416` |
| data-manifest.json | `4a6026b41b0abe9594a0320e6455dee17e3fc8ecddcb3aff97657a75b7ae4bcd` |
| 別のfuture-heldout.npz | `7220f0d7282c83732c203325f523da180e4701d8a001647ea0eff4cdc0dbee4f` |

## モデルと限定的な選択

54→1024→1024→7、ReLU2個、正方middleはbiasなし。IOはDenseのまま全層を学習する。
Dense、CST alpha.001（A1048）、alpha.01（A10485）を比較する。
変更なしの公開torchcst975演算とstock CSTAdamW、position倍率1を使い、座標LR pilotを
採用せず、private Graph adapterも使わない。既存linear/update Planは公開registryから
明示loadし、dispatcher guardは変更しない。ordinary AdamW decayは.01、raw atom decayは0、
polar amplitude decayは.01でradius policyを維持する。

全cellはB4、FP32、TF32off、eager。ordinary IO seedはseed+1/+2、middleはseed+3。
各epochのCPU randpermをseed+100+epoch−1とし、全method/LRで同じstreamを使う。
毎epoch100k行全てを提示し25k更新を行う。campaign全体でruntime/device profileを一つに固定する。

1. seed17、各methodをLR1e-4/3e-4/1e-3で初期化から3epoch。winnerが端なら
   対応するLR3e-5または3e-3を正確に一つだけ追加して3epoch。
2. 全methodで実際の上位2LRを初期化から10epoch再学習する。best-development CE最小、
   最早checkpoint epoch、完全tieなら低LRの順で選ぶ。epoch0は候補にせず3設定を固定する。
3. paired seed29/43/71で各methodを固定LRの10epoch学習し、同じdev-only規則で
   checkpointを選ぶ。9個全てのraw selected checkpoint/configをsealするまでfutureを評価しない。
4. 別assessorにseal JSONの正確なSHAを要求する。future bytesを開く前に全9モデルの
   source/config/file hash、finite state、optimizer clockを検証する。評価は探索・記述に限定する。

HPO/final cellはraw artifact manifestが完了したものだけ再利用する。partialや不確かなcellは
検査を要求して拒否し、自動再実行しない。stateごとにphaseを一つずつ実行する。

## Admissionと保持する証拠

各実GPUで学習前に、公開dense_weightで組み立てた独立CPU FP64 networkとfull Y/dX/
全parameter VJPを比較する。次に8step eager軌道と4step時点save/reload後を比較し、
parameter・全gradient・moment・clockを確認する。raw qualification tensorを保存する。
CPU smoke testだけではGPU admissionや学習精度の結果にならない。

raw admissionは全5parameterのmoment/gradientのshape・dtype・ownershipと、
CST model-side optimizer storeとの一致を要求する。
raw初期状態、全epoch、best/final checkpoint、全optimizer moment/clock/gradient、
logit/label/row ID、曲線とfile hashを保持する。CUDA同期したcomplete-step train時間と、
startup・qualification・preparation・evaluationを分離して記録し、学習時allocated/reserved peakも
保存する（process全体のmemoryではない）。kernel compileはadmission/preparationへ含め、
学習時間へ入れない。VMをまたぐHPO時間比較は探索として扱う。

CE・accuracy・macroF1・macro one-vs-rest AUCを記録する。CEはfloat64 log-sum-expを使う。
$p_{ik}=\operatorname{softmax}(z_i)_k$として、
$\mathrm{CE}=-n^{-1}\sum_i\log p_{i,y_i}$と
$\mathrm{macroAUC}=7^{-1}\sum_k\mathrm{AUC}(p_{:k},1[y=k])$を記録する。保存FP32 logitからfloat64
softmaxを計算し、classごとに外部975 binary_aucを使う。raw class logitを確率とみなしてrankしない。
同じselected checkpointでfull trainとdevelopmentを学習clock外に評価し、両方のpredictionと
CE差を保存する。再計算dev logit・label・row ID・metricは、保存したbest prediction/metadataと
array bytesで一致しなければならず、どちらの評価もselected model tensorを変更してはならない。
online epoch train lossは更新中の異なる状態の平均であり、同一状態の
generalization gapとは扱わない。

## Source review後の再現

review済みconsumerのclean Git archiveと、変更なしの外部
`97580582ed28c4e494bf5c2656bbe94a85c544c3`を使う。1188file manifest SHAは
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`。
依存先はjob内の隔離環境に置き、commit後にconsumer全体のmanifestを固定する。
state/outputはimmutable sourceの外へ置く。direct-script driverはimport前にbytecodeを無効化する。

```bash
python src/cst_research/experiments/covertype_driver.py --mode campaign \
  --consumer-commit "$CONSUMER_COMMIT" --consumer-manifest "$CONSUMER_MANIFEST" \
  --external-source "$TORCHCST975" --external-manifest "$EXTERNAL_MANIFEST" \
  --data "$DATA/train-development.npz" --data-manifest "$DATA/data-manifest.json" \
  --data-manifest-sha256 4a6026b41b0abe9594a0320e6455dee17e3fc8ecddcb3aff97657a75b7ae4bcd \
  --state "$CAMPAIGN_STATE" --stage screen --method dense
```

cst-1000/cst-100でもscreenし、全3methodをretuneした後freezeする。confirmは1回に
methodとseedを一つ指定し、sealは全9個を要求する。provision/submit機能は含めない。
GPU承認・確保・証拠回収・停止は所有者の責任とする。別assessorは`--mode assess`、
共通pin入力、`--state`、`--frozen-index`、`--frozen-index-sha256`、`--heldout`、fresh `--output`を使う。
それ以前のphaseにはfuture fileをstageしない。
