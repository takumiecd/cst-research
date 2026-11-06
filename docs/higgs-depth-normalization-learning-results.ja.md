# HIGGS深層正規化学習：範囲固定の結果

[English (canonical)](higgs-depth-normalization-learning-results.md) | 日本語

**明示したrecovery系譜による範囲固定の完了結果です。** 単一seed・3epochの
pilotでは、較正したRMS/LNがstock parameterizationで停滞したA104のCST中間3層を学習させました。
ただしDense同等ではなく、選択DEV AUCはRMS/LNが0.772532/0.775996、Dense-stockは0.802750です。
正規化なしのA1048-stockは0.792666まで学習しました。より大きいatom予算でLayerNormが必要だとも、
標準recipeにすべきだとも示していません。新しいlinear-atom研究の結果は含みません。

## 固定した範囲と選択

構造28→1024→1024→1024→1024→2は、biasなしの正方中間3層とDense入出力を持ち、層間はReLUです。
全model parameterはtrainableのままで、登録済みoptimizerが所有します。各条件seed17、TRAIN100000行を完全3epoch、B4、実Adam更新75000、
LR0.0003、stock q1、FP32/TF32off/eager、DEV50000行です。8更新診断と既存development観測後に
選んだ適応的・探索的followupで、DEVは既知であり、新たなheldout testではありません。Confirmation/
test採点、LR/epsilon/density探索、追加seed、formal noninferiority解析は行っていません。

元dataはUCI HIGGSの圧縮CSVで、SHA256は
`ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698`です。
Zero-based半開区間はTRAIN [1,000,000, 1,100,000)、DEV [1,100,000, 1,150,000)です。
14,285,659bytesの学習artifactはこの両cohortだけを含み、予約future confirmation
[1,150,000, 1,200,000)は含まず採点していません。入力標準化のfitは新TRAIN行だけであり、
network内部のRMS/LN介入とは別です。過去の集約特徴統計 [0, 10,400,000) にはこれら行が
既に含まれていたため、全研究履歴で未見だったとは主張せず、全過去のlabel利用についても
主張しません。正確なdata manifestは以下で識別します。

名目alpha0.0001はA104で約0.102N、名目alpha0.001はA1048で約1.023Nです。RMS/LNはaffineなし・
epsilon1e-5です。固定の正のoutput gainを最初のTRAIN256特徴行だけで、labelsを使わず逐次較正し、
Dense初期pre-ReLU RMSに合わせます。Post-gain条件だけはpost-ReLU RMSをtargetにします。
継承した固定較正で、gain探索はありません。全DEV CEが最小のepoch1..3を選び、同点は最初です。
全TRAINとDEVは同じ選択checkpointで評価し、TRAIN評価を選択に使いません。

\[
e^*=\arg\min_{e\in\{1,2,3\}}(\mathrm{CE}_{\rm dev}(\theta_e),e),\qquad
\Delta_{\rm CE}=\mathrm{CE}_{\rm dev}(\theta_{e^*})-\mathrm{CE}_{\rm train}(\theta_{e^*}).
\]

\[
K_{\rm Dense}=3N^2+31N+2,\qquad K_{\rm CST}=12A+31N+2.
\]

## 選択checkpointの精度

表はFP32 logitsによるproducer報告scalarを記録し、保存予測配列から独立に確認しています。許容差は変更せず、
CE3e-6、accuracy2e-7、AUC1e-12です。AUCはFP32 logit差から順位を求め、FP64 sigmoid変換で
代用しません。Accuracyは百分率です。CE gapは同じcheckpointでの両cohortの差です。

| Case | Epoch | TRAIN CE | DEV CE | DEV accuracy (%) | DEV AUC | DEV − TRAIN CE |
|---|---:|---:|---:|---:|---:|---:|
| Dense stock | 3 | 0.529078 | 0.544667 | 72.418 | 0.802750 | +0.015589 |
| Dense RMS | 2 | 0.536822 | 0.546350 | 71.786 | 0.796577 | +0.009528 |
| Dense LN | 2 | 0.539684 | 0.547439 | 71.600 | 0.795264 | +0.007755 |
| CST A104 stock | 2 | 0.691587 | 0.691394 | 52.960 | 0.500000 | -0.000193 |
| CST A104 pre-gain | 2 | 0.591317 | 0.590339 | 68.812 | 0.754402 | -0.000978 |
| CST A104 post-gain | 3 | 0.586959 | 0.587594 | 69.638 | 0.764557 | +0.000635 |
| CST A104 RMS | 2 | 0.566468 | 0.567374 | 69.986 | 0.772532 | +0.000906 |
| CST A104 LN | 2 | 0.561341 | 0.565227 | 70.402 | 0.775996 | +0.003886 |
| CST A1048 stock | 3 | 0.547023 | 0.551018 | 71.518 | 0.792666 | +0.003995 |

A104のLN/RMSはstockよりTRAIN・DEV両CEを下げましたが、TRAIN CE0.561341/0.566468は
Dense-stock0.529078より高いままです。Gap0.003886/0.000906だけでは、残る差を大きなtrain–dev gap
だけで説明できません。3epochで最適化収束や長期汎化は示せません。Dense側のRMS/LNもこの条件では
stockより低い精度でした。固定gainだけの条件はReLUの正の同次性と最終trainable Dense weightへの
吸収により理想実数上のglobal function familyを維持しますが、parameterization・gradient scale・
Adam epsilon・regularizationの対応は変わります。入力依存のRMS/LNはfunction mapを変えます。
したがって初期化だけ・conditioningだけの因果証明ではなく、最小atom数の主張でもありません。

## 記録された資源量

承認済みRTX4090 profile（CC8.9、128SM、device25262096384bytes）、Python3.12.9、
PyTorch2.11.0+cu126、CUDA12.6、Triton3.6.0、NumPy2.0.2、driver570.133.20です。
2attemptのsourceと計時由来を分けて保持します。学習秒は同期したeager complete-step loopの3epoch合計で、
準備・評価・diagnostics・checkpoint I/O・hashを含みません。Allocated/reservedは各epoch学習区間の
peakの最大値で、epoch後評価の前に取得します。Process全体やCPU oracleのmemoryではありません。
常駐TRAIN/DEV配列・model/optimizer/native bufferを含み、reserved cacheは前のDEV評価から残る場合が
あります。その後の選択評価のpeakは含まず、最適化Graph/sparse throughputの主張にもなりません。

| Case | Trainable scalars | Training, 3 epochs (s) | Allocated MiB | Reserved MiB | Case preparation (s) | 3 DEV evaluations (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense stock | 3,177,474 | 31.767 | 83.934 | 124.000 | 0.056 | 2.439 |
| Dense RMS | 3,177,474 | 36.630 | 83.689 | 124.000 | 0.060 | 3.019 |
| Dense LN | 3,177,474 | 37.551 | 83.689 | 124.000 | 0.055 | 3.092 |
| CST A104 stock | 32,994 | 96.255 | 37.557 | 64.000 | 0.063 | 12.232 |
| CST A104 pre-gain | 32,994 | 96.552 | 38.646 | 70.000 | 0.064 | 12.397 |
| CST A104 post-gain | 32,994 | 98.870 | 37.557 | 64.000 | 0.063 | 12.840 |
| CST A104 RMS | 32,994 | 104.610 | 37.589 | 64.000 | 0.068 | 13.894 |
| CST A104 LN | 32,994 | 103.421 | 37.590 | 64.000 | 0.064 | 13.960 |
| CST A1048 stock | 44,322 | 99.533 | 37.814 | 64.000 | 0.068 | 12.890 |

Parameter数は上式とadmit済み構造shapeから求め、固定gain3個はbuffer、RMS/LNに学習affineはありません。
Case準備には初期diagnosticsとstate0 admit/保存を含むため、別記diagnostic内訳を再加算しません。
詳細な別計上内訳は[resources.csv](assets/higgs-depth-normalization-learning/resources.csv)です。未記録のparent restore内訳は不明であり、0とは
しません。Recovery共通準備2.989716秒はsource/runtime/旧証拠/parentのguard・admissionとdata/device
準備を含む合計で、内訳と足しません。Driverのarchive/cache bootstrapはouter walltimeにだけ含まれ、
最初の選択TRAIN評価にlazy native compile/placementを含む可能性があります。完全なadoption-time比較や
hardwareだけのthroughput主張ではありません。

## Recovery系譜と証拠

Ea0 parentは4条件300000更新を学習し、全9条件のprefix再現（別枠72更新）を通過しました。Dense3条件の
選択評価は完了しましたが、CST-stockはepoch2選択後、復元時のlayout世代をliteral比較するreader側gateで
停止しました。Parentはexit1・未完了のままで、75 raw全件を変更せず保持しています。明示的な32428 adapterは
public975の`loaded=max(before_load+1,serialized)`を実CST owner3個にだけ要求し、他tensor/metadataは完全一致、
その後もpost-load全stateの評価時変更・reloadを完全照合します。最初にstockの選択評価を完了し、残り5条件だけを
fresh0から375000更新学習しました。結合main予算675000、新prefix0です。300000更新の再学習やparentの完了
への読み替えはありません。

結合監査は最初の実行で合格しました。以前のparent-readerのJSON tuple/list・optimizer store表現による停止と、
rootの独立AUC-readerでFP64 scoreを用いた誤りは、reader側の失敗として保存しています。固定学習・operator・
配列・数値許容差は変えていません。Qualificationは継承4fで、新たな独立native FP64 oracleを主張しません。
曲線表・図はadmit済みmetadataだけから導出し、tensor・予測・特徴・datasetの再読込、GPU操作、推論の追加実行は
行っていません。

[全9曲線](assets/higgs-depth-normalization-learning/all-nine-learning-curves.png) · [対応する正規化control](assets/higgs-depth-normalization-learning/matched-normalization-learning-curves.png) · [Gain/density曲線](assets/higgs-depth-normalization-learning/gain-density-learning-curves.png) · [PDF/CSV一覧](assets/higgs-depth-normalization-learning/README.ja.md)

| Item | SHA / revision |
|---|---|
| External public operators | `97580582ed28c4e494bf5c2656bbe94a85c544c3` |
| Inherited model/calibration/qualification | `4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b` |
| Parent training | `ea0d5701c45b188d619853ebd303767f224e36b8` |
| Explicit recovery | `32428aa8112ea6f0e206f0e1c12cda0206d920a0` |
| Recovery source manifest | `a2995731b62c601cadf7e7baafa86390bf54991bac7bd9f935e1dbe5e73008db` |
| Unchanged training recipe protocol | `c2e27deeed740cc299471a2fc1b7f13c03864a678a293d45f05986353911fd4d` |
| Recovery protocol | `80afd04557b70b13b82df9efb2104864562e9d5f638b9bdb8bae6b829e7be629` |
| Train/development artifact | `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3` |
| Data manifest | `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526` |
| Failed parent independent CPU admission | `7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7` |
| Combined independent CPU audit | `eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84` |
| Root independent science review | `ac19830c54f67610319a02fc334c74322a5ddf0077239cf11aefd476b8d2daa6` |
| Normal recovery outer receipt | `32f30113088d08e030522016062a8d5e3176a467a8d790715a8a024a1e2d0ef9` |
| Root all89-file release | `0bce347e75b7ed9ab666c97a1233028723ed918abd09aa8bd736fe937836741e` |
| Combined producer summary | `b1e4f20f6f83b37efa7a4de4fdda752dda04c57258e0ea28518ffe0fa21c0f33` |
| Derived metadata report | `b23a58b490c0d852fc18cd96928b27d5443ef667588e95f14ad9cb5089987b92` |

| Additional provenance | SHA256 |
|---|---|
| Parent source manifest | `68bf4e7eb80ed9c48ce4c27fc26bd549f55a309276ddd519d63d00dcf2860fb0` |
| Parent failed outer receipt | `18403283955ab2a4d6f5dbdfe9f2094cec54efe9aadbeb3bff15e5285275bcab` |
| Parent 75-file admission | `dbe3081a726d6e175f880d483b298ce86319eb28cf4bdbca98090769553ea4fb` |
| External source 975 manifest | `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931` |
| Inherited source 4f manifest | `aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53` |
| Inherited 151-file CPU admission | `89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8` |
| Inherited 151-file outer receipt | `ae0c2bbb7f000f5465c41b24e66c08223ffdd722a45382d70251db71f905c6ab` |
| Combined CPU reader | `83158c46134bdadc4d93b7136f2065af90e5effdb5cb545d0bdb17b832896c6f` |
| Frozen parent reader 03 | `e2d119153dc816c34cb6876f226a368a0337fc847418520092be29f6942caa0a` |
| Root derived 27-row review | `9c81c79b42e7088c887562d87d93dfd16397b2daec76854835a41436ddae56d3` |

## 再現と外部証拠

この文書checkoutではなく、上記の実測固定revisionを使います。次のcommandは実際の固定
entrypointの記録であり、自動再実行、追加GPU許可、既知の失敗parentを繰り返す指示ではありません。
元の入力10ファイルとrecoveryの11番目`parent-admission.json`は、source/data/runtime/proofの
固定hashと一致させます。[元recipe](higgs-depth-normalization-learning.ja.md)と
[明示的修正](higgs-normalization-learning-recovery.ja.md)を参照してください。

```bash
# Historical failed-parent entrypoint at ea0; not an instruction to retry it.
python -B src/cst_research/experiments/higgs_depth_normalization_learning_driver.py \
  --consumer-commit ea0d5701c45b188d619853ebd303767f224e36b8 \
  --prior-root READONLY_4F_ATTEMPT_ROOT --cache-root OWNED_PARENT_CACHE \
  --output FRESH_PARENT_OUTPUT --timeout 7200

# Explicit recovery entrypoint at 32428, requiring the exact admitted failed parent.
python -B src/cst_research/experiments/higgs_normalization_learning_recovery_driver.py \
  --consumer-commit 32428aa8112ea6f0e206f0e1c12cda0206d920a0 \
  --parent-admission-sha256 7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7 \
  --prior-root READONLY_4F_ATTEMPT_ROOT --parent-root READONLY_FAILED_EA0_ROOT \
  --cache-root OWNED_CORRECTION_CACHE --output FRESH_RECOVERY_OUTPUT --timeout 7200
```

[外部証拠規約](reproducibility.ja.md)に従い、`CST_RESEARCH_EVIDENCE_ROOT`の下の
`higgs-normalization-learning-recovery-20261106/evidence-index.json`を解決します。
外部indexは継承4f151、失敗parent75、正常recovery89、両outer receipt、独立parent/結合audit、
rootのscience/derived review、固定CPU reader（以前の失敗reader系譜を含む）、元の導出packを
不変artifactへ対応付けます。ローカル保存pathを含む場合があり、公開dataではありません。
文書統合でraw artifactを移動・再検証・公開しません。公開6図は元の導出bytesを維持し、公開3CSVは
改行だけを正規化します。元/公開hashは[provenance.json](assets/higgs-depth-normalization-learning/provenance.json)
に記録しています。以後のより大きいatom研究の結果や正規化必須という結論は含みません。
