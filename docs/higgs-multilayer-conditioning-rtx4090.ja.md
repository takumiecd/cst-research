# HIGGS多層conditioning：独立RTX4090 cohort

[English (canonical)](higgs-multilayer-conditioning-rtx4090.md) | 日本語

**PLANNED / PREPARED、実行結果ではありません。**
[対応するA100の6条件設計](higgs-multilayer-conditioning.ja.md)を正確にRTX4090で反復します。
利用可能な4090の並行利用をuserが要求したためです。N=1024、H=8、A=4N=4096を固定し、
幅とdeviceを混同しません。N2048や追加介入はありません。各deviceで6モデルをfreshにpaired実行し、
新たなnative admissionを行います。A100/旧4090のstate・calibration・資格markerは再利用・改名しません。

## 変更しない科学的契約

`a43d6a938db7ff1bbf3522343cd35f10f3dcac1c` の元conditioning実装/driverの7ファイルを
byte単位で維持し、model構築・数値実行前にSHAを検査します。新sourceは4090 cohortのorchestrationと
厳格なforeground deploymentだけです。変更なしの `qualification.qualify_all`、`base.run_case`、
`base.calibration_features`、activations、budget helperを直接呼びます。
monkeypatch・属性override・数値runtimeコピー・Graph変更はありません。
raw model configurationのcohortは元の数値契約IDを保持し、新しいrun identity・runtime・protocolが
この独立4090実験を識別します。

| Case | Square maps | No-affine LN | Gains |
|---|---|---|---|
| dense-none | Dense | no |1|
| dense-bare-ln | Dense | yes |1|
| cst-none | CST | no |1|
| cst-fixed-gain | CST | no | fixed calibrated |
| cst-bare-ln | CST | yes |1|
| cst-ln-gain | CST | yes | fixed calibrated |

全modelは28→1024×8→2、biasなし正方写像7個、ReLU8個、学習可能なDense IOです。
seed17、従来初期化、eager FP32/TF32 off、B4、q1 AdamW/LR0.0003、betas(0.9,0.999)、
Adam epsilon1e−8、ordinary/amplitude WD0.01、raw Atom WD0を固定します。
同じmethod内で初期Atom・ordinary/IO tensorのstreamを共有します。
LNは1024 feature単位・nonaffine・epsilon1e−5です。CST gain両条件は、変更なしのCPU FP64
referenceで先頭256 TRAIN feature上の同じ初期Dense-stock pre-ReLU RMSへ順次合わせます。
正のFP32 gain[1e−6,1e6]を固定し、label/DEV・ゼロ/非有限fallback・clippingを使いません。
raw trainableは維持し、amplitude初期化の変更ではありません。

各条件で全100kの1 epoch＝25000真の更新、計150000です。実際の共有seed117 TRAIN順序を保存します。
更新0,1000,5000,10000,15000,20000,25000で全TRAIN100k/DEV50kを評価します。
選択は最終25000固定で、最小DEVではありません。最終restoreで予測配列/指標の完全一致と
変更なしのnative7-owner generation規則を要求し、他の全core byteを実POSTLOAD baselineと比較します。
HPO・early stop・tail/test・新規確認データはありません。従来の適応的DEV/feature露出を開示します。

全6条件のfresh資格検証がtask更新に先行します。全Y/dX/全11 parameter勾配を変更なしのmanual
FP64 LN/gain/CST reverseと公開chunked referenceに照合し、normal1/amplified1e24 cotangent、
既存のexact-zero/material/relative/elementwise gate、独立31-Atom quadratureとscalar Adam、
8ordinary対4+load+4、observed-first+7対8ordinaryのfullstate一致を検査します。
synthetic120更新はtask150000と別枠で、amplified勾配をoptimizerへ渡しません。
taskモデルはfresh0から再開します。許容差・material coverage・数値acceptanceは変更しません。

実際の最初のB4観測・勾配/m/v/clock・accepted-state FP32 denominator再構成・7時点予測・
固定256 forward probe・最終独立copy backward probeは元の範囲を維持します。
kernel中間値capture・中間raw全再生・epsilon因果主張は加えません。厳密な共通数式とguardはA100設計を参照してください。
固定gainのfunction-family等価性はNONE・biasなしReLU・無制限headのみで、epsilon付き中間LN+gainには拡張しません。

## 厳密なresource/source境界

RTX4090 runtime pin:
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`。
Python3.12.9/Torch2.11.0+cu126/CUDA12.6/Triton3.6.0/NumPy2.0.2、
driver570.133.20、NVIDIA GeForce RTX4090、CC8.9、128SM、device25262096384 bytesです。
A100/3090置換・runtime pin不一致を拒否し、compiler targetは8.9です。
operatorだけが承認Hub profileを確認・起動し、consumerはallocation/install/retryをせずforegroundで動きます。

外部公開sourceは `97580582ed28c4e494bf5c2656bbe94a85c544c3`、manifestは
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931` です。
TRAIN/DEV NPZは `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`、
data manifestは `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526` です。
正確なregular入力6個はconsumer-source-manifest.json、original-source-manifest.json、
original-source.tar.gz、train-development.npz、data-manifest.json、runtime-pin.jsonです。
将来/公式/元確認payloadは含めません。全commit済consumerファイルと全入力を検証します。

```bash
python -B -m cst_research.experiments.higgs_multilayer_conditioning_rtx4090_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/source-runtime-verified-cache \
  --output /path/to/fresh-diagnostic-output --timeout 7200
```

既存4090 public975 cacheは、外部archive・manifest・全runtime bindingが厳密一致するときのみ再利用できます。
中身はcompile sourceであり、task stateや資格ではありません。source byteを再検証し、
consumer pathをcommitごとに分離します。outputはfreshかつ別pathです。
timeoutは所有process groupを回収し、部分証拠を保持します。自動再実行/resumeはありません。

新cohort独立のraw cap6GiB・JSON cap64MiBです。remote immutable rawを保持し、compact proof/予測と
実空き容量で認可した圧縮archiveだけを回収するか、別pin済peer CPU証拠auditorをCUDA offで使用します。
local全展開copyは必須ではありません。時間はDIAGNOSTICで、準備/calibration/admission/観測/評価/I/Oを分離し、
包含成分は非加算です。block peakは評価前に測り、常駐配列/model/optimizer/nativebuffer/reserved cacheを含み、
process memoryではありません。device間比較は記述的で、hardwareだけの速度採用・Atomオーダ/capacity・LN必要性・
正式NIを主張しません。自動追試はありません。証拠locator:
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-multilayer-conditioning-rtx4090-20261106/`。

継承131ファイルのうちregistry-count12→13の1テストmetadata以外を全てbyte単位で維持し、
従来registry12ファイルも不変です。進行中の固定A100 source/証拠と旧A/B consumerには触れません。
