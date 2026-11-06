# HIGGS正規化学習：失敗後のload契約修正

[English (canonical)](higgs-normalization-learning-recovery.md) | 日本語

**COMPLETED。範囲固定9条件のrecoveryは正常exit0で完了し、独立監査も合格した。**
Ea0 parentはexit1・未完了を維持し、保持300000と新375000 main更新の完了はこの明示的系譜に限る。
[結果と限界](higgs-depth-normalization-learning-results.ja.md)を参照。
[範囲固定の学習研究](higgs-depth-normalization-learning.ja.md)のadapter修正であり、
正規化recipe・選択規則・学習予算の変更ではない。凍結parent consumerは
`ea0d5701c45b188d619853ebd303767f224e36b8`、source manifestは
`68bf4e7eb80ed9c48ce4c27fc26bd549f55a309276ddd519d63d00dcf2860fb0`。
Parentの科学module27個と研究登録9個はすべて変更しない。修正は別module・目的branchに置く。

## 失敗と厳密なnative契約

Parentは全9条件のfresh8更新prefix再現を通過した。Dense-stock/RMS/LNとCST A104-stockを
所定の3epochまで学習し、main300000更新とprefix72更新を実行した。Dense3条件の
選択state評価は完了した。CST-stockはepoch2を選んだが、新consumerのliteral metadata
照合が、選択したTRAIN/development評価の前で止まった。Parentはexit1で、成功の9条件
summaryは存在しない。この事実と証拠をouter receipt
`18403283955ab2a4d6f5dbdfe9f2094cec54efe9aadbeb3bff15e5285275bcab`で保持する。
失敗結果を9条件完了に読み替えない。

Public975の`AtomState.set_extra_state`はformat/shape/generationの厳密なschemaを検証し、
load時に配置cache世代を意図的に進める。

\[
v_{\rm loaded}=\max(v_{\rm before}+1,v_{\rm serialized}).
\]

これは一時的placement/workspaceの無効化世代であり、Adam clockや学習parameterではない。
Fresh target0へsaved0をloadすると1になる。元consumerはserialized0との一致を誤って要求した。
元のcore comparatorとprefix/fresh0 gateは変更しない。本修正でgeneration差を一般的に隠したり、
public loaderを変更したりしない。

新selected-state loaderはsite2/4/6の**実際の**CST atom owner3個を列挙し、load前のversionを
記録する。既存public model/optimizer loaderを呼び、この3 metadata pathだけで上式を厳密に
要求する。全tensor・gradient・moment・clock・row map・shape・他metadataはbyte完全一致を
維持する。計算したgenerationは期待値の**照合用copy**にのみ置き、adapterは保存counterにも
live counterにも書き込まない。Denseに例外はない。Load後の評価による変更チェックは実際の
post-load全stateを基準にするので、その後のgeneration変更も引き続きerrorになる。

実parentの選択checkpointを用いた独立CPU診断は
`494bd33f7e9ccb8d1e06decc2a3636f878218883bbc849e00caa09a12409a346`、
checkpoint SHAは
`75780b8066c0543ed699f56a5bb814b9e53b3c64eb13538c5fb2f6d5356aa676`。
別の合成public975 CPU fixtureでも失敗を再現し、resumeとtamper拒否を確認した。
これはloader契約の証拠であり、新GPU/native oracleやrecovery評価成功ではない。

その後、実際の保持4条件admissionは
`7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7`でPASSし、
全75ファイルproofは
`dbe3081a726d6e175f880d483b298ce86319eb28cf4bdbca98090769553ea4fb`となった。
Statusは明示的に`passed-retained-fourcase-evidence-only-run-INCOMPLETE`を維持する。
Adapterは両reportを固定し、recovery前後に75 raw全件を再確認する。別の手書きadmissionは
拒否する。Tuple/listのJSON表現と動的optimizer store表現で停止したreader01/02も保持し、
raw/static契約や数値基準は変更していない。

## 明示的で有限のrecovery

Recovery GPU作業より前に、ownerは失敗parentの全75 archive member（receipt内74ファイルと
receipt自身）を回収・独立admitし、元source/data/runtimeと保存state/prediction/orderを
検証する。本修正のsourceを凍結し、新しい運用attemptを1回sealする。Parent rawはreadonlyを
維持し、入力rootと出力rootを分ける。自動retry、state8からの学習継続、元fileの暗黙修復はない。

1. 独立admit済みのDense3条件の完了結果をparent参照で再利用し、元producer identityと
   file hashを保持する。
2. CST-stockの選択epoch2についてのみ、修正loaderで全TRAIN100k/development50k評価を
   完了する。予測・proofはfresh recovery出力rootにだけ書く。選択DEV array byteがparentの
   epoch2予測と一致し、全state再load一致が成立することを要求する。このgateは残りの条件の
   学習**前**に通過しなければならない。
3. 未実行5条件だけを元順序で実行する。A104 pre-gain、post-gain、RMS、LN、A1048-stock。
   すべて同じ継承calibration/重み/orderのfresh0から、既存native `ordinary_step`で3epoch、
   75000更新ずつ学習する。残main予算375000、parentとrecoveryを合わせ675000更新。
   既に学習した300000更新の反復は、この修正では許可しない。

Parent prefix証拠はea0 identityと継承4f oracle identityを維持する。Parent runnerの
selected evaluatorだけを狭いscopeで置き換えることを、旧/新function名・source hash付きで
amendment provenanceへ明記し、`finally`で復元する。Model・optimizer・bare step・epoch loop・
元照合は変えない。新consumer/attempt identityと元producer参照は別である。完了したrecoveryも
この混合producer lineageを記録し、9条件を単一sourceで実行したとは見せない。

元の全DEV選択（epoch1..3の最小CE、同点は早いepoch）、data、calibration、device/runtime、
学習率、normalization epsilon、optimizer、保持、計時、解釈は同じ。Heldout、HPO、追加seed/
epoch/density、throughput/default採用、最小Atom則、formal NI主張は追加しない。
RootがGPUの唯一のoperatorを維持する。Source review成功だけでは実行済みにならず、
parent-admission証明を捏造してよいことにもならない。

## 固定foreground driver

Staging入力は厳密に11個。元のSHA固定済み学習入力10個に`parent-admission.json`を加え、
後者のSHAは上記の実reportに固定する。失敗parent75と旧4f151のraw treeは、別々のreadonly
パス引数で参照し、大きいpayloadを再stagingしない。Source/bootstrap/cache/runtime guardと、
明示的な失敗後identity/parent lineageを分ける。

```bash
python -B src/cst_research/experiments/higgs_normalization_learning_recovery_driver.py \
  --consumer-commit FULL_REVIEWED_CORRECTION_COMMIT \
  --parent-admission-sha256 7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7 \
  --prior-root READONLY_4F_ATTEMPT_ROOT --parent-root READONLY_FAILED_EA0_ATTEMPT_ROOT \
  --cache-root OWNED_CORRECTION_CACHE --output FRESH_CORRECTION_OUTPUT --timeout 7200
```

このcommandはreview・seal済みrecoveryの実entrypointを記録する。その有限attemptは完了したが、
追加実行や自動retryの許可ではない。Source・raw証拠・失敗parentは元identityへ固定して保持する。

Source/runtime guard、旧151 admission、失敗parent75 admission、data/calibration metadata/
device転送、厳密state admitを含む同期したnative restore2回の秒を記録する。各評価、cache済み
array読込、integrity check、予測保存も別記する。Child/preparation/selectedの合計は内訳を含み、
内訳と足し合わせない。Driverのarchive/cache bootstrapはここで個別計測せず、outer receiptの
walltimeにだけ含まれる。最初の選択TRAIN評価にはlazy native compile/placement初期化を含む
可能性があり、完全なadoption-timeを主張しない。最後にparent75全SHAと厳密なmember一覧を
再検証し、予期せず追加されたfileも拒否する。そのfileを削除・変更はしない。
