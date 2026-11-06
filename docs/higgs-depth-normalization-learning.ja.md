# HIGGS正規化学習：範囲を固定した3 epoch追試

[English (canonical)](higgs-depth-normalization-learning.md) | 日本語 · [正確なJSON](../src/cst_research/studies/higgs-depth-normalization-learning-20261106.json)

**PREPARED。実装とCPUチェックのみ。新たなGPU実行・学習結果はまだない。** Study IDは
`higgs-depth-normalization-learning-20261106`。完了済み4fの8更新観測と過去f6/69の
development結果を**見た後**に選んだ、別の3epoch追試である。初動の動きの差が、
限られた学習・精度の差に続くかを見る。過去に確認したdevelopmentを意図的に再利用し、
新しいtestとは呼ばない。Primary HIGGS/DSTと完了済み8更新研究は変更しない。
有限比較後の自動採用・候補昇格・追加実行はない。

## 固定caseとrecipe

9caseを、同じ予約済みRTX4090/runtime上で以下の順に実行する。

| Order | Case | Square maps | Before each square-output ReLU |
|---:|---|---|---|
| 1 | dense-stock | Dense | None; unit gains |
| 2 | dense-rms | Dense | No-affine RMS, fixed Dense pre-ReLU RMS target |
| 3 | dense-ln | Dense | No-affine LN, fixed Dense pre-ReLU RMS target |
| 4 | cst104-stock | CST A104 ×3 | None; unit gains |
| 5 | cst104-gain | CST A104 ×3 | Fixed Dense pre-ReLU RMS target |
| 6 | cst104-postgain | CST A104 ×3 | Fixed Dense post-ReLU RMS target |
| 7 | cst104-rms | CST A104 ×3 | No-affine RMS, fixed Dense pre-ReLU RMS target |
| 8 | cst104-ln | CST A104 ×3 | No-affine LN, fixed Dense pre-ReLU RMS target |
| 9 | cst1048-stock | CST A1048 ×3 | None; unit gains |

構造は28→1024→1024→1024→1024→2。Site2/4/6に3つの**biasなし**正方写像、
site1/3/5/7にReLU、入出力は学習する通常Dense。A104はdivisor10000（α=.0001）、
A1048はdivisor1000（α=.001）で、既存の整数切捨てを使う。全caseをseed17で新規初期化し、
**3epoch×25000真のB4更新=75000更新**、総学習予算675000更新とする。
Epoch順列seedは117/118/119、既存`torch.randperm(100000)`を使い、全caseで順列を揃える。
A1048は別予算のanchorで、閾値検定ではない。

LR=.0003、q1のstock public975 CSTAdamW（Denseは通常AdamW）、β=(.9,.999)、
Adam ε=1e−8、通常WD=.01、amplitude WD=.01、raw atom WD=0、fused=True、
capturable=False、FP32/TF32 off/**eager**。全trainable parameterを学習する。
HPO、LR/ε/密度探索、optimizer変更、精度による早期停止はしない。
時間計測loopは既存の**観測なし**`ordinary_step`を呼ぶ。

各exampleの1024特徴について、各正方写像の後、ReLUの前で次を使う。

\[
R_\epsilon(z)=\frac{z}{\sqrt{\operatorname{mean}(z^2)+\epsilon}},\qquad
L_\epsilon(z)=\frac{z-\operatorname{mean}(z)}
{\sqrt{\operatorname{mean}((z-\operatorname{mean}(z))^2)+\epsilon}},
\qquad\epsilon=10^{-5}.
\]

学習するnorm affine/biasはない。既存の順次CPU-FP64 public `dense_weight()`較正で、
元の順序の先頭256件の**TRAIN特徴だけ**を使い、labelは使わない。正のgainは
[1e−6,1e6]の非学習FP32 buffer。目標とpostgain例外は表の通り固定する。
Signal/targetのzero・非有限値、範囲外gain、trainable変更はadmission停止とし、
clip、別ε、代替targetは使わない。

## データとidentity

既存TRAIN100k/development50k artifactのみを使う。SHAは
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`、manifestは
`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`。
Raw TRAIN行は[1000000,1100000)、developmentは[1100000,1150000)。前処理はこの
TRAINだけでfitし、過去の集約特徴統計の露出は引き続き開示する。
Future-confirmation[1150000,1200000)、元のHIGGS confirmation、official testの
payloadは除外する。評価を禁止しただけで「byteが一度も開かれていない」とは呼ばない。
今回はdeployment自体から当該payloadを除外する。

実測consumer `4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b` の既存24科学moduleを
byte単位で維持する。Manifestは
`aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53`。
正確な24ファイル一覧はこの実測85ファイルmanifestから取得し、現在checkoutから
再構成しない。[モデル契約](higgs-depth-normalization-diagnostic.ja.md)も参照。
Public sourceは引き続き`97580582ed28c4e494bf5c2656bbe94a85c544c3`、manifestは
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`。
**別の学習loop/driver/tests**を準備した。review済みsource commitとmanifestは
実行前に凍結する。継承model-contract identityと新学習study identityを分け、旧結果を
新しい結果へ読み替えない。

実行対象はNVIDIA GeForce RTX4090、CC8.9、128SM、25262096384 bytes、承認profile
`gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`。
実runtime pinは
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`。
Python3.12.9、Torch2.11.0+cu126/CUDA12.6、Triton3.6.0、NumPy2.0.2、
driver570.133.20である。Rootが唯一のGPU操作者で、L4 campaign/source/queueは変更しない。

## 継承qualificationと新loop admission

以前の**独立CPU生データadmission**は
`89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8`、
151ファイルadmissionは
`3d857e153906395c76cf516f0afeb54508bf532acfa375b99e0c69f6e916ab3a`、
outer receiptは
`ae0c2bbb7f000f5465c41b24e66c08223ffdd722a45382d70251db71f905c6ab`。
既存の不変raw証拠をその場所で参照し、2.64GBを再upload/copyしない。外部相対locatorは
`higgs-depth-normalization-diagnostic-20261106/evidence-index.json`。

これは**継承した4f qualification**と明記し、native/FP64 oracleを新たに実行したとは
しない。旧範囲の全9case normal Y/dX/全gradient、norm/packed-column/material、
8対4+resume+4、scalar-Adam、observer/nativeを継承する。独立retained比較はfinal8、
producerのliveチェックは合成全stepである。Denseを含むamplified componentwise失敗は
失敗のまま保存する。新epoch loop、指標/選択code、最終精度、throughputは未検証である。

3epoch学習をどのcaseも始める前に、以下を満たす。

1. Source/data/runtime/profileと24ファイルhashを検証する。新CPUチェックはepoch順序・
   counter・予算・欠落なし・reset、最初の最小CE tie、採点/保存、eval/train mode復帰、
   heldout path不在を扱う。Rootが新凍結sourceを独立reviewする。
2. **全9caseの新規zero-update model**を作り、特徴だけで較正する。Trainable tensor
   field、FP32 gain、通常IO、較正記録を、実際の旧4f **state0/calibration**と照合する。
   Study metadataが異なるcheckpointファイル全体ではなくtensor fieldを比較する。
   ローカル2.13で新規生成した値をproducer byteのoracleとしない。
3. 小さな新loop admissionとして、各caseで新規のbare `ordinary_step`に元と同じ先頭32
   TRAIN indexを与え、実際の旧4f **state1/2/4/8**とmodel/gradient/moment/clockの全tensor
   byteを照合する。このtask先頭再実行72更新は総学習675000更新と別で、合成更新でも
   FP64 qualification再実行でもない。小さな一致/hash結果を残し、破棄してfresh0へ
   戻す。**state8から続行しない。**

## 選択・fitting・resource証拠

各epoch後にdevelopment50k全件を評価し、**epoch1..3だけから最も早い最小development
CE**を選ぶ。Epoch0、case、LR選択はない。State0と全3epochのraw model/optimizer/
gradient/clock/orderを保持し、bestは選ばれたepochのSHA参照とする（tensor重複は不要）。

**同じ選択checkpoint**でTRAIN100k・development50k全件のFP32 logits、INT64 labels、
元の行順とprovenanceを保存する。選択epochのdev予測を再利用する場合は正確なcheckpoint/
予測hashで一致を証明し、できなければ学習clock外でselected devを別途評価する。
選択stateを再構築・reloadし、固定batching/runtimeでtrain/dev logit配列の完全byte一致を
必要とする。
CE/accuracy（argmax tieはclass0）/AUC/logitRMSを既存固定規約で独立再計算する。
同一固定stateのdevCE−trainCEとtrain/dev accuracy/AUCを報告する。オンラインtrain CEは
別のtraceであり、固定checkpointのtrain評価ではない。全9caseを必ず報告する。

9case全てのオンラインtrainCE/devCE/dev accuracy/AUCを、真のminibatch数と同期した
**観測なしcomplete-step**累積学習秒に対して重ね、Dense-stock/RMS/LN対照も含める。
Caseの`preparation_seconds`は構築・state0読込/照合・state0保存・初期診断を含む合計である。
別記`initial_diagnostic_seconds`はその内数で、合算しない。その他の準備・評価・保存の秒も
各scopeを確認し、重複した内訳を足し合わせない。
Preparation・calibration・admission・evaluation・checkpoint/hash秒を分け、計測stepへ
proposal observerを入れない。固定256 TRAIN特徴のactivation/RMS/ReLU energy probeと、最後の実minibatchで保持した
parameter gradient・m/v/clock統計を計測外で保持する。Epoch両端から実際の1更新proposal/
変位を推測しない。保存stateのepoch間移動を調べる場合は別scopeの診断と明記する。各epochの同期した学習loop直前にallocated/reserved peakをresetし、
checkpoint/評価より前に取得する。常駐train/dev配列・model・optimizer・native bufferを含み、
前の評価で確保したallocator予約領域は残り得る。process全体のpeakとは呼ばない。旧8更新の観測込みwalltime/memoryは別の診断scopeである。
同一device/runtimeの逐次実行にも順序・cache・熱の影響があるため、一般的throughput、
hardware因果、独立統計主張にはしない。

## 固定終了条件と解釈

Hard学習予算は9case、各3epoch/75000更新、総675000更新。Foreground wallclock
上限7200秒は**運用案**であり所要時間の予測ではない。Rootが実行前に実際の上限と
利用可能storage予算を固定する。Source/init/calibration/gate不一致、epochの非有限
loss/state/moment/gradient、ownership/clock異常、OOM/storage/timeout/transfer失敗で
batchを停止し、途中証拠を保存する。自動retry/fallbackはしない。有限値のまま学習が
停滞したcaseは有効な負の結果であり、決めた3epochを完了する。Rootがprimaryのため
中断した場合は未完了とする。成功時も9caseと回収・監査で終了し、**追加epoch・seed・
密度・ε/LR探索・候補昇格を自動実行しない**。

Gainだけのcaseは、正のgain、biasなし正方写像、ReLU、上限なしの学習Dense出力の
理想実数演算で、次を満たす。

\[
G=g_1g_2g_3>0,\qquad F_{\rm gain}(x;W_{\rm out},b_{\rm out})
=F_{\rm stock}(x;G W_{\rm out},b_{\rm out}).
\]

Gainだけならglobal function familyは変わらないが、gradient scale、parameterization/
Adam ε/正則化の対応とFP32軌道は変わるため、初期化だけの因果証明ではない。
RMS/LNは**入力依存の非線形写像**で、非zero εとLN centeringがあり、gainだけの恒等式を
正規化modelへ適用して同値とはできない。対応するDense norm対照で一般的正規化効果と
CST recipeの効果を切り分けるが、1seed・3epochではcentering単独の救済原因、内在容量、
最小α、scale則、正式NI/有意差/CI、default採用を証明しない。

## 準備済みの直接Hub adapter

新adapterのstaging入力は厳密に10個：`consumer-source-manifest.json`、
`original-source-manifest.json`、`original-source.tar.gz`、`train-development.npz`、
`data-manifest.json`、`runtime-pin.json`、`prior-full-CPU-admission.json`、
`prior-file-admission.json`、`prior-outer-receipt.json`、
`prior-launch-intent.json`。すべてSHA固定し、新consumer manifestは今後review済みの
完全commitへ束縛する。以前の151ファイルの結果treeはreadonlyパス引数で参照する。
remote親パスの相違により移せるのは小さい旧receipt/intentの2個だけで、厳密なbyte hashを
確認する。残る149個は元の相対result位置のままstream hashする。

review済みconsumer checkoutの`research-inputs/`へ上記入力を置いて実行する。

```bash
python -B src/cst_research/experiments/higgs_depth_normalization_learning_driver.py \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --prior-root PRIOR_ATTEMPT_ROOT --cache-root OWNED_COMPILE_CACHE \
  --output FRESH_ATTEMPT_OUTPUT --timeout 7200
```

これは実行手順であり、実行許可や完了結果ではない。timeout/失敗時はforeground子process
groupを回収し、実際の部分出力とSHA inventoryを保持する。自動resume/retryはしない。
state0/各epoch状態は1回だけ保持する（bestはSHA参照）。新証拠の見積りは合計2GiB未満。
実行前にownerがremote10GiB以上/local25GiB以上の空きを確認する。別deployment envelopeで
これら運用receiptを固定し、adapterはhardware確保やaccount/session cache変更を行わない。
